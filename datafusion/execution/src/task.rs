// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

use crate::{
    config::SessionConfig,
    memory_pool::MemoryPool,
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
};
use datafusion_common::ParamValues;
use datafusion_expr::{AggregateUDF, ScalarUDF, WindowUDF};

/// [`TaskContext`] shared state across forks.
#[derive(Debug)]
pub struct TaskContextSharedState {
    /// Session Id
    session_id: String,
    /// Optional Task Identify
    task_id: Option<String>,
    /// Session configuration
    session_config: SessionConfig,
    /// Scalar functions associated with this task context
    #[allow(dead_code)]
    scalar_functions: HashMap<String, Arc<ScalarUDF>>,
    /// Aggregate functions associated with this task context
    #[allow(dead_code)]
    aggregate_functions: HashMap<String, Arc<AggregateUDF>>,
    /// Window functions associated with this task context
    #[allow(dead_code)]
    window_functions: HashMap<String, Arc<WindowUDF>>,
    /// Runtime environment associated with this task context
    runtime: Arc<RuntimeEnv>,
}

impl Default for TaskContextSharedState {
    fn default() -> Self {
        let runtime = RuntimeEnvBuilder::new()
            .build_arc()
            .expect("default runtime created successfully");
        // Create a default task context shared state, mostly useful for testing.
        Self {
            session_id: "DEFAULT".to_string(),
            task_id: None,
            session_config: SessionConfig::new(),
            scalar_functions: HashMap::new(),
            aggregate_functions: HashMap::new(),
            window_functions: HashMap::new(),
            runtime,
        }
    }
}

/// Task Execution Context
///
/// A [`TaskContext`] contains the state required during a single query's
/// execution. Please see the documentation on [`SessionContext`] for more
/// information.
///
/// [`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html
#[derive(Debug, Default)]
pub struct TaskContext {
    /// State shared between forks.
    shared_state: Arc<TaskContextSharedState>,
    /// Param values for physical placeholders.
    param_values: Option<ParamValues>,
    /// Metrics associated with a execution plan address.
    /// std mutex is used because too concurrent access is not assumed.
    metrics: std::sync::Mutex<HashMap<usize, ExecutionPlanMetricsSet>>,
    /// Session plans state by an execution plan address.
    /// Resources that shared across execution partitions.
    plan_state: std::sync::Mutex<HashMap<usize, Arc<dyn PlanState>>>,
}

/// Generic plan state.
pub trait PlanState: Debug + Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

fn plan_addr(plan: &dyn Any) -> usize {
    plan as *const _ as *const () as usize
}

impl TaskContext {
    /// Create a new [`TaskContext`] instance.
    ///
    /// Most users will use [`SessionContext::task_ctx`] to create [`TaskContext`]s
    ///
    /// [`SessionContext::task_ctx`]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html#method.task_ctx
    pub fn new(
        task_id: Option<String>,
        session_id: String,
        session_config: SessionConfig,
        scalar_functions: HashMap<String, Arc<ScalarUDF>>,
        aggregate_functions: HashMap<String, Arc<AggregateUDF>>,
        window_functions: HashMap<String, Arc<WindowUDF>>,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        Self {
            shared_state: Arc::new(TaskContextSharedState {
                task_id,
                session_id,
                session_config,
                scalar_functions,
                aggregate_functions,
                window_functions,
                runtime,
            }),
            param_values: Default::default(),
            metrics: Default::default(),
            plan_state: Default::default(),
        }
    }

    /// Fork a task context.
    ///
    /// Forked context contains the same:
    /// * session related attributes (id, udfs, etc),
    /// * runtime environment
    ///
    /// But an empty:
    /// * params
    /// * metrics
    /// * plan state
    ///
    pub fn fork(&self) -> Self {
        Self {
            shared_state: Arc::clone(&self.shared_state),
            param_values: Default::default(),
            metrics: Default::default(),
            plan_state: Default::default(),
        }
    }

    /// Return the SessionConfig associated with this [TaskContext]
    pub fn session_config(&self) -> &SessionConfig {
        &self.shared_state.session_config
    }

    /// Return the `session_id` of this [TaskContext]
    pub fn session_id(&self) -> String {
        self.shared_state.session_id.clone()
    }

    /// Return the `task_id` of this [TaskContext]
    pub fn task_id(&self) -> Option<String> {
        self.shared_state.task_id.clone()
    }

    /// Return the [`MemoryPool`] associated with this [TaskContext]
    pub fn memory_pool(&self) -> &Arc<dyn MemoryPool> {
        &self.shared_state.runtime.memory_pool
    }

    /// Return the [RuntimeEnv] associated with this [TaskContext]
    pub fn runtime_env(&self) -> Arc<RuntimeEnv> {
        Arc::clone(&self.shared_state.runtime)
    }

    /// Return param values associated with thix [`TaskContext`].
    pub fn param_values(&self) -> &Option<ParamValues> {
        &self.param_values
    }

    /// Update the param values.
    pub fn with_param_values(mut self, param_values: ParamValues) -> Self {
        self.param_values = Some(param_values);
        self
    }

    /// Return plan metrics by execution plan addr if some.
    pub fn plan_metrics(&self, plan: &dyn Any) -> Option<MetricsSet> {
        let addr = plan_addr(plan);
        self.metrics
            .lock()
            .unwrap()
            .get(&addr)
            .map(|m| m.clone_inner())
    }

    /// Associate metrics with execution plan addr or return existed metric set.
    /// Execution plan should register metrics in `execute` using it to have an ability
    /// to display it in the future.
    pub fn get_or_register_metric_set(&self, plan: &dyn Any) -> ExecutionPlanMetricsSet {
        let addr = plan_addr(plan);
        let mut metrics = self.metrics.lock().unwrap();
        if let Some(metric_set) = metrics.get(&addr) {
            metric_set.clone()
        } else {
            let metric_set = ExecutionPlanMetricsSet::new();
            metrics.insert(addr, metric_set.clone());
            metric_set
        }
    }

    /// Associate metrics with execution plan addr or return existed metric set.
    /// If there is no associated metric set uses provided callback to create
    /// default set.
    pub fn get_or_register_metric_set_with_default(
        &self,
        plan: &dyn Any,
        default_set: impl Fn() -> ExecutionPlanMetricsSet,
    ) -> ExecutionPlanMetricsSet {
        let addr = plan_addr(plan);
        let mut metrics = self.metrics.lock().unwrap();
        if let Some(metric_set) = metrics.get(&addr) {
            metric_set.clone()
        } else {
            let metric_set = default_set();
            metrics.insert(addr, metric_set.clone());
            metric_set
        }
    }

    /// Get state for specific plan or register a new state.
    pub fn get_or_register_plan_state<F>(
        &self,
        plan: &dyn Any,
        f: F,
    ) -> Arc<dyn PlanState>
    where
        F: FnOnce() -> Arc<dyn PlanState>,
    {
        let addr = plan_addr(plan);
        let mut plan_state = self.plan_state.lock().unwrap();
        if let Some(state) = plan_state.get(&addr) {
            Arc::clone(state)
        } else {
            let state = f();
            plan_state.insert(addr, Arc::clone(&state));
            state
        }
    }
}

/// Helps to build a [`TaskContext`].
#[derive(Default)]
pub struct TaskContextBuilder {
    session_id: Option<String>,
    task_id: Option<String>,
    session_config: Option<SessionConfig>,
    scalar_functions: HashMap<String, Arc<ScalarUDF>>,
    aggregate_functions: HashMap<String, Arc<AggregateUDF>>,
    window_functions: HashMap<String, Arc<WindowUDF>>,
    runtime: Option<Arc<RuntimeEnv>>,
    param_values: Option<ParamValues>,
    metrics: HashMap<usize, ExecutionPlanMetricsSet>,
    plan_state: HashMap<usize, Arc<dyn PlanState>>,
}

impl TaskContextBuilder {
    /// Make a new [`TaskContextBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a session id.
    pub fn with_session_id(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id;
        self
    }

    /// Set a task id.
    pub fn with_task_id(mut self, task_id: Option<String>) -> Self {
        self.task_id = task_id;
        self
    }

    pub fn with_session_config(mut self, session_config: Option<SessionConfig>) -> Self {
        self.session_config = session_config;
        self
    }

    /// Set scalar functions.
    pub fn with_scalar_functions(
        mut self,
        scalar_functions: HashMap<String, Arc<ScalarUDF>>,
    ) -> Self {
        self.scalar_functions = scalar_functions;
        self
    }

    /// Set aggregate functions.
    pub fn with_aggregate_functions(
        mut self,
        aggregate_funtions: HashMap<String, Arc<AggregateUDF>>,
    ) -> Self {
        self.aggregate_functions = aggregate_funtions;
        self
    }

    /// Set window functions.
    pub fn with_window_functions(
        mut self,
        window_functions: HashMap<String, Arc<WindowUDF>>,
    ) -> Self {
        self.window_functions = window_functions;
        self
    }

    /// Set a runtime.
    pub fn with_runtime(mut self, runtime: Option<Arc<RuntimeEnv>>) -> Self {
        self.runtime = runtime;
        self
    }

    /// Set param values.
    pub fn with_param_values(mut self, param_values: Option<ParamValues>) -> Self {
        self.param_values = param_values;
        self
    }

    /// Set metrics.
    pub fn with_metrics(
        mut self,
        metrics: HashMap<usize, ExecutionPlanMetricsSet>,
    ) -> Self {
        self.metrics = metrics;
        self
    }

    /// Set a plan state.
    pub fn with_plan_state(
        mut self,
        plan_state: HashMap<usize, Arc<dyn PlanState>>,
    ) -> Self {
        self.plan_state = plan_state;
        self
    }

    /// Build a task context.
    pub fn build(self) -> TaskContext {
        let Self {
            task_id,
            session_id,
            session_config,
            scalar_functions,
            aggregate_functions,
            window_functions,
            runtime,
            param_values,
            metrics,
            plan_state,
        } = self;

        let shared_state = TaskContextSharedState {
            session_id: session_id.unwrap_or("DEFAULT".to_string()),
            task_id,
            session_config: session_config.unwrap_or(SessionConfig::new()),
            scalar_functions,
            aggregate_functions,
            window_functions,
            runtime: runtime.unwrap_or_else(|| {
                RuntimeEnvBuilder::new()
                    .build_arc()
                    .expect("default runtime created successfully")
            }),
        };

        TaskContext {
            shared_state: Arc::new(shared_state),
            param_values,
            metrics: std::sync::Mutex::new(metrics),
            plan_state: std::sync::Mutex::new(plan_state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::{
        config::{ConfigExtension, ConfigOptions, Extensions},
        extensions_options, Result,
    };

    extensions_options! {
        struct TestExtension {
            value: usize, default = 42
        }
    }

    impl ConfigExtension for TestExtension {
        const PREFIX: &'static str = "test";
    }

    #[test]
    fn task_context_extensions() -> Result<()> {
        let runtime = Arc::new(RuntimeEnv::default());
        let mut extensions = Extensions::new();
        extensions.insert(TestExtension::default());

        let mut config = ConfigOptions::new().with_extensions(extensions);
        config.set("test.value", "24")?;
        let session_config = SessionConfig::from(config);

        let task_context = TaskContext::new(
            Some("task_id".to_string()),
            "session_id".to_string(),
            session_config,
            HashMap::default(),
            HashMap::default(),
            HashMap::default(),
            runtime,
        );

        let test = task_context
            .session_config()
            .options()
            .extensions
            .get::<TestExtension>();
        assert!(test.is_some());

        assert_eq!(test.unwrap().value, 24);

        Ok(())
    }
}
