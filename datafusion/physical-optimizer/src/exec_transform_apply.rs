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

//! [`ExecutionTransformationApplier`] ensures that the required execution transformations
//! are applied to the physical plan.

use std::{borrow::Cow, sync::Arc};

use datafusion_common::{
    Result,
    config::ConfigOptions,
    tree_node::{Transformed, TreeNode},
};
use datafusion_physical_plan::{
    ExecutionPlan,
    plan_transformer::{ExecutionTransformationRule, TransformPlanExec},
};

use crate::PhysicalOptimizerRule;

/// The phase in which the [`ExecutionTransformationApplier`] rule is applied.
#[derive(Debug)]
pub enum ExecutionTransformationApplierPhase {
    /// Optimization that happens before most other optimizations.
    /// This optimization removes all [`TransformPlanExec`] execution plans from the plan
    /// tree.
    Pre,
    /// Optimization that happens after most other optimizations.
    /// This optimization checks if `rule` requires to transform the plan and wraps the plan with
    /// [`TransformPlanExec`] if it so, or adds rule to the existing transformation node.
    Post {
        rule: Arc<dyn ExecutionTransformationRule>,
    },
}

/// Physical optimizer rule that wraps the plan with a certain execution-stage transformation.
#[derive(Debug)]
pub struct ExecutionTransformationApplier {
    phase: ExecutionTransformationApplierPhase,
    name: Cow<'static, str>,
}

impl ExecutionTransformationApplier {
    /// Creates a new [`ExecutionTransformationApplier`] optimizer rule that runs in the
    /// pre-optimization phase.
    pub fn new() -> Self {
        Self {
            phase: ExecutionTransformationApplierPhase::Pre,
            name: Cow::Borrowed("ExecutionTransformationApplier"),
        }
    }

    /// Creates a new [`ExecutionTransformationApplier`] optimizer rule that runs in the
    /// post-optimization phase.
    pub fn new_post_optimization(rule: Arc<dyn ExecutionTransformationRule>) -> Self {
        let name = format!("ExecutionTransformationApplier({})", rule.name());
        Self {
            phase: ExecutionTransformationApplierPhase::Post { rule },
            name: name.into(),
        }
    }
}

impl Default for ExecutionTransformationApplier {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for ExecutionTransformationApplier {
    fn name(&self) -> &str {
        &self.name
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &self.phase {
            ExecutionTransformationApplierPhase::Pre => plan
                .transform_up(|plan| {
                    if let Some(plan) = plan.as_any().downcast_ref::<TransformPlanExec>()
                    {
                        Ok(Transformed::yes(Arc::clone(plan.input())))
                    } else {
                        Ok(Transformed::no(plan))
                    }
                })
                .map(|t| t.data),
            ExecutionTransformationApplierPhase::Post { rule } => {
                if let Some(transformer) =
                    plan.as_any().downcast_ref::<TransformPlanExec>()
                {
                    let has_rule = transformer.has_dyn_rule(rule);
                    if has_rule {
                        // Rule is already applied.
                        Ok(plan)
                    } else {
                        transformer
                            .add_rule(Arc::clone(rule))
                            .map(|r| Arc::new(r) as Arc<_>)
                    }
                } else {
                    let transformer =
                        TransformPlanExec::try_new(plan, vec![Arc::clone(rule)])?;
                    if transformer.plans_to_transform() > 0 {
                        Ok(Arc::new(transformer))
                    } else {
                        Ok(Arc::clone(transformer.input()))
                    }
                }
            }
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}
