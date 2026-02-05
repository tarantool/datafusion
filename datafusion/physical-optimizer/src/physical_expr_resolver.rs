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

//! [`PhysicalExprResolver`] ensures that the physical plan is prepared for placeholder resolution
//! by wrapping it in a [`TransformPlanExec`] with a [`ResolvePlaceholdersRule`] if the plan
//! contains any unresolved placeholders. The actual resolution happens during execution.

use std::sync::Arc;

use datafusion_common::{
    Result,
    config::ConfigOptions,
    tree_node::{Transformed, TreeNode},
};
use datafusion_physical_plan::{
    ExecutionPlan,
    plan_transformer::{ResolvePlaceholdersRule, TransformPlanExec},
};

use crate::PhysicalOptimizerRule;

/// The phase in which the [`PhysicalExprResolver`] rule is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalExprResolverPhase {
    /// Optimization that happens before most other optimizations.
    /// This optimization removes all [`TransformPlanExec`] execution plans from the plan
    /// tree.
    Pre,
    /// Optimization that happens after most other optimizations.
    /// This optimization checks for the presence of placeholders in the optimized plan, and if
    /// they are present, wraps the plan in a [`TransformPlanExec`] with a [`ResolvePlaceholdersRule`].
    Post,
}

/// Physical optimizer rule that prepares the plan for placeholder resolution during execution.
#[derive(Debug)]
pub struct PhysicalExprResolver {
    phase: PhysicalExprResolverPhase,
}

impl PhysicalExprResolver {
    /// Creates a new [`PhysicalExprResolver`] optimizer rule that runs in the pre-optimization
    /// phase. In this phase, the rule removes any existing [`TransformPlanExec`] from the
    /// plan tree.
    pub fn new() -> Self {
        Self {
            phase: PhysicalExprResolverPhase::Pre,
        }
    }

    /// Creates a new [`PhysicalExprResolver`] optimizer rule that runs in the post-optimization
    /// phase. In this phase, the rule wraps the physical plan in a [`TransformPlanExec`] with a
    /// [`ResolvePlaceholdersRule`] if the plan contains any unresolved placeholders.
    pub fn new_post_optimization() -> Self {
        Self {
            phase: PhysicalExprResolverPhase::Post,
        }
    }
}

impl Default for PhysicalExprResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for PhysicalExprResolver {
    fn name(&self) -> &str {
        match self.phase {
            PhysicalExprResolverPhase::Pre => "PhysicalExprResolver",
            PhysicalExprResolverPhase::Post => "PhysicalExprResolver(Post)",
        }
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match self.phase {
            PhysicalExprResolverPhase::Pre => plan
                .transform_up(|plan| {
                    if let Some(plan) = plan.as_any().downcast_ref::<TransformPlanExec>()
                    {
                        Ok(Transformed::yes(Arc::clone(plan.input())))
                    } else {
                        Ok(Transformed::no(plan))
                    }
                })
                .map(|t| t.data),
            PhysicalExprResolverPhase::Post => {
                if let Some(transformer) =
                    plan.as_any().downcast_ref::<TransformPlanExec>()
                {
                    let resolves_placeholders =
                        transformer.has_rule::<ResolvePlaceholdersRule>();

                    if resolves_placeholders {
                        Ok(plan)
                    } else {
                        transformer
                            .add_rule(Box::new(ResolvePlaceholdersRule::new()))
                            .map(|r| Arc::new(r) as Arc<_>)
                    }
                } else {
                    let transformer = TransformPlanExec::try_new(
                        plan,
                        vec![Box::new(ResolvePlaceholdersRule::new())],
                    )?;

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
