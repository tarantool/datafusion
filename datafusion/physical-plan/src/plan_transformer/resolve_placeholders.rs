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

//! Rule to resolve placeholders in the physical plan with actual values.

use std::sync::Arc;

use datafusion_common::{
    ParamValues, Result, exec_err,
    tree_node::{Transformed, TreeNode},
};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{
    PhysicalExpr,
    expressions::{Literal, PlaceholderExpr, has_placeholders},
    simplifier::const_evaluator::simplify_const_expr,
};

use crate::{ExecutionPlan, plan_transformer::ExecutionTransformationRule};

/// A transformation rule that replaces [`PlaceholderExpr`] with actual values.
///
/// This rule is applied to the physical plan when actual parameter values are provided in the
/// [`TaskContext`]. It traverses the plan and replaces any placeholders found in physical
/// expressions with their corresponding literal values.
#[derive(Debug, Clone, Default)]
pub struct ResolvePlaceholdersRule {}

impl ResolvePlaceholdersRule {
    /// Create a new [`ResolvePlaceholdersRule`].
    pub fn new() -> Self {
        Self {}
    }
}

impl ExecutionTransformationRule for ResolvePlaceholdersRule {
    fn name(&self) -> &str {
        "ResolvePlaceholders"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn matches(&self, node: &Arc<dyn ExecutionPlan>) -> Result<bool> {
        let Some(exprs) = node.physical_expressions() else {
            return Ok(false);
        };

        for expr in exprs {
            if has_placeholders(&expr) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn rewrite(
        &self,
        node: Arc<dyn ExecutionPlan>,
        ctx: &TaskContext,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        let Some(param_values) = ctx.param_values() else {
            return Ok(Transformed::no(node));
        };

        let Some(exprs) = node.physical_expressions() else {
            return exec_err!("no physical expressions found");
        };

        let new_exprs = exprs
            .map(|expr| resolve_expr_placeholders(expr, param_values))
            .collect::<Result<Vec<_>>>()?;

        match node.with_physical_expressions(new_exprs.into())? {
            Some(transformed_plan) => Ok(Transformed::yes(transformed_plan)),
            None => exec_err!("failed to rewrite execution plan"),
        }
    }
}

/// Resolves [`PlaceholderExpr`] in the physical expression using the provided [`ParamValues`].
pub fn resolve_expr_placeholders(
    expr: Arc<dyn PhysicalExpr>,
    param_values: &ParamValues,
) -> Result<Arc<dyn PhysicalExpr>> {
    let expr = expr.transform_up(|node| {
        let Some(placeholder) = node.as_any().downcast_ref::<PlaceholderExpr>() else {
            return Ok(Transformed::no(node));
        };

        if let Some(ref field) = placeholder.field {
            let scalar = param_values.get_placeholders_with_values(&placeholder.id)?;
            let value = scalar.value.cast_to(field.data_type())?;
            let literal = Literal::new_with_metadata(value, scalar.metadata);
            Ok(Transformed::yes(Arc::new(literal)))
        } else {
            Ok(Transformed::no(node))
        }
    })?;

    if expr.transformed {
        simplify_const_expr(expr.data).map(|t| t.data)
    } else {
        Ok(expr.data)
    }
}
