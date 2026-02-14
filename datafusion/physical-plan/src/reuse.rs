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

use std::sync::Arc;

use datafusion_common::{
    ParamValues, Result, exec_err,
    tree_node::{Transformed, TreeNode, TreeNodeRecursion},
};
use datafusion_physical_expr::{
    PhysicalExpr,
    expressions::{has_placeholders, resolve_expr_placeholders},
};

use crate::{ExecutionPlan, execution_plan::ReplacePhysicalExpr};

/// Wraps an [`ExecutionPlan`] that may contain placeholders, making
/// it re-executable, avoiding re-planning stage.
///
/// # Limitations
///
/// Plan does not support re-execution if it (OR):
///
/// * uses dynamic filters,
/// * represents a recursive query.
///
/// This invariant is not enforced by [`ReusableExecutionPlan`] itself, so it
/// must be checked by user.
///
pub struct ReusableExecutionPlan {
    binder: Binder,
    bound_plan: Option<Arc<dyn ExecutionPlan>>,
}

impl ReusableExecutionPlan {
    /// Make a new [`ReusableExecutionPlan`] bound to the passed `plan`.
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        let binder = Binder::new(plan);
        Self {
            binder,
            bound_plan: None,
        }
    }

    /// Return a ready to execution instance of [`ReusableExecutionPlan`],
    /// where placeholders are bound to the passed `params`.
    pub fn bind(&self, params: Option<&ParamValues>) -> Result<Self> {
        let bound_plan = self.binder.bind(params).map(Some)?;
        Ok(Self {
            binder: self.binder.clone(),
            bound_plan,
        })
    }

    /// Return an inner plan to execute.
    ///
    /// If this plan is a result of [`Self::bind`] call, then bound plan is returned.
    /// Otherwise, an initial plan is returned.
    pub fn plan(&self) -> Arc<dyn ExecutionPlan> {
        self.bound_plan
            .clone()
            .unwrap_or_else(|| Arc::clone(&self.binder.plan))
    }
}

impl From<Arc<dyn ExecutionPlan>> for ReusableExecutionPlan {
    fn from(plan: Arc<dyn ExecutionPlan>) -> Self {
        Self::new(plan)
    }
}

#[derive(Debug, Clone)]
struct NodeWithPlaceholders {
    /// The index of the node in the tree traversal.
    idx: usize,
    /// Positions of the placeholders among plan physical expressions.
    placeholder_idx: Vec<usize>,
}

impl NodeWithPlaceholders {
    /// Returns [`Some`] if passed `node` contains placeholders and must
    /// be resolved on binding stage.
    fn new(node: &Arc<dyn ExecutionPlan>, idx: usize) -> Option<Self> {
        let placeholder_idx = if let Some(iter) = node.physical_expressions() {
            iter.enumerate()
                .filter_map(|(i, expr)| {
                    if has_placeholders(&expr) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![]
        };

        if placeholder_idx.is_empty() {
            None
        } else {
            Some(Self {
                idx,
                placeholder_idx,
            })
        }
    }

    fn resolve(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        params: Option<&ParamValues>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(expr) = node.physical_expressions() else {
            return exec_err!("node {} does not support expressions export", node.name());
        };
        let mut exprs: Vec<Arc<dyn PhysicalExpr>> = expr.collect();
        for idx in self.placeholder_idx.iter() {
            exprs[*idx] = resolve_expr_placeholders(Arc::clone(&exprs[*idx]), params)?;
        }
        let Some(resolved_node) =
            node.with_physical_expressions(ReplacePhysicalExpr { exprs })?
        else {
            return exec_err!(
                "node {} does not support expressions replace",
                node.name()
            );
        };
        Ok(resolved_node)
    }
}

/// Helper to bound placeholders and reset plan nodes state.
#[derive(Clone)]
struct Binder {
    /// Created during [`Binder`] construction.
    /// This way we avoid runtime rebuild for expressions without placeholders.
    nodes_to_resolve: Arc<[NodeWithPlaceholders]>,
    plan: Arc<dyn ExecutionPlan>,
}

impl Binder {
    fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        let mut nodes_to_resolve = vec![];
        let mut cursor = 0;

        plan.apply(|node| {
            let idx = cursor;
            cursor += 1;
            if let Some(node) = NodeWithPlaceholders::new(node, idx) {
                nodes_to_resolve.push(node);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();

        Self {
            plan,
            nodes_to_resolve: nodes_to_resolve.into(),
        }
    }

    fn bind(&self, params: Option<&ParamValues>) -> Result<Arc<dyn ExecutionPlan>> {
        let mut cursor = 0;
        let mut resolve_node_idx = 0;
        Arc::clone(&self.plan)
            .transform_down(|node| {
                let idx = cursor;
                cursor += 1;
                if resolve_node_idx < self.nodes_to_resolve.len()
                    && self.nodes_to_resolve[resolve_node_idx].idx == idx
                {
                    // Note: `resolve` replaces plan expressions, which also resets a plan state.
                    let resolved_node =
                        self.nodes_to_resolve[resolve_node_idx].resolve(&node, params)?;
                    resolve_node_idx += 1;
                    Ok(Transformed::yes(resolved_node))
                } else {
                    // Reset state.
                    Ok(Transformed::yes(node.reset_state()?))
                }
            })
            .map(|tnr| tnr.data)
    }
}
