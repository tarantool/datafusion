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

//! Placeholder expression

use std::{
    any::Any,
    fmt::{self, Formatter},
    hash::Hasher,
    sync::Arc,
};

use arrow_schema::{DataType, Schema};
use datafusion_common::{
    exec_err, plan_err,
    tree_node::{Transformed, TreeNode, TreeNodeRewriter},
    ParamValues, Result,
};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr_common::physical_expr::{down_cast_any_ref, PhysicalExpr};
use std::hash::Hash;

use crate::expressions::lit;

/// Physical plan placeholders.
/// Never are executed, replaced with scalar values before execution.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PlaceholderExpr {
    // Placeholder id, e.g. $1 or $a.
    id: String,
    // Derived from expression where placeholder is met.
    data_type: DataType,
}

impl PlaceholderExpr {
    pub fn new(id: String, data_type: DataType) -> Self {
        Self { id, data_type }
    }
}

pub fn placeholder<I: Into<String>>(id: I, data_type: DataType) -> Arc<dyn PhysicalExpr> {
    Arc::new(PlaceholderExpr::new(id.into(), data_type))
}

impl fmt::Display for PlaceholderExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl PhysicalExpr for PlaceholderExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.data_type.clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        // By default placeholders are nullable, non nullable placeholders
        // can be supported in the future.
        Ok(true)
    }

    fn evaluate(&self, _batch: &arrow_array::RecordBatch) -> Result<ColumnarValue> {
        exec_err!("placeholders are not supposed to be evaluated")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        assert!(children.is_empty());
        Ok(self)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut s = state;
        self.id.hash(&mut s);
    }
}

impl PartialEq<dyn Any> for PlaceholderExpr {
    /// Comparing IDs.
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| self.id == x.id)
            .unwrap_or(false)
    }
}

/// Resolve all physical placeholders in physical expression.
/// If an input expression does not contain any placeholders then
/// `param_values` is unused.
///
/// Besides resolved expression, also returns a flag that tells
/// whether placeholder in the original expression.
pub fn resolve_placeholders(
    expr: &Arc<dyn PhysicalExpr>,
    param_values: &Option<ParamValues>,
) -> Result<(Arc<dyn PhysicalExpr>, bool)> {
    struct PlaceholderRewriter<'a> {
        param_values: &'a Option<ParamValues>,
    }

    impl<'a> TreeNodeRewriter for PlaceholderRewriter<'a> {
        type Node = Arc<dyn PhysicalExpr>;

        fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
            match node.as_any().downcast_ref::<PlaceholderExpr>() {
                Some(PlaceholderExpr { id, data_type }) => {
                    if let Some(param_values) = self.param_values {
                        /* Extract a value and cast to the target type. */
                        let value = param_values
                            .get_placeholders_with_values(&id)?
                            .cast_to(data_type)?;
                        Ok(Transformed::yes(lit(value)))
                    } else {
                        plan_err!("There is no param for placeholder with id {}", id)
                    }
                }
                /* Nothing to do. */
                _ => Ok(Transformed::no(node)),
            }
        }
    }

    let rewrited = Arc::clone(&expr).rewrite(&mut PlaceholderRewriter {
        param_values: &param_values,
    })?;

    Ok((rewrited.data, rewrited.transformed))
}

/// Resolves all placeholders in the seq of physical expressions,
/// if there are no placeholders returns `None`, otherwise creates
/// and returns a new vector where all placeholders are resolved.
pub fn resolve_placeholders_seq<'a>(
    exprs: &[Arc<dyn PhysicalExpr>],
    param_values: &Option<ParamValues>,
) -> Result<Option<Vec<Arc<dyn PhysicalExpr>>>> {
    for (i, expr) in exprs.iter().enumerate() {
        let (resolved_expr, contains_placeholders) =
            resolve_placeholders(expr, param_values)?;
        if !contains_placeholders {
            // Use the original expression.
            continue;
        }
        // Create new vector and collect all expressions.
        let mut result = Vec::with_capacity(exprs.len());
        for j in 0..i {
            // We know that there are no placeholders at the prefix.
            result.push(Arc::clone(&exprs[j]));
        }
        result.push(resolved_expr);
        for j in i + 1..exprs.len() {
            let (resolved_expr, _) = resolve_placeholders(&exprs[j], param_values)?;
            result.push(resolved_expr);
        }
        return Ok(Some(result));
    }

    Ok(None)
}
