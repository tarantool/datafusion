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

//! This module provides a mechanism for two-pass [`ExecutionPlan`] tree transformations.
//!
//! In the first pass, the tree is visited (top-down) to identify nodes that require transformation
//! based on certain criteria.
//!
//! In the second pass, the stored plans are applied to the tree (top-down) using a
//! [`TreeNodeRewriter`].
//!
//! This approach is beneficial because it allows making multiple transformations during a single
//! pass and caches node indices, ensuring that only nodes matching the criteria are transformed,
//! which can improve performance.

mod resolve_placeholders;

pub use resolve_placeholders::ResolvePlaceholdersRule;

use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;

use datafusion_common::{
    Result,
    tree_node::{
        Transformed, TreeNode, TreeNodeRecursion, TreeNodeRewriter, TreeNodeVisitor,
    },
};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use itertools::Itertools;

use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics};

/// A rule for transforming an [`ExecutionPlan`] node.
///
/// This trait is used in the two-pass tree transformation process. Both passes use a top-down
/// traversal. In the first pass, the [`ExecutionTransformationRule::matches`] method is used to
/// identify nodes that require transformation. In the second pass, the
/// [`ExecutionTransformationRule::rewrite`] method is used to apply the transformation.
pub trait ExecutionTransformationRule: Send + Sync + Debug {
    /// Returns the rule name.
    fn name(&self) -> &str;

    /// Returns the rule as [`Any`] so that it can be downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    /// Clones this rule.
    fn clone_box(&self) -> Box<dyn ExecutionTransformationRule>;

    /// Checks if the given [`ExecutionPlan`] node matches the criteria for this rule.
    fn matches(&mut self, _node: &Arc<dyn ExecutionPlan>) -> Result<bool> {
        Ok(false)
    }

    /// Transforms the given [`ExecutionPlan`] node.
    fn rewrite(
        &self,
        node: Arc<dyn ExecutionPlan>,
        _ctx: &TaskContext,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        Ok(Transformed::no(node))
    }
}

/// Stores the transformation operations to be applied to an [`ExecutionPlan`] node at a specific
/// index.
#[derive(Debug, Clone)]
struct TransformationPlan {
    /// The index of the node in the tree traversal.
    pub node_index: usize,
    /// The list of transformation rule indices to apply.
    pub rule_indices: Vec<usize>,
}

/// Helper for building transformation plans for an [`ExecutionPlan`] tree during a single pass.
struct TransformationPlanner {
    cursor: usize,
    rules: Vec<Box<dyn ExecutionTransformationRule>>,
    plans: Vec<TransformationPlan>,
}

impl TransformationPlanner {
    fn new(rules: Vec<Box<dyn ExecutionTransformationRule>>) -> Self {
        Self {
            cursor: 0,
            rules,
            plans: Vec::new(),
        }
    }
}

impl<'n> TreeNodeVisitor<'n> for TransformationPlanner {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        if node.as_any().is::<TransformPlanExec>() {
            return Ok(TreeNodeRecursion::Jump);
        }

        let index = self.cursor;
        self.cursor += 1;

        let mut rule_indices = Vec::new();
        for (rule_index, rule) in self.rules.iter_mut().enumerate() {
            if rule.matches(node)? {
                rule_indices.push(rule_index);
            }
        }

        if !rule_indices.is_empty() {
            self.plans.push(TransformationPlan {
                node_index: index,
                rule_indices,
            });
        }

        Ok(TreeNodeRecursion::Continue)
    }
}

/// Helper for applying transformation plans to an [`ExecutionPlan`] tree during a single pass.
///
/// This applier uses a [`TaskContext`] to provide necessary information for the transformation
/// rules.
struct TransformationApplier<'rules, 'plans, 'ctx> {
    cursor: usize,
    rules: &'rules [Box<dyn ExecutionTransformationRule>],
    plans: &'plans [TransformationPlan],
    ctx: &'ctx TaskContext,
}

impl<'rules, 'plans, 'ctx> TransformationApplier<'rules, 'plans, 'ctx> {
    fn new(
        rules: &'rules [Box<dyn ExecutionTransformationRule>],
        plans: &'plans [TransformationPlan],
        ctx: &'ctx TaskContext,
    ) -> Self {
        Self {
            cursor: 0,
            rules,
            plans,
            ctx,
        }
    }
}

impl<'rules, 'plans, 'ctx> TreeNodeRewriter
    for TransformationApplier<'rules, 'plans, 'ctx>
{
    type Node = Arc<dyn ExecutionPlan>;

    fn f_down(&mut self, mut node: Self::Node) -> Result<Transformed<Self::Node>> {
        let Some(plan) = self.plans.first() else {
            return Ok(Transformed::new(node, false, TreeNodeRecursion::Stop));
        };

        if node.as_any().is::<TransformPlanExec>() {
            return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
        }

        let index = self.cursor;
        self.cursor += 1;

        if index != plan.node_index {
            return Ok(Transformed::no(node));
        }

        self.plans = &self.plans[1..];

        let mut transformed = false;
        for rule_index in plan.rule_indices.iter() {
            let rule = &self.rules[*rule_index];
            let transform = rule.rewrite(node, self.ctx)?;
            node = transform.data;
            transformed |= transform.transformed;
        }

        Ok(Transformed::new(
            node,
            transformed,
            TreeNodeRecursion::Continue,
        ))
    }
}

/// An [`ExecutionPlan`] that applies transformation rules during execution.
#[derive(Debug)]
pub struct TransformPlanExec {
    /// The input execution plan.
    input: Arc<dyn ExecutionPlan>,
    /// The transformation rules to apply.
    rules: Vec<Box<dyn ExecutionTransformationRule>>,
    /// The pre-calculated transformation plans.
    plans: Vec<TransformationPlan>,
    /// Execution metrics.
    metrics: ExecutionPlanMetricsSet,
}

impl TransformPlanExec {
    /// Create a new [TransformPlanExec].
    ///
    /// This method returns an error if any of the transformation rules return an error during the
    /// initial traversal of the input plan.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        rules: Vec<Box<dyn ExecutionTransformationRule>>,
    ) -> Result<Self> {
        let mut planner = TransformationPlanner::new(rules);
        input.visit(&mut planner)?;

        Ok(Self {
            input,
            rules: planner.rules,
            plans: planner.plans,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Returns the number of nodes that match at least one transformation rule.
    pub fn plans_to_transform(&self) -> usize {
        self.plans.len()
    }

    pub fn transform(
        &self,
        context: &Arc<TaskContext>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut applier = TransformationApplier::new(&self.rules, &self.plans, context);
        let input = Arc::clone(&self.input);
        input.rewrite(&mut applier).map(|t| t.data)
    }

    /// Returns the input plan.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Returns the transformation rules.
    pub fn rules(&self) -> &[Box<dyn ExecutionTransformationRule>] {
        &self.rules
    }

    /// Checks if the transformation rules contains a rule of a specific type.
    pub fn has_rule<T: ExecutionTransformationRule + 'static>(&self) -> bool {
        self.rules.iter().any(|r| r.as_any().is::<T>())
    }

    /// Adds a new transformation rule and recalculates transformation plans.
    pub fn add_rule(
        &self,
        new_rule: Box<dyn ExecutionTransformationRule>,
    ) -> Result<Self> {
        self.add_rules(vec![new_rule])
    }

    /// Adds new transformation rules and recalculates transformation plans.
    pub fn add_rules(
        &self,
        new_rules: Vec<Box<dyn ExecutionTransformationRule>>,
    ) -> Result<Self> {
        let mut planner = TransformationPlanner::new(new_rules);
        self.input.visit(&mut planner)?;
        let new_rules = planner.rules;
        let new_plans = planner.plans;

        let mut current_rules = self
            .rules
            .iter()
            .map(|rule| rule.clone_box())
            .collect::<Vec<_>>();

        let offset = current_rules.len();
        current_rules.extend(new_rules);

        let mut merged_plans = Vec::with_capacity(self.plans.len() + new_plans.len());
        let mut old_plans_iter = self.plans.iter().peekable();
        let mut new_plans_iter = new_plans.into_iter().peekable();

        while old_plans_iter.peek().is_some() || new_plans_iter.peek().is_some() {
            match (old_plans_iter.peek(), new_plans_iter.peek()) {
                (Some(&old), Some(new)) if old.node_index == new.node_index => {
                    let mut merged_plan = old.clone();
                    for &rule_index in &new.rule_indices {
                        merged_plan.rule_indices.push(offset + rule_index);
                    }
                    merged_plans.push(merged_plan);
                    old_plans_iter.next();
                    new_plans_iter.next();
                }
                (Some(&old), Some(new)) if old.node_index < new.node_index => {
                    merged_plans.push(old.clone());
                    old_plans_iter.next();
                }
                (Some(_), Some(_)) => {
                    // old_node_index > new.node_index
                    let mut new_plan = new_plans_iter.next().unwrap();
                    for rule_index in new_plan.rule_indices.iter_mut() {
                        *rule_index += offset;
                    }
                    merged_plans.push(new_plan);
                }
                (Some(&old), None) => {
                    merged_plans.push(old.clone());
                    old_plans_iter.next();
                }
                (None, Some(_)) => {
                    let mut new_plan = new_plans_iter.next().unwrap();
                    for rule_index in new_plan.rule_indices.iter_mut() {
                        *rule_index += offset;
                    }
                    merged_plans.push(new_plan);
                }
                (None, None) => unreachable!(),
            }
        }

        Ok(Self {
            input: Arc::clone(&self.input),
            rules: current_rules,
            plans: merged_plans,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for TransformPlanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let mut rule_to_nodes_count = vec![0; self.rules.len()];
                for plan in self.plans.iter() {
                    for rule_index in plan.rule_indices.iter() {
                        rule_to_nodes_count[*rule_index] += 1;
                    }
                }

                let rules = rule_to_nodes_count
                    .into_iter()
                    .enumerate()
                    .map(|(rule_index, nodes_count)| {
                        let rule_name = self.rules[rule_index].name();
                        format!("{rule_name}: plans_to_modify={nodes_count}")
                    })
                    .join(", ");

                write!(f, "TransformPlanExec: rules=[{rules}]")
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }
}

impl ExecutionPlan for TransformPlanExec {
    fn static_name() -> &'static str
    where
        Self: Sized,
    {
        "TransformPlanExec"
    }

    fn name(&self) -> &str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let rules = self.rules.iter().map(|r| r.clone_box()).collect();
        Ok(Arc::new(TransformPlanExec::try_new(
            Arc::clone(&children[0]),
            rules,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metric = MetricBuilder::new(&self.metrics).elapsed_compute(partition);
        let _timer = metric.timer();

        let transformed = self.transform(&context)?;
        transformed.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        self.partition_statistics(None)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        self.input.partition_statistics(partition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coalesce_partitions::CoalescePartitionsExec;
    use crate::empty::EmptyExec;
    use crate::limit::GlobalLimitExec;
    use crate::placeholder_row::PlaceholderRowExec;
    use crate::plan_transformer::resolve_placeholders::ResolvePlaceholdersRule;
    use crate::projection::ProjectionExec;
    use crate::{get_plan_string, test};
    use arrow_schema::{DataType, Schema};
    use datafusion_common::{ParamValues, ScalarValue, tree_node::Transformed};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, lit, placeholder};
    use datafusion_physical_expr::projection::ProjectionExpr;
    use insta::assert_snapshot;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct MockRule {
        name: String,
        match_name: String,
    }

    impl ExecutionTransformationRule for MockRule {
        fn name(&self) -> &str {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_box(&self) -> Box<dyn ExecutionTransformationRule> {
            Box::new(self.clone())
        }

        fn matches(&mut self, node: &Arc<dyn ExecutionPlan>) -> Result<bool> {
            Ok(node.name() == self.match_name || self.match_name == "all")
        }

        fn rewrite(
            &self,
            node: Arc<dyn ExecutionPlan>,
            _ctx: &TaskContext,
        ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
            Ok(Transformed::no(node))
        }
    }

    #[test]
    fn test_has_rule() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        let input = Arc::new(EmptyExec::new(schema));

        let resolve_rule = Box::new(ResolvePlaceholdersRule::new());
        let exec = TransformPlanExec::try_new(input, vec![resolve_rule])?;
        assert!(exec.has_rule::<ResolvePlaceholdersRule>());
        assert!(!exec.has_rule::<MockRule>());

        let exec = exec.add_rule(Box::new(MockRule {
            name: "mock".to_string(),
            match_name: "any".to_string(),
        }))?;

        assert!(exec.has_rule::<ResolvePlaceholdersRule>());
        assert!(exec.has_rule::<MockRule>());

        Ok(())
    }

    #[test]
    fn test_add_rules_merge() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        let empty = Arc::new(EmptyExec::new(schema));
        let coalesce = Arc::new(CoalescePartitionsExec::new(empty));
        let input = Arc::new(GlobalLimitExec::new(coalesce, 0, None)) as Arc<_>;

        // Node 0: GlobalLimitExec
        // Node 1: CoalescePartitionsExec
        // Node 2: EmptyExec

        let rule_a = Box::new(MockRule {
            name: "ruleA".to_string(),
            match_name: "CoalescePartitionsExec".to_string(),
        });
        let exec = TransformPlanExec::try_new(Arc::clone(&input), vec![rule_a.clone()])?;

        assert_eq!(exec.rules.len(), 1);
        assert_eq!(exec.plans.len(), 1);
        assert_eq!(exec.plans[0].node_index, 1);
        assert_eq!(exec.plans[0].rule_indices, vec![0]);

        let rule_b = Box::new(MockRule {
            name: "ruleB".to_string(),
            match_name: "GlobalLimitExec".to_string(),
        });
        let exec = exec.add_rules(vec![rule_b.clone()])?;

        assert_eq!(exec.rules.len(), 2);
        assert_eq!(exec.plans.len(), 2);
        assert_eq!(exec.plans[0].node_index, 0);
        assert_eq!(exec.plans[0].rule_indices, vec![1]);
        assert_eq!(exec.plans[1].node_index, 1);
        assert_eq!(exec.plans[1].rule_indices, vec![0]);

        let rule_c = Box::new(MockRule {
            name: "ruleC".to_string(),
            match_name: "EmptyExec".to_string(),
        });
        let exec = exec.add_rules(vec![rule_c.clone()])?;

        assert_eq!(exec.rules.len(), 3);
        assert_eq!(exec.plans.len(), 3);
        assert_eq!(exec.plans[0].node_index, 0);
        assert_eq!(exec.plans[0].rule_indices, vec![1]);
        assert_eq!(exec.plans[1].node_index, 1);
        assert_eq!(exec.plans[1].rule_indices, vec![0]);
        assert_eq!(exec.plans[2].node_index, 2);
        assert_eq!(exec.plans[2].rule_indices, vec![2]);

        let rule_d = Box::new(MockRule {
            name: "ruleD".to_string(),
            match_name: "all".to_string(),
        });

        let exec = exec.add_rules(vec![rule_d.clone()])?;
        let check_full_plan = |exec: TransformPlanExec| {
            assert_eq!(exec.rules.len(), 4);
            assert_eq!(exec.plans.len(), 3);
            assert_eq!(exec.plans[0].node_index, 0);
            assert_eq!(exec.plans[0].rule_indices, vec![1, 3]);
            assert_eq!(exec.plans[1].node_index, 1);
            assert_eq!(exec.plans[1].rule_indices, vec![0, 3]);
            assert_eq!(exec.plans[2].node_index, 2);
            assert_eq!(exec.plans[2].rule_indices, vec![2, 3]);
        };

        check_full_plan(exec);

        let exec = TransformPlanExec::try_new(
            Arc::clone(&input),
            vec![
                rule_a.clone(),
                rule_b.clone(),
                rule_c.clone(),
                rule_d.clone(),
            ],
        )?;

        check_full_plan(exec);

        let exec = TransformPlanExec::try_new(Arc::clone(&input), vec![rule_a])?
            .add_rules(vec![rule_b, rule_c, rule_d])?;

        check_full_plan(exec);

        Ok(())
    }

    #[tokio::test]
    async fn test_consts_evaluation() -> Result<()> {
        let schema = test::aggr_test_schema();
        let expr = binary(
            lit(10),
            Operator::Plus,
            binary(
                lit(5),
                Operator::Multiply,
                placeholder("$1", DataType::Int32),
                &schema,
            )?,
            &schema,
        )?;

        let projection_expr = ProjectionExpr {
            expr,
            alias: "sum".to_string(),
        };

        let row = Arc::new(PlaceholderRowExec::new(schema));
        let projection = ProjectionExec::try_new(vec![projection_expr], row)?;
        let transformer = TransformPlanExec::try_new(
            Arc::new(projection),
            vec![Box::new(ResolvePlaceholdersRule::new())],
        )?;

        let param_values = ParamValues::List(vec![ScalarValue::Int32(Some(20)).into()]);
        let task_ctx = Arc::new(TaskContext::default().with_param_values(param_values));

        let plan = transformer.transform(&task_ctx)?;
        let plan_string = get_plan_string(&plan).join("\n");

        assert_snapshot!(plan_string, @r"
        ProjectionExec: expr=[110 as sum]
          PlaceholderRowExec
        ");

        Ok(())
    }
}
