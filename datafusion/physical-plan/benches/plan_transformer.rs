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

use std::any::Any;
use std::sync::Arc;

use arrow_schema::Schema;
use criterion::measurement::WallTime;
use datafusion_common::Result;
use datafusion_common::tree_node::{
    Transformed, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::plan_transformer::{
    ExecutionTransformationRule, TransformPlanExec,
};

use criterion::{BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main};

#[derive(Debug, Clone)]
struct ResetAllRule {}

impl ExecutionTransformationRule for ResetAllRule {
    fn name(&self) -> &str {
        "ResetAllRule"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ExecutionTransformationRule> {
        Box::new(self.clone())
    }

    fn matches(&mut self, _node: &Arc<dyn ExecutionPlan>) -> Result<bool> {
        Ok(true)
    }

    fn rewrite(
        &self,
        node: Arc<dyn ExecutionPlan>,
        _ctx: &TaskContext,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        node.reset_state().map(Transformed::yes)
    }
}

#[derive(Debug, Clone)]
struct ResetByNameRule {
    node_name: String,
}

impl ExecutionTransformationRule for ResetByNameRule {
    fn name(&self) -> &str {
        "ResetByNameRule"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ExecutionTransformationRule> {
        Box::new(self.clone())
    }

    fn matches(&mut self, node: &Arc<dyn ExecutionPlan>) -> Result<bool> {
        Ok(node.name() == self.node_name)
    }

    fn rewrite(
        &self,
        node: Arc<dyn ExecutionPlan>,
        _ctx: &TaskContext,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        node.reset_state().map(Transformed::yes)
    }
}

struct ResetAllRewriter;

impl TreeNodeRewriter for ResetAllRewriter {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        // This part is needed for handling nested rewriters.
        if node.as_any().is::<MockRewriterExec>() {
            return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
        }
        node.reset_state().map(Transformed::yes)
    }
}

struct ResetByNameRewriter {
    node_name: String,
}

impl TreeNodeRewriter for ResetByNameRewriter {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        // This part is needed for handling nested rewriters.
        if node.as_any().is::<MockRewriterExec>() {
            return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
        }

        if node.name() == self.node_name {
            node.reset_state().map(Transformed::yes)
        } else {
            Ok(Transformed::no(node))
        }
    }
}

struct MockRewriterExec {
    input: Arc<dyn ExecutionPlan>,
}

impl MockRewriterExec {
    fn execute(
        &self,
        rewriter: &mut impl TreeNodeRewriter<Node = Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input);
        input.rewrite(rewriter).map(|t| t.data)
    }
}

fn create_deep_tree(depth: usize) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::empty());
    let mut node: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
    for _ in 0..depth {
        node = Arc::new(CoalescePartitionsExec::new(node));
    }
    node
}

fn nodes_amount(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let mut amount = 0;
    plan.apply(|_| {
        amount += 1;
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    amount
}

fn benchmark_with_transformer_exec(
    group: &mut BenchmarkGroup<'_, WallTime>,
    batch_label: &str,
    plan: &Arc<dyn ExecutionPlan>,
    rules: &[Box<dyn ExecutionTransformationRule>],
    batch_size: BatchSize,
) {
    let ctx = Arc::new(TaskContext::default());
    let nodes_amount = nodes_amount(plan);

    group.bench_function(
        format!(
            "transform_plan_exec_two_phases_{batch_label}_{nodes_amount}_nodes_{}_rule(s)",
            rules.len()
        ),
        |b| {
            b.iter_batched(
                || {
                    (
                        Arc::clone(plan),
                        rules.iter().map(|r| r.clone_box()).collect(),
                    )
                },
                |(plan, rules)| {
                    let transformer = TransformPlanExec::try_new(plan, rules).unwrap();
                    transformer.transform(&ctx).unwrap();
                },
                batch_size,
            )
        },
    );

    group.bench_function(
        format!(
            "transform_plan_exec_second_phase_{batch_label}_{nodes_amount}_nodes_{}_rule(s)",
            rules.len()
        ),
        |b| {
            let plan = Arc::clone(plan);
            let rules = rules.iter().map(|r| r.clone_box()).collect();
            let transformer = Arc::new(TransformPlanExec::try_new(plan, rules).unwrap());

            b.iter_batched(
                || Arc::clone(&transformer),
                |transformer| {
                    transformer.transform(&ctx).unwrap();
                },
                batch_size,
            )
        },
    );
}

fn benchmark_with_tree_node_rewriter(
    group: &mut BenchmarkGroup<'_, WallTime>,
    batch_label: &str,
    plan: &Arc<dyn ExecutionPlan>,
    mut rewriter: impl TreeNodeRewriter<Node = Arc<dyn ExecutionPlan>>,
    rewrites_amount: usize,
    batch_size: BatchSize,
) {
    let nodes_amount = nodes_amount(plan);
    let mock_rewriter_exec = Arc::new(MockRewriterExec {
        input: Arc::clone(plan),
    });

    group.bench_function(
        format!(
            "tree_node_rewriter_{batch_label}_{nodes_amount}_nodes_{rewrites_amount}_iteration(s)"
        ),
        |b| {
            b.iter_batched(
                || Arc::clone(&mock_rewriter_exec),
                |plan| {
                    for _ in 0..rewrites_amount {
                        plan.execute(&mut rewriter).unwrap();
                    }
                },
                batch_size,
            )
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let depths = [5, 30];
    let rules_count = [1, 2];
    let batch_size = BatchSize::SmallInput;
    let mut group = c.benchmark_group("plan_transformation");

    for depth in depths {
        let plan = create_deep_tree(depth);

        for count in rules_count {
            let reset_all_rules = (0..count)
                .map(|_| Box::new(ResetAllRule {}) as Box<_>)
                .collect::<Vec<_>>();

            let reset_one_rules = (0..count)
                .map(|_| {
                    Box::new(ResetByNameRule {
                        node_name: "EmptyExec".to_string(),
                    }) as Box<_>
                })
                .collect::<Vec<_>>();

            benchmark_with_transformer_exec(
                &mut group,
                "reset_all",
                &plan,
                &reset_all_rules,
                batch_size,
            );

            benchmark_with_tree_node_rewriter(
                &mut group,
                "reset_all",
                &plan,
                ResetAllRewriter,
                count,
                batch_size,
            );

            benchmark_with_transformer_exec(
                &mut group,
                "reset_one",
                &plan,
                &reset_one_rules,
                batch_size,
            );

            benchmark_with_tree_node_rewriter(
                &mut group,
                "reset_one",
                &plan,
                ResetByNameRewriter {
                    node_name: "EmptyExec".to_string(),
                },
                count,
                batch_size,
            );
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
