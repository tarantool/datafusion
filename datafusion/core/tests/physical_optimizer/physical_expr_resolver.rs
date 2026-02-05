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

//! Integration tests for [`PhysicalExprResolver`] optimizer rule.

use std::{collections::HashMap, sync::Arc};

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::config::ConfigOptions;
use datafusion_common::{ParamValues, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::{
    Partitioning,
    expressions::{BinaryExpr, col, lit, placeholder},
};
use datafusion_physical_optimizer::{
    PhysicalOptimizerRule, physical_expr_resolver::PhysicalExprResolver,
};
use datafusion_physical_plan::{
    ExecutionPlan, filter::FilterExec, get_plan_string,
    plan_transformer::TransformPlanExec, repartition::RepartitionExec,
};

use crate::physical_optimizer::test_utils::{
    coalesce_partitions_exec, global_limit_exec, resolve_placeholders_exec, stream_exec,
};

fn create_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("c1", DataType::Int32, true),
        Field::new("c2", DataType::Int32, true),
        Field::new("c3", DataType::Int32, true),
    ]))
}

fn filter_exec(
    schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(FilterExec::try_new(
        Arc::new(BinaryExpr::new(
            col("c3", schema.as_ref()).unwrap(),
            Operator::Gt,
            lit(0),
        )),
        input,
    )?))
}

fn filter_exec_with_placeholders(
    schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(FilterExec::try_new(
        Arc::new(BinaryExpr::new(
            col("c3", schema.as_ref()).unwrap(),
            Operator::Gt,
            placeholder("$foo", DataType::Int32),
        )),
        input,
    )?))
}

fn repartition_exec(
    streaming_table: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(RepartitionExec::try_new(
        streaming_table,
        Partitioning::RoundRobinBatch(8),
    )?))
}

#[test]
fn test_noop_if_no_placeholders_found() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec(schema, repartition)?;
    let coalesce_partitions = coalesce_partitions_exec(filter);
    let plan = global_limit_exec(coalesce_partitions, 0, Some(5));

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 0",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize = PhysicalExprResolver::new_post_optimization()
        .optimize(plan, &ConfigOptions::new())?;

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(initial, optimized_plan_string);

    Ok(())
}

#[test]
fn test_wrap_plan_with_transformer() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec_with_placeholders(schema, repartition)?;
    let coalesce_partitions = coalesce_partitions_exec(filter);
    let plan = global_limit_exec(coalesce_partitions, 0, Some(5));

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > $foo",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize = PhysicalExprResolver::new_post_optimization()
        .optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      FilterExec: c3@2 > $foo",
        "        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "          StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(optimized_plan_string, expected_optimized);

    let transformer = after_optimize
        .as_ref()
        .as_any()
        .downcast_ref::<TransformPlanExec>()
        .expect("should downcast");

    let param_values = ParamValues::Map(HashMap::from_iter([(
        "foo".to_string(),
        ScalarValue::Int32(Some(100)).into(),
    )]));

    let ctx = Arc::new(TaskContext::default().with_param_values(param_values));
    let resolved_plan = transformer.transform(&ctx)?;
    let resolved_plan_string = get_plan_string(&resolved_plan);
    let expected_resolved = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 100",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(resolved_plan_string, expected_resolved);

    Ok(())
}

#[test]
fn test_remove_useless_transformers() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec(schema, repartition)?;
    let transformer = resolve_placeholders_exec(filter);
    let coalesce_partitions = coalesce_partitions_exec(transformer);
    let global_limit = global_limit_exec(coalesce_partitions, 0, Some(5));
    let plan = resolve_placeholders_exec(global_limit);

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "        FilterExec: c3@2 > 0",
        "          RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "            StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize =
        PhysicalExprResolver::new().optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 0",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(optimized_plan_string, expected_optimized);

    Ok(())
}

#[test]
fn test_combine_transformers() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let transformer = resolve_placeholders_exec(repartition);
    let filter = filter_exec_with_placeholders(schema, transformer)?;
    let transformer = resolve_placeholders_exec(filter);
    let coalesce_partitions = coalesce_partitions_exec(transformer);
    let global_limit = global_limit_exec(coalesce_partitions, 0, Some(5));
    let plan = resolve_placeholders_exec(global_limit);

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "        FilterExec: c3@2 > $foo",
        "          TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "            RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "              StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_pre_optimization =
        PhysicalExprResolver::new().optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > $foo",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_pre_optimization);
    assert_eq!(optimized_plan_string, expected_optimized);

    let after_post_optimization = PhysicalExprResolver::new_post_optimization()
        .optimize(after_pre_optimization, &ConfigOptions::new())?;

    let expected_optimized = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      FilterExec: c3@2 > $foo",
        "        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "          StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_post_optimization);
    assert_eq!(optimized_plan_string, expected_optimized);

    Ok(())
}
