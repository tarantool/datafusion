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

use std::cell::OnceCell;
use std::sync::{Arc, LazyLock};

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use criterion::measurement::WallTime;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::prelude::SessionContext;
use datafusion_catalog::MemTable;
use datafusion_common::metadata::ScalarAndMetadata;
use datafusion_common::{ParamValues, ScalarValue};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::displayable;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::reuse::ReusableExecutionPlan;
use tokio::runtime::Runtime;

const NUM_FIELDS: usize = 1000;
const PREDICATE_LEN: usize = 50;

static SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(
        (0..NUM_FIELDS)
            .map(|i| Arc::new(Field::new(format!("x_{i}"), DataType::Int64, false)))
            .collect::<Fields>(),
    ))
});

/// Decides when to generate placeholders, helping to form a query
/// with a certain placeholders percent.
struct PlaceholderCounter {
    placeholders_percent: usize,
    c: usize,
    placeholder_idx: usize,
    num_placeholders: usize,
}

impl PlaceholderCounter {
    fn new(placeholders_percent: usize) -> Self {
        Self {
            placeholders_percent,
            c: 0,
            placeholder_idx: 0,
            num_placeholders: 0,
        }
    }

    fn placeholder(&mut self) -> Option<String> {
        let is_placeholder = self.c < self.placeholders_percent;
        self.c += 1;
        if self.c >= 100 {
            self.c = 0;
        }
        if is_placeholder {
            self.num_placeholders += 1;
            Some("$1".to_owned())
        } else {
            None
        }
    }

    fn placeholder_or(&mut self, f: impl FnOnce() -> String) -> String {
        self.placeholder().unwrap_or_else(f)
    }
}

fn col_name(i: usize) -> String {
    format!("x_{i}")
}

fn aggr_name(i: usize) -> String {
    format!("aggr_{i}")
}

fn physical_plan(
    ctx: &SessionContext,
    rt: &tokio::runtime::Handle,
    sql: &str,
) -> Arc<dyn ExecutionPlan> {
    rt.block_on(async {
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    })
}

fn predicate(mut comparee: impl FnMut(usize) -> (String, String), len: usize) -> String {
    let mut predicate = String::new();
    for i in 0..len {
        if i > 0 {
            predicate.push_str(" AND ");
        }
        let (lhs, rhs) = comparee(i);
        predicate.push_str(&lhs);
        predicate.push_str(" = ");
        predicate.push_str(&rhs);
    }
    predicate
}

/// Returns a typical plan for the query like:
///
/// ```sql
/// SELECT aggr1(col1) as aggr1, aggr2(col2) as aggr2 FROM t
/// WHERE p1
/// HAVING p2
/// ```
///
/// Where `p1` and `p2` some long predicates.
///
fn query0(placeholders_percent: usize) -> (String, usize) {
    let mut plc = PlaceholderCounter::new(placeholders_percent);
    let mut query = String::new();
    query.push_str("SELECT ");
    for i in 0..NUM_FIELDS {
        if i > 0 {
            query.push_str(", ");
        }
        query.push_str("AVG(");

        if let Some(placeholder) = plc.placeholder() {
            query.push_str(&format!("{}+{}", placeholder, col_name(i)));
        } else {
            query.push_str(&col_name(i));
        }

        query.push_str(") AS ");
        query.push_str(&aggr_name(i));
    }
    query.push_str(" FROM t WHERE ");
    query.push_str(&predicate(
        |i| {
            (
                plc.placeholder_or(|| col_name(i)),
                plc.placeholder_or(|| col_name(i + 1)),
            )
        },
        PREDICATE_LEN,
    ));
    query.push_str(" HAVING ");
    query.push_str(&predicate(
        |i| {
            (
                plc.placeholder_or(|| aggr_name(i)),
                plc.placeholder_or(|| aggr_name(i + 1)),
            )
        },
        PREDICATE_LEN,
    ));
    (query, plc.num_placeholders)
}

/// Returns a typical plan for the query like:
///
/// ```sql
/// SELECT projection FROM t JOIN v ON t.a = v.a
/// WHERE p1
/// ```
///
fn query1(placeholders_percent: usize) -> (String, usize) {
    let mut plc = PlaceholderCounter::new(placeholders_percent);
    let mut query = String::new();
    query.push_str("SELECT ");
    for i in (0..NUM_FIELDS).step_by(2) {
        if i > 0 {
            query.push_str(", ");
        }
        let col = if (i / 2) % 2 == 0 {
            format!("t.{}", col_name(i))
        } else {
            format!("v.{}", col_name(i))
        };
        let add = plc.placeholder_or(|| "1".to_owned());
        let proj = format!("{col} + {add}");
        query.push_str(&proj);
    }
    query.push_str(" FROM t JOIN v ON t.x_0 = v.x_0 WHERE ");

    query.push_str(&predicate(
        |i| {
            (
                plc.placeholder_or(|| format!("t.{}", col_name(i))),
                plc.placeholder_or(|| i.to_string()),
            )
        },
        PREDICATE_LEN,
    ));
    (query, plc.num_placeholders)
}

/// Returns a typical plan for the query like:
///
/// ```sql
/// SELECT projection FROM t
/// WHERE p
/// ```
///
fn query2(placeholders_percent: usize) -> (String, usize) {
    let mut plc = PlaceholderCounter::new(placeholders_percent);
    let mut query = String::new();
    query.push_str("SELECT ");

    // Create non-trivial projection.
    for i in 0..NUM_FIELDS / 2 {
        if i > 0 {
            query.push_str(", ");
        }
        query.push_str(&col_name(i * 2));
        query.push_str(" + ");
        query.push_str(&plc.placeholder_or(|| col_name(i * 2 + 1)));
    }

    query.push_str(" FROM t WHERE ");
    query.push_str(&predicate(
        |i| {
            (
                plc.placeholder_or(|| col_name(i)),
                plc.placeholder_or(|| i.to_string()),
            )
        },
        PREDICATE_LEN,
    ));
    (query, plc.num_placeholders)
}

fn init() -> (SessionContext, Runtime) {
    let rt = Runtime::new().unwrap();
    let ctx = SessionContext::new();
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(Arc::clone(&SCHEMA), vec![vec![], vec![]]).unwrap()),
    )
    .unwrap();

    ctx.register_table(
        "v",
        Arc::new(MemTable::try_new(Arc::clone(&SCHEMA), vec![vec![], vec![]]).unwrap()),
    )
    .unwrap();
    (ctx, rt)
}

/// Benchmark is intended to measure overhead of actions, required to perform
/// making an independent instance of the execution plan to re-execute it, avoiding
/// re-planning stage.
fn bench_reset(
    g: &mut criterion::BenchmarkGroup<'_, WallTime>,
    query_fn: impl FnOnce() -> String,
) {
    let (ctx, rt) = init();
    let query = query_fn();
    let rt = rt.handle();
    let plan: OnceCell<Arc<dyn ExecutionPlan>> = OnceCell::new();
    g.bench_function("reset", |b| {
        let plan = plan.get_or_init(|| {
            log::info!("sql:\n{}\n\n", query);
            let plan = physical_plan(&ctx, rt, &query);
            log::info!("plan:\n{}", displayable(plan.as_ref()).indent(true));
            plan
        });
        b.iter(|| std::hint::black_box(reset_plan_states(Arc::clone(&plan)).unwrap()))
    });
}

/// The same as [`bench_reset`] for placeholdered plans.
/// `placeholders_percent` is a percent of placeholders that must be used in generated queries.
fn bench_bind(
    g: &mut criterion::BenchmarkGroup<'_, WallTime>,
    placeholders_percent: usize,
    query_fn: impl FnOnce(usize) -> (String, usize),
) {
    let (ctx, rt) = init();
    let params = ParamValues::List(vec![ScalarAndMetadata::new(
        ScalarValue::Int64(Some(42)),
        None,
    )]);
    let (query, num_placeholders) = query_fn(placeholders_percent);
    let rt = rt.handle();
    let plan: OnceCell<ReusableExecutionPlan> = OnceCell::new();
    g.bench_function(format!("{num_placeholders}_placeholders"), move |b| {
        let plan = plan.get_or_init(|| {
            log::info!("sql:\n{}\n\n", query);
            let plan = physical_plan(&ctx, rt, &query);
            log::info!("plan:\n{}", displayable(plan.as_ref()).indent(true));
            plan.into()
        });
        b.iter(|| std::hint::black_box(plan.bind(Some(&params))))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    env_logger::init();

    for (query_idx, query_fn) in [query0, query1, query2].iter().enumerate() {
        {
            let mut g = c.benchmark_group(format!("reset_query{query_idx}"));
            bench_reset(&mut g, || query_fn(0).0);
        }
        {
            let mut g = c.benchmark_group(format!("bind_query{query_idx}"));
            for placeholders_percent in [0, 1, 10, 50, 100] {
                bench_bind(&mut g, placeholders_percent, query_fn);
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
