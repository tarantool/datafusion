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

//! Values execution plan

use std::any::Any;
use std::sync::Arc;

use super::{
    common, DisplayAs, ExecutionMode, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use crate::{
    memory::MemoryStream, ColumnarValue, DisplayFormatType, ExecutionPlan, Partitioning,
    PhysicalExpr,
};

use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion_common::{internal_err, plan_err, ParamValues, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::resolve_placeholders;
use datafusion_physical_expr::EquivalenceProperties;

/// Execution plan for values list based relation (produces constant rows)
#[derive(Debug)]
pub struct ValuesExec {
    /// The schema
    schema: SchemaRef,
    /// The data
    data: ValuesData,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
}

/// Represents maybe placeholdered values recordbatch.
#[derive(Debug, Clone)]
enum ValuesData {
    /// Resolved batch.
    Resolved(Vec<RecordBatch>),
    /// Batch with placeholders.
    Placeholdered(Vec<Vec<Arc<dyn PhysicalExpr>>>),
}

impl ValuesExec {
    fn new(schema: SchemaRef, data: ValuesData) -> Self {
        let cache = Self::compute_properties(Arc::clone(&schema));
        Self {
            schema,
            data,
            cache,
        }
    }

    /// Create a new values exec from data as expr.
    ///
    /// `has_placeholders` indicates if there are possible placeholders within rows.
    /// If there are placeholders they will be resolved in execution time.
    /// Flag is passed to avoid extra values bypass is there are placeholders and
    /// the caller knows it from somewhere.
    ///
    pub fn try_new(
        schema: SchemaRef,
        data: Vec<Vec<Arc<dyn PhysicalExpr>>>,
        has_placeholders: bool,
    ) -> Result<Self> {
        if data.is_empty() {
            return plan_err!("Values list cannot be empty");
        }
        let data = if has_placeholders {
            ValuesData::Placeholdered(data)
        } else {
            let batch = data_to_recordbatch(data, &None, &schema)?;
            ValuesData::Resolved(vec![batch])
        };

        Ok(Self::new(schema, data))
    }

    /// Create a new plan using the provided schema and batches.
    ///
    /// Errors if any of the batches don't match the provided schema, or if no
    /// batches are provided.
    pub fn try_new_from_batches(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<Self> {
        if batches.is_empty() {
            return plan_err!("Values list cannot be empty");
        }

        for batch in &batches {
            let batch_schema = batch.schema();
            if batch_schema != schema {
                return plan_err!(
                    "Batch has invalid schema. Expected: {schema}, got: {batch_schema}"
                );
            }
        }

        Ok(Self::new(schema, ValuesData::Resolved(batches)))
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for ValuesExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ValuesExec")
            }
        }
    }
}

impl ExecutionPlan for ValuesExec {
    fn name(&self) -> &'static str {
        "ValuesExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&self.schema),
            self.data.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // ValuesExec has a single output partition
        if 0 != partition {
            return internal_err!(
                "ValuesExec invalid partition {partition} (expected 0)"
            );
        }

        let data = match &self.data {
            ValuesData::Resolved(batches) => batches.clone(),
            ValuesData::Placeholdered(raw_batch) => vec![data_to_recordbatch(
                raw_batch.clone(),
                context.param_values(),
                &self.schema,
            )?],
        };

        Ok(Box::pin(MemoryStream::try_new(
            data,
            Arc::clone(&self.schema),
            None,
        )?))
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(match &self.data {
            ValuesData::Resolved(batches) => common::compute_record_batch_statistics(
                &[batches.clone()],
                &self.schema,
                None,
            ),
            ValuesData::Placeholdered(_) => Statistics::new_unknown(&self.schema),
        })
    }
}

/// Evaluate values expressions and convert them into recordbatch.
///
fn data_to_recordbatch(
    data: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    params: &Option<ParamValues>,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let n_col = schema.fields().len();
    let n_row = data.len();

    // We have this single row batch as a placeholder to satisfy evaluation argument,
    // and generate a single output row.
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;

    let arr = (0..n_col)
        .map(|j| {
            (0..n_row)
                .map(|i| {
                    let value = if params.is_some() {
                        resolve_placeholders(&data[i][j], params)?
                            .0
                            .evaluate(&batch)
                    } else {
                        // No param values, so even the expression contains placeholders,
                        // there will be a resolving error.
                        data[i][j].evaluate(&batch)
                    };
                    match value {
                        Ok(ColumnarValue::Scalar(scalar)) => Ok(scalar),
                        Ok(ColumnarValue::Array(a)) if a.len() == 1 => {
                            ScalarValue::try_from_array(&a, 0)
                        }
                        Ok(ColumnarValue::Array(a)) => {
                            plan_err!("Cannot have array values {a:?} in a values list")
                        }
                        Err(err) => Err(err),
                    }
                })
                .collect::<Result<Vec<_>>>()
                .and_then(ScalarValue::iter_to_array)
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&schema),
        arr,
        &RecordBatchOptions::new().with_row_count(Some(n_row)),
    )?;

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::lit;
    use crate::test::{self, make_partition};

    use arrow::util::pretty::pretty_format_batches;
    use arrow_schema::{DataType, Field};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, placeholder};

    #[tokio::test]
    async fn values_empty_case() -> Result<()> {
        let schema = test::aggr_test_schema();
        let empty = ValuesExec::try_new(schema, vec![], false);
        assert!(empty.is_err());
        Ok(())
    }

    #[test]
    fn new_exec_with_batches() {
        let batch = make_partition(7);
        let schema = batch.schema();
        let batches = vec![batch.clone(), batch];

        let _exec = ValuesExec::try_new_from_batches(schema, batches).unwrap();
    }

    #[test]
    fn new_exec_with_batches_empty() {
        let batch = make_partition(7);
        let schema = batch.schema();
        let _ = ValuesExec::try_new_from_batches(schema, Vec::new()).unwrap_err();
    }

    #[test]
    fn new_exec_with_batches_invalid_schema() {
        let batch = make_partition(7);
        let batches = vec![batch.clone(), batch];

        let invalid_schema = Arc::new(Schema::new(vec![
            Field::new("col0", DataType::UInt32, false),
            Field::new("col1", DataType::Utf8, false),
        ]));
        let _ = ValuesExec::try_new_from_batches(invalid_schema, batches).unwrap_err();
    }

    // Test issue: https://github.com/apache/datafusion/issues/8763
    #[test]
    fn new_exec_with_non_nullable_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col0",
            DataType::UInt32,
            false,
        )]));
        let _ = ValuesExec::try_new(Arc::clone(&schema), vec![vec![lit(1u32)]], false)
            .unwrap();
        // Test that a null value is rejected
        let _ = ValuesExec::try_new(
            schema,
            vec![vec![lit(ScalarValue::UInt32(None))]],
            false,
        )
        .unwrap_err();
    }

    #[test]
    fn data_to_recordbatch_resolves_placeholders() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col0",
            DataType::UInt32,
            false,
        )]));

        let sum = binary(
            lit(27u32),
            Operator::Plus,
            placeholder("$1", DataType::UInt32),
            &schema,
        )?;

        let rb = data_to_recordbatch(
            vec![vec![sum]],
            &Some(ParamValues::List(vec![ScalarValue::UInt32(Some(15))])),
            &schema,
        )?;

        let actual = format!("{}", pretty_format_batches(&[rb])?).to_lowercase();
        let expected = r#"
+------+
| col0 |
+------+
| 42   |
+------+
        "#;
        assert_eq!(actual, expected.trim());
        Ok(())
    }
}
