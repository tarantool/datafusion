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

//! Values source for reading constant values or expressions with placeholders.
//!
//! This module provides the [`ValuesSource`] struct, which can be used to read values that may
//! contain placeholders.

use std::{
    any::Any,
    fmt::Formatter,
    sync::{Arc, LazyLock},
};

use arrow::{
    array::{RecordBatch, RecordBatchOptions},
    datatypes::{Schema, SchemaRef},
};
use datafusion_common::{
    Result, ScalarValue, Statistics, assert_eq_or_internal_err, exec_err, plan_err,
};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::{
    EquivalenceProperties, Partitioning, PhysicalExpr,
    expressions::{has_placeholders, lit},
};
use datafusion_physical_plan::{
    DisplayFormatType, common::compute_record_batch_statistics, coop::cooperative,
    execution_plan::ReplacePhysicalExpr, memory::MemoryStream,
};

use crate::{
    memory::MemorySourceConfig,
    source::{DataSource, DataSourceExec},
};

/// Information about a record with placeholders.
#[derive(Clone, Debug)]
struct RecordWithPlaceholders {
    /// The physical expression.
    pub expr: Arc<dyn PhysicalExpr>,
    /// The row index.
    pub row: usize,
    /// The column index.
    pub column: usize,
}

/// A data source for reading values that may contain placeholders.
///
/// This source is used when the values contain placeholders that need to be resolved at execution
/// time. If all values are constant, [`MemorySourceConfig`] is typically used instead.
#[derive(Clone, Debug)]
pub struct ValuesSource {
    /// The schema of the values.
    schema: SchemaRef,
    /// The record batch containing the values.
    batch: RecordBatch,
    /// Positions of rows that contain placeholders.
    records_with_placeholders: Vec<RecordWithPlaceholders>,
}

impl ValuesSource {
    /// Create a new [`ValuesSource`] from the provided schema and data.
    #[expect(clippy::needless_pass_by_value)]
    fn try_new(schema: SchemaRef, data: Vec<Vec<Arc<dyn PhysicalExpr>>>) -> Result<Self> {
        if data.is_empty() {
            return plan_err!("Values list cannot be empty");
        }

        let n_row = data.len();
        let n_col = schema.fields().len();

        let mut records_with_placeholders = Vec::new();

        // Evaluate each column
        let arrays = (0..n_col)
            .map(|j| {
                (0..n_row)
                    .map(|i| {
                        let expr = &data[i][j];
                        if has_placeholders(expr) {
                            let record = RecordWithPlaceholders {
                                expr: Arc::clone(expr),
                                row: i,
                                column: j,
                            };

                            records_with_placeholders.push(record);

                            let data_type = schema.field(j).data_type();
                            return ScalarValue::new_default(data_type);
                        }

                        evaluate_to_scalar(expr.as_ref())
                    })
                    .collect::<Result<Vec<_>>>()
                    .and_then(ScalarValue::iter_to_array)
            })
            .collect::<Result<Vec<_>>>()?;

        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(n_row)),
        )?;

        Ok(Self {
            batch,
            records_with_placeholders,
            schema,
        })
    }

    /// Create a new execution plan from a list of values.
    ///
    /// If the values contain placeholders, a [`ValuesSource`] is used which will resolve them at
    /// execution time. Otherwise, [`MemorySourceConfig`] will be used as it is more efficient for
    /// constant values.
    pub fn try_new_exec(
        schema: SchemaRef,
        data: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    ) -> Result<Arc<DataSourceExec>> {
        let source = Self::try_new(schema, data)?;
        if source.records_with_placeholders.is_empty() {
            MemorySourceConfig::try_new_from_batches(source.schema, vec![source.batch])
        } else {
            Ok(DataSourceExec::from_data_source(source))
        }
    }

    /// Returns the schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Returns the physical expressions for each value in the source.
    ///
    /// For values that do not contain placeholders, the expression will be a literal.
    /// For values that contain placeholders, the original expression is returned.
    pub fn expressions(&self) -> Vec<Vec<Arc<dyn PhysicalExpr>>> {
        let columns = self.batch.columns();
        let columns_len = columns.len();
        let rows = columns.first().map(|c| c.len()).unwrap_or(0);
        let mut exprs = Vec::with_capacity(rows);

        for row in 0..rows {
            let mut column_exprs = Vec::with_capacity(columns_len);
            for column in columns {
                let scalar = ScalarValue::try_from_array(&column, row)
                    .expect("should build scalar");

                column_exprs.push(lit(scalar));
            }
            exprs.push(column_exprs);
        }

        for placeholder in self.records_with_placeholders.iter() {
            exprs[placeholder.row][placeholder.column] = Arc::clone(&placeholder.expr);
        }

        exprs
    }
}

impl DataSource for ValuesSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition > 0 {
            return exec_err!("ValuesSource only supports a single partition");
        }

        if let Some(record) = self.records_with_placeholders.first() {
            // Return an error, if placeholders are not resolved.
            evaluate_to_scalar(record.expr.as_ref())?;
        }

        Ok(Box::pin(cooperative(MemoryStream::try_new(
            vec![self.batch.clone()],
            Arc::clone(&self.schema),
            None,
        )?)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "placeholders={}", self.records_with_placeholders.len())
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.schema))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(compute_record_batch_statistics(
            &[vec![self.batch.clone()]],
            &self.schema,
            None,
        ))
    }

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        None
    }

    fn fetch(&self) -> Option<usize> {
        None
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &datafusion_physical_expr::projection::ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }

    fn physical_expressions<'a>(
        &'a self,
    ) -> Option<Box<dyn Iterator<Item = Arc<dyn PhysicalExpr>> + 'a>> {
        Some(Box::new(
            self.records_with_placeholders
                .iter()
                .map(|r| Arc::clone(&r.expr)),
        ))
    }

    fn with_physical_expressions(
        &self,
        params: ReplacePhysicalExpr,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        let expected_count = self.records_with_placeholders.len();
        let exprs_count = params.exprs.len();

        assert_eq_or_internal_err!(
            expected_count,
            exprs_count,
            "Inconsistent number of physical expressions for ValuesSource",
        );

        let mut records_with_placeholders = Vec::new();
        let mut column_updates = vec![vec![]; self.schema.fields().len()];
        for (record, expr) in self.records_with_placeholders.iter().zip(params.exprs) {
            if has_placeholders(&expr) {
                records_with_placeholders.push(record.clone());
                continue;
            }

            let scalar = evaluate_to_scalar(expr.as_ref())?;
            column_updates[record.column].push((record.row, scalar));
        }

        let mut columns = self.batch.columns().to_vec();
        for (col_idx, updates) in column_updates.into_iter().enumerate() {
            if updates.is_empty() {
                continue;
            }

            let column = &columns[col_idx];
            let mut scalars = (0..column.len())
                .map(|i| ScalarValue::try_from_array(column, i))
                .collect::<Result<Vec<_>>>()?;

            for (row_idx, scalar) in updates {
                scalars[row_idx] = scalar;
            }

            columns[col_idx] = ScalarValue::iter_to_array(scalars)?;
        }

        let batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
        let data_source = Arc::new(ValuesSource {
            schema: Arc::clone(&self.schema),
            batch,
            records_with_placeholders,
        });

        Ok(Some(data_source))
    }
}

/// Evaluates a physical expression to a scalar value.
fn evaluate_to_scalar(expr: &dyn PhysicalExpr) -> Result<ScalarValue> {
    static PLACEHOLDER_BATCH: LazyLock<RecordBatch> = LazyLock::new(|| {
        let placeholder_schema = Arc::new(Schema::empty());
        RecordBatch::try_new_with_options(
            placeholder_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("Failed to create placeholder batch")
    });

    let result = expr.evaluate(&PLACEHOLDER_BATCH)?;
    match result {
        ColumnarValue::Scalar(scalar) => Ok(scalar),
        ColumnarValue::Array(array) if array.len() == 1 => {
            ScalarValue::try_from_array(&array, 0)
        }
        _ => plan_err!("Cannot have array values in a values list"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::{
        ColumnStatistics, ParamValues, assert_batches_eq, stats::Precision,
    };
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, lit, placeholder};
    use datafusion_physical_plan::{
        ExecutionPlan, collect, reuse::ReusableExecutionPlan,
    };

    #[test]
    fn test_values_source_no_placeholders() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));

        let data = vec![vec![lit(1i32), lit("foo")], vec![lit(2i32), lit("bar")]];

        let exec = ValuesSource::try_new_exec(Arc::clone(&schema), data)?;

        // Should be MemorySourceConfig because no placeholders.
        assert!(exec.data_source().as_any().is::<MemorySourceConfig>());

        Ok(())
    }

    #[test]
    fn test_values_stats_with_nulls_and_placeholders() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("col0", DataType::Int32, true)]));

        let data = vec![
            vec![lit(ScalarValue::Int32(None))],
            vec![lit(ScalarValue::Int32(None))],
            vec![placeholder("$1", DataType::Int32)],
        ];
        let rows = data.len();
        let nulls = rows - 1;

        let values = ValuesSource::try_new_exec(schema, data)?;
        assert!(values.data_source().as_any().is::<ValuesSource>());

        assert_eq!(
            values.partition_statistics(None)?,
            Statistics {
                num_rows: Precision::Exact(rows),
                total_byte_size: Precision::Exact(176), // not important
                column_statistics: vec![ColumnStatistics {
                    null_count: Precision::Exact(nulls), // there are only nulls and placeholders
                    distinct_count: Precision::Absent,
                    max_value: Precision::Absent,
                    min_value: Precision::Absent,
                    sum_value: Precision::Absent,
                    byte_size: Precision::Absent,
                },],
            }
        );

        Ok(())
    }

    // Test issue: https://github.com/apache/datafusion/issues/8763
    #[test]
    fn test_values_with_non_nullable_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col0",
            DataType::UInt32,
            false,
        )]));
        let _ =
            ValuesSource::try_new(Arc::clone(&schema), vec![vec![lit(1u32)]]).unwrap();
        // Test that a null value is rejected
        let _ = ValuesSource::try_new(schema, vec![vec![lit(ScalarValue::UInt32(None))]])
            .unwrap_err();
    }

    #[tokio::test]
    async fn test_values_source_with_placeholders() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Int32, false),
        ]));

        // $1 for the first column, second row.
        let data = vec![
            vec![lit("foo"), lit(1i32)],
            vec![
                lit("bar"),
                Arc::new(BinaryExpr::new(
                    lit(30i32),
                    Operator::Plus,
                    placeholder("$1", DataType::Int32),
                )),
            ],
        ];

        let values_exec = ValuesSource::try_new_exec(Arc::clone(&schema), data)?;

        // Should be ValuesSource because of placeholder.
        assert!(values_exec.data_source().as_any().is::<ValuesSource>());

        let exec = ReusableExecutionPlan::new(values_exec as _);
        let task_ctx = Arc::new(TaskContext::default());

        let batch = collect(
            exec.bind(Some(&ParamValues::List(vec![
                ScalarValue::Int32(Some(10)).into(),
            ])))?
            .plan(),
            task_ctx,
        )
        .await?;
        let expected = [
            "+-----+----+",
            "| a   | b  |",
            "+-----+----+",
            "| foo | 1  |",
            "| bar | 40 |",
            "+-----+----+",
        ];
        assert_batches_eq!(expected, &batch);

        Ok(())
    }

    #[tokio::test]
    async fn test_values_source_multiple_placeholders() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let data: Vec<Vec<Arc<dyn PhysicalExpr>>> = vec![vec![
            placeholder("$1", DataType::Int32),
            placeholder("$2", DataType::Int32),
        ]];

        let values_exec = ValuesSource::try_new_exec(Arc::clone(&schema), data)? as _;
        let exec = ReusableExecutionPlan::new(values_exec);

        let task_ctx = Arc::new(TaskContext::default());

        let batch = collect(
            exec.bind(Some(&ParamValues::List(vec![
                ScalarValue::Int32(Some(10)).into(),
                ScalarValue::Int32(Some(20)).into(),
            ])))?
            .plan(),
            Arc::clone(&task_ctx),
        )
        .await?;
        let expected = [
            "+----+----+",
            "| a  | b  |",
            "+----+----+",
            "| 10 | 20 |",
            "+----+----+",
        ];
        assert_batches_eq!(expected, &batch);

        let batch = collect(
            exec.bind(Some(&ParamValues::List(vec![
                ScalarValue::Int32(Some(30)).into(),
                ScalarValue::Int32(Some(40)).into(),
            ])))?
            .plan(),
            task_ctx,
        )
        .await?;
        let expected = [
            "+----+----+",
            "| a  | b  |",
            "+----+----+",
            "| 30 | 40 |",
            "+----+----+",
        ];
        assert_batches_eq!(expected, &batch);

        Ok(())
    }

    #[test]
    fn test_values_source_empty_data() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let result = ValuesSource::try_new_exec(schema, vec![]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fails_if_not_all_placeholders_resolved() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let data: Vec<Vec<Arc<dyn PhysicalExpr>>> =
            vec![vec![lit(10), placeholder("$foo", DataType::Int32)]];

        let values_exec = ValuesSource::try_new_exec(Arc::clone(&schema), data)? as _;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::clone(&values_exec), task_ctx).await;
        assert!(result.is_err());

        let exec = ReusableExecutionPlan::new(values_exec);

        let task_ctx = Arc::new(TaskContext::default());
        let batch = collect(
            exec.bind(Some(&ParamValues::Map(HashMap::from_iter([(
                "foo".to_string(),
                ScalarValue::Int32(Some(20)).into(),
            )]))))?
            .plan(),
            task_ctx,
        )
        .await?;
        let expected = [
            "+----+----+",
            "| a  | b  |",
            "+----+----+",
            "| 10 | 20 |",
            "+----+----+",
        ];
        assert_batches_eq!(expected, &batch);

        Ok(())
    }
}
