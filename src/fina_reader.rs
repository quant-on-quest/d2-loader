use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Date32Type, Field, Float64Type, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::gbk::decode_gbk;
use crate::stock_reader::{ColType, ColumnBuilder, SchemaSpec};

/// 排除的列前缀
const EXCLUDE_PREFIXES: &[&str] = &["Unnamed", "抓取时间"];

/// 从字节解析单个财务 CSV → RecordBatch
fn parse_fina_csv_from_bytes(
    raw: &[u8],
    skip_rows: usize,
    schema_spec: &SchemaSpec,
    renames: &HashMap<String, String>,
) -> Result<RecordBatch, String> {
    let text = decode_gbk(raw);
    let mut lines = text.lines();

    for _ in 0..skip_rows {
        lines.next();
    }

    let header_line = lines
        .next()
        .ok_or_else(|| "文件无表头".to_string())?;
    let raw_headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();

    // 过滤排除列，确定保留的列索引和最终列名
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut final_headers: Vec<String> = Vec::new();
    for (i, h) in raw_headers.iter().enumerate() {
        if h.is_empty() || EXCLUDE_PREFIXES.iter().any(|prefix| h.starts_with(prefix)) {
            continue;
        }
        keep_indices.push(i);
        let name = renames.get(*h).cloned().unwrap_or_else(|| h.to_string());
        final_headers.push(name);
    }

    // 创建 builders
    let col_types: Vec<ColType> = final_headers
        .iter()
        .map(|h| schema_spec.col_type(h))
        .collect();
    let mut builders: Vec<ColumnBuilder> = col_types.iter().map(ColumnBuilder::new).collect();

    // 流式解析
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // 按 ',' 分割字段
        let fields: Vec<&str> = line.split(',').collect();
        for (builder_idx, &col_idx) in keep_indices.iter().enumerate() {
            let val = fields.get(col_idx).map(|s| s.trim()).unwrap_or("");
            builders[builder_idx].append(val);
        }
    }

    // builders → RecordBatch
    let mut fields = Vec::with_capacity(final_headers.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(final_headers.len());
    for (header, builder) in final_headers.iter().zip(builders) {
        let (field, array) = builder.finish(header);
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("构建 RecordBatch 失败: {e}"))
}

/// 批量读取财务 CSV（异构 schema，diagonal concat）
/// tokio 异步 I/O + rayon CPU 解析
pub fn read_fina_csvs_to_batch(
    paths: &[String],
    skip_rows: usize,
    schema_spec: &SchemaSpec,
    renames: &HashMap<String, String>,
    io_threads: usize,
) -> Result<RecordBatch, String> {
    if paths.is_empty() {
        return Err("路径列表为空".to_string());
    }

    // 阶段 1：tokio 异步批量读取
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(io_threads)
        .enable_all()
        .build()
        .map_err(|e| format!("创建 tokio runtime 失败: {e}"))?;

    let raw_files: Vec<Result<Vec<u8>, String>> = rt.block_on(async {
        let tasks: Vec<_> = paths
            .iter()
            .map(|p| {
                let p = p.clone();
                tokio::spawn(async move {
                    tokio::fs::read(&p)
                        .await
                        .map_err(|e| format!("读取文件失败 {p}: {e}"))
                })
            })
            .collect();
        futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(|r| r.map_err(|e| format!("task 失败: {e}")).and_then(|v| v))
            .collect()
    });

    // 阶段 2：rayon 并行解析
    let results: Vec<Result<RecordBatch, String>> = raw_files
        .into_par_iter()
        .map(|result| {
            let bytes = result?;
            parse_fina_csv_from_bytes(&bytes, skip_rows, schema_spec, renames)
        })
        .collect();

    let mut batches: Vec<RecordBatch> = Vec::new();
    for r in results {
        match r {
            Ok(batch) if batch.num_rows() > 0 => batches.push(batch),
            Ok(_) => {} // 空 batch 跳过
            Err(e) => eprintln!("警告: {e}"),
        }
    }

    if batches.is_empty() {
        return Err("无有效数据".to_string());
    }

    // 收集全局列名（保持出现顺序）
    let mut all_col_order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for batch in &batches {
        for field in batch.schema().fields() {
            let name = field.name().clone();
            if seen.insert(name.clone()) {
                all_col_order.push(name);
            }
        }
    }

    // 构建全局 schema
    let global_fields: Vec<Field> = all_col_order
        .iter()
        .map(|name| {
            // 找到该列在某个 batch 中的类型
            for batch in &batches {
                if let Some((idx, _)) = batch.schema().column_with_name(name) {
                    return batch.schema().field(idx).clone();
                }
            }
            // 全部 batch 都没有该列（不应发生）→ 默认 Float64
            Field::new(name, DataType::Float64, true)
        })
        .collect();
    let global_schema = Arc::new(Schema::new(global_fields));

    // 对齐每个 batch 到全局 schema（缺失列补 null）
    let aligned: Vec<RecordBatch> = batches
        .into_iter()
        .map(|batch| align_batch_to_schema(&batch, &global_schema))
        .collect::<Result<Vec<_>, _>>()?;

    arrow::compute::concat_batches(&global_schema, &aligned)
        .map_err(|e| format!("合并 RecordBatch 失败: {e}"))
}

/// 将 batch 对齐到目标 schema：补缺失列为 null 数组
fn align_batch_to_schema(
    batch: &RecordBatch,
    target_schema: &Arc<Schema>,
) -> Result<RecordBatch, String> {
    let num_rows = batch.num_rows();
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(target_schema.fields().len());

    for field in target_schema.fields() {
        if let Some((idx, _)) = batch.schema().column_with_name(field.name()) {
            let col = batch.column(idx);
            // 类型一致直接用，否则 cast
            if col.data_type() == field.data_type() {
                columns.push(col.clone());
            } else {
                // 尝试 cast（如 Int64 → Float64）
                match arrow::compute::cast(col.as_ref(), field.data_type()) {
                    Ok(casted) => columns.push(casted),
                    Err(_) => {
                        // cast 失败，填 null
                        columns.push(new_null_array(field.data_type(), num_rows));
                    }
                }
            }
        } else {
            // 该列在此 batch 中不存在，补 null
            columns.push(new_null_array(field.data_type(), num_rows));
        }
    }

    RecordBatch::try_new(target_schema.clone(), columns)
        .map_err(|e| format!("对齐 RecordBatch 失败: {e}"))
}
