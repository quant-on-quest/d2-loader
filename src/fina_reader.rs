use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::batch_util::concat_aligned;
use crate::csv_scan::Scanner;
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
    quoting: bool,
) -> Result<RecordBatch, String> {
    let text = decode_gbk(raw);
    let mut scanner = Scanner::new(&text, skip_rows, quoting);

    let raw_headers = scanner.read_row().ok_or_else(|| "文件无表头".to_string())?;

    // 过滤排除列，确定保留的列索引和最终列名
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut final_headers: Vec<String> = Vec::new();
    for (i, h) in raw_headers.iter().enumerate() {
        if h.is_empty() || EXCLUDE_PREFIXES.iter().any(|prefix| h.starts_with(prefix)) {
            continue;
        }
        keep_indices.push(i);
        let name = renames.get(h).cloned().unwrap_or_else(|| h.to_string());
        final_headers.push(name);
    }

    // 创建 builders
    let col_types: Vec<ColType> = final_headers
        .iter()
        .map(|h| schema_spec.col_type(h))
        .collect();
    let mut builders: Vec<ColumnBuilder> = col_types.iter().map(ColumnBuilder::new).collect();

    // 流式解析
    let mut cursor;
    loop {
        cursor = 0;
        let got = scanner.next_record(|field_idx, val| {
            if cursor < keep_indices.len() && field_idx == keep_indices[cursor] {
                builders[cursor].append(val);
                cursor += 1;
            }
            cursor < keep_indices.len()
        });
        if !got {
            break;
        }
        // 字段数不够的行补 null
        while cursor < builders.len() {
            builders[cursor].append("");
            cursor += 1;
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
    quoting: bool,
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
        .zip(paths.par_iter())
        .map(|(result, path)| {
            let bytes = result?;
            parse_fina_csv_from_bytes(&bytes, skip_rows, schema_spec, renames, quoting)
                .map_err(|e| format!("{path}: {e}"))
        })
        .collect();

    // 解析失败不再只打 stderr 后静默跳过——那会让调用方拿到少了行的结果还以为成功
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(results.len());
    for r in results {
        let batch = r?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }

    // 列名并集 + 缺失列补 null（diagonal concat）
    concat_aligned(&batches)
}
