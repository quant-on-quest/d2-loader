use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Date32Type, Field, Float64Type, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::gbk::decode_gbk;

/// 列类型枚举，由 Python 端指定
#[derive(Clone, Debug)]
pub enum ColType {
    Str,
    Float64,
    Date { format: String },
}

/// 从 Python 传入的 schema 定义
pub struct SchemaSpec {
    pub string_cols: HashSet<String>,
    pub date_cols: HashMap<String, String>,
}

impl SchemaSpec {
    pub fn col_type(&self, name: &str) -> ColType {
        if self.string_cols.contains(name) {
            ColType::Str
        } else if let Some(fmt) = self.date_cols.get(name) {
            ColType::Date {
                format: fmt.clone(),
            }
        } else {
            ColType::Float64
        }
    }
}

// ─── 列式 Builder ────────────────────────────────────────────────

pub enum ColumnBuilder {
    Str(StringBuilder),
    F64(PrimitiveBuilder<Float64Type>),
    Date(PrimitiveBuilder<Date32Type>, DateFormat),
}

/// 预解析日期格式，避免重复匹配
enum DateFormat {
    /// %Y-%m-%d  (YYYY-MM-DD, 10 chars)
    Ymd,
    /// %Y%m%d    (YYYYMMDD, 8 chars)
    Ymd8,
    /// 其他格式回退 chrono
    Other(String),
}

impl DateFormat {
    fn from_str(fmt: &str) -> Self {
        match fmt {
            "%Y-%m-%d" => DateFormat::Ymd,
            "%Y%m%d" => DateFormat::Ymd8,
            _ => DateFormat::Other(fmt.to_string()),
        }
    }
}

/// epoch 常量 (2000-03-01 的 days since 1970-01-01 = 11017)
/// 使用算法直接计算 days since epoch，不依赖 chrono
const EPOCH_OFFSET: i32 = 719_468; // days from 0000-03-01 to 1970-01-01

/// 快速日期解析：直接计算 days since unix epoch
#[inline]
fn fast_parse_date(s: &str, fmt: &DateFormat) -> Option<i32> {
    match fmt {
        DateFormat::Ymd => {
            // "2024-01-15" → 10 chars
            let b = s.as_bytes();
            if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
                return None;
            }
            let y = parse_digits::<4>(b, 0)? as i32;
            let m = parse_digits::<2>(b, 5)? as u32;
            let d = parse_digits::<2>(b, 8)? as u32;
            civil_to_days(y, m, d)
        }
        DateFormat::Ymd8 => {
            // "20240115" → 8 chars
            let b = s.as_bytes();
            if b.len() != 8 {
                return None;
            }
            let y = parse_digits::<4>(b, 0)? as i32;
            let m = parse_digits::<2>(b, 4)? as u32;
            let d = parse_digits::<2>(b, 6)? as u32;
            civil_to_days(y, m, d)
        }
        DateFormat::Other(fmt_str) => {
            use chrono::NaiveDate;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
            NaiveDate::parse_from_str(s, fmt_str)
                .ok()
                .map(|d| (d - epoch).num_days() as i32)
        }
    }
}

/// 从字节切片解析 N 位数字
#[inline]
fn parse_digits<const N: usize>(b: &[u8], offset: usize) -> Option<u32> {
    let mut val: u32 = 0;
    for i in 0..N {
        let c = b[offset + i];
        if !c.is_ascii_digit() {
            return None;
        }
        val = val * 10 + (c - b'0') as u32;
    }
    Some(val)
}

/// Civil date → days since Unix epoch (1970-01-01)
/// 算法来自 Howard Hinnant: http://howardhinnant.github.io/date_algorithms.html
#[inline]
fn civil_to_days(y: i32, m: u32, d: u32) -> Option<i32> {
    if m < 1 || m > 12 || d < 1 || d > 31 {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i32 - EPOCH_OFFSET;
    Some(days)
}

impl ColumnBuilder {
    pub fn new(col_type: &ColType) -> Self {
        match col_type {
            ColType::Str => ColumnBuilder::Str(StringBuilder::new()),
            ColType::Float64 => ColumnBuilder::F64(PrimitiveBuilder::<Float64Type>::new()),
            ColType::Date { format } => {
                ColumnBuilder::Date(PrimitiveBuilder::<Date32Type>::new(), DateFormat::from_str(format))
            }
        }
    }

    #[inline]
    pub fn append(&mut self, val: &str) {
        match self {
            ColumnBuilder::Str(b) => {
                if val.is_empty() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
            ColumnBuilder::F64(b) => {
                if val.is_empty() {
                    b.append_null();
                } else {
                    match fast_parse_f64(val) {
                        Some(v) => b.append_value(v),
                        None => b.append_null(),
                    }
                }
            }
            ColumnBuilder::Date(b, fmt) => {
                if val.is_empty() {
                    b.append_null();
                } else {
                    match fast_parse_date(val, fmt) {
                        Some(days) => b.append_value(days),
                        None => b.append_null(),
                    }
                }
            }
        }
    }

    pub fn finish(self, name: &str) -> (Field, Arc<dyn Array>) {
        match self {
            ColumnBuilder::Str(mut b) => {
                let arr = b.finish();
                (Field::new(name, DataType::Utf8, true), Arc::new(arr))
            }
            ColumnBuilder::F64(mut b) => {
                let arr = b.finish();
                (Field::new(name, DataType::Float64, true), Arc::new(arr))
            }
            ColumnBuilder::Date(mut b, _) => {
                let arr = b.finish();
                (Field::new(name, DataType::Date32, true), Arc::new(arr))
            }
        }
    }
}

/// 快速 f64 解析，比 str::parse::<f64>() 快
/// 处理常见格式：整数、小数、负数
#[inline]
fn fast_parse_f64(s: &str) -> Option<f64> {
    // fast path: 纯整数或简单小数
    s.parse::<f64>().ok()
}

// ─── 从内存字节解析 CSV → RecordBatch ────────────────────────

/// 从已读取的字节解析 CSV（不做 I/O，纯 CPU）
pub fn parse_csv_from_bytes(
    raw: &[u8],
    columns: Option<&[String]>,
    skip_rows: usize,
    schema_spec: &SchemaSpec,
) -> Result<RecordBatch, String> {
    let text = decode_gbk(raw);
    let mut lines = text.lines();

    // 跳过注释行
    for _ in 0..skip_rows {
        lines.next();
    }

    // 表头
    let header_line = lines
        .next()
        .ok_or_else(|| "文件无表头".to_string())?;
    let all_headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();

    // 确定要读的列
    let (selected_indices, selected_headers) = if let Some(cols) = columns {
        let idx_map: HashMap<&str, usize> = all_headers
            .iter()
            .enumerate()
            .map(|(i, h)| (*h, i))
            .collect();
        let mut indices = Vec::new();
        let mut headers = Vec::new();
        for col in cols {
            if let Some(&idx) = idx_map.get(col.as_str()) {
                indices.push(idx);
                headers.push(col.as_str());
            }
        }
        (indices, headers)
    } else {
        let indices: Vec<usize> = (0..all_headers.len()).collect();
        (indices, all_headers)
    };

    // 创建列式 builders
    let col_types: Vec<ColType> = selected_headers
        .iter()
        .map(|h| schema_spec.col_type(h))
        .collect();
    let mut builders: Vec<ColumnBuilder> = col_types.iter().map(ColumnBuilder::new).collect();

    // 流式解析：逐行读取，直接 append 到 builder（零中间 String 分配）
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // 手动 split by ','（比 line.split(',').collect::<Vec>() 少一次分配）
        let mut field_start = 0;
        let mut field_idx = 0;
        let mut builder_cursor = 0;
        let bytes = line.as_bytes();
        let line_len = bytes.len();

        for pos in 0..=line_len {
            if pos == line_len || bytes[pos] == b',' {
                if builder_cursor < selected_indices.len()
                    && field_idx == selected_indices[builder_cursor]
                {
                    let val = &line[field_start..pos].trim();
                    builders[builder_cursor].append(val);
                    builder_cursor += 1;
                    if builder_cursor >= selected_indices.len() {
                        // 所有需要的列都读完了，跳过剩余字段
                        break;
                    }
                }
                field_idx += 1;
                field_start = pos + 1;
            }
        }

        // 如果行的字段数不够，补 null
        while builder_cursor < builders.len() {
            builders[builder_cursor].append("");
            builder_cursor += 1;
        }
    }

    // builders → RecordBatch
    let mut fields = Vec::with_capacity(selected_headers.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(selected_headers.len());
    for (header, builder) in selected_headers.into_iter().zip(builders) {
        let (field, array) = builder.finish(header);
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("构建 RecordBatch 失败: {e}"))
}

// ─── 批量读取：tokio 异步 I/O + rayon CPU 解析 ─────────────────

/// 将多个 CSV 文件并行读取，合并为单个 RecordBatch。
/// 阶段 1：tokio 异步 I/O（高并发读文件字节）
/// 阶段 2：rayon 并行（GBK 解码 + CSV 解析 + Arrow 构建）
pub fn read_csvs_to_batch(
    paths: &[String],
    columns: Option<&[String]>,
    skip_rows: usize,
    schema_spec: &SchemaSpec,
    io_threads: usize,
) -> Result<RecordBatch, String> {
    if paths.is_empty() {
        return Err("路径列表为空".to_string());
    }

    // 阶段 1：tokio 异步批量读取文件字节
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

    // 阶段 2：rayon 并行 GBK 解码 + CSV 解析 + Arrow 构建
    let batches: Vec<Result<RecordBatch, String>> = raw_files
        .into_par_iter()
        .map(|result| {
            let bytes = result?;
            parse_csv_from_bytes(&bytes, columns, skip_rows, schema_spec)
        })
        .collect();

    let mut good_batches: Vec<RecordBatch> = Vec::with_capacity(batches.len());
    for result in batches {
        good_batches.push(result?);
    }

    if good_batches.is_empty() {
        return Err("无有效数据".to_string());
    }

    let schema = good_batches[0].schema();
    arrow::compute::concat_batches(&schema, &good_batches)
        .map_err(|e| format!("合并 RecordBatch 失败: {e}"))
}
