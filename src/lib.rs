pub mod batch_util;
pub mod chunked_io;
pub mod csv_scan;
pub mod fina_reader;
pub mod gbk;
pub mod stock_reader;

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};

use arrow::ipc::writer::StreamWriter;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use stock_reader::{ColType, ParseOptions, SchemaSpec};

/// 把 "str" / "int64" / "float64" / "date:FMT" 解析成 ColType
fn parse_col_type(typ: &str) -> PyResult<ColType> {
    match typ {
        "str" => Ok(ColType::Str),
        "int64" => Ok(ColType::Int64),
        "float64" => Ok(ColType::Float64),
        other => match other.strip_prefix("date:") {
            Some(fmt) => Ok(ColType::Date {
                format: fmt.to_string(),
            }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "不认识的列类型 {other:?}，可用: \"str\" / \"int64\" / \"float64\" / \"date:%Y-%m-%d\""
            ))),
        },
    }
}

/// 从 Python dict 构建 SchemaSpec
///
/// schema 格式：
///     {"列名": "str", "列名": "date:%Y-%m-%d", "列名": "int64", "列名": "float64"}
///     未指定的列走 default_type
fn parse_schema(overrides: Option<&Bound<PyDict>>, default_type: &str) -> PyResult<SchemaSpec> {
    let mut string_cols = HashSet::new();
    let mut date_cols = HashMap::new();
    let mut int_cols = HashSet::new();
    let mut float_cols = HashSet::new();

    if let Some(d) = overrides {
        for (key, val) in d.iter() {
            let col: String = key.extract()?;
            let typ: String = val.extract()?;
            match parse_col_type(&typ)? {
                ColType::Str => {
                    string_cols.insert(col);
                }
                ColType::Int64 => {
                    int_cols.insert(col);
                }
                ColType::Float64 => {
                    float_cols.insert(col);
                }
                ColType::Date { format } => {
                    date_cols.insert(col, format);
                }
            }
        }
    }

    Ok(SchemaSpec {
        string_cols,
        date_cols,
        int_cols,
        float_cols,
        default_type: parse_col_type(default_type)?,
    })
}

/// 把 Rust 侧的工作包起来，任何 panic 都转成普通的 RuntimeError。
///
/// pyo3 默认把 panic 转成 PanicException，它继承 BaseException，
/// Python 侧的 `except Exception` 兜不住，会直接打死调用方进程。
fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "未知错误".to_string());
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "读取 CSV 时发生内部错误: {msg}"
            )))
        }
    }
}

/// Arrow RecordBatch 列表 → IPC bytes → pyarrow → polars DataFrame
///
/// 多个 batch 直接写进同一个 IPC 流，pyarrow 读成 chunked Table，
/// 省掉先 concat 成一整块的那份完整拷贝。
fn batches_to_py(
    py: Python<'_>,
    schema: arrow::datatypes::SchemaRef,
    batches: Vec<arrow::record_batch::RecordBatch>,
) -> PyResult<PyObject> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC writer: {e}")))?;
        for batch in &batches {
            writer
                .write(batch)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC write: {e}")))?;
        }
        writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC finish: {e}")))?;
    }
    drop(batches);

    let bytes = pyo3::types::PyBytes::new(py, &buf);
    let pa = py.import("pyarrow")?;
    let ipc = pa.getattr("ipc")?;
    let reader = ipc.call_method1("open_stream", (bytes,))?;
    let table = reader.call_method0("read_all")?;

    let pl = py.import("polars")?;
    let df = pl.call_method1("from_arrow", (table,))?;
    Ok(df.into_pyobject(py)?.into_any().unbind())
}

/// 批量读取 GBK 编码的 CSV 文件
///
/// Args:
///     paths: CSV 文件路径列表
///     columns: 可选，要读取的列名列表（按列名映射，顺序任意；
///         输出列顺序与本参数一致，文件里没有的列输出为整列 null）
///     skip_rows: 跳过文件开头的行数（默认 1，跳过注释行）
///     schema: 可选，列类型定义 dict
///         - "str": 字符串
///         - "date:%Y-%m-%d": 日期（指定格式）
///         - "float64": 浮点数（默认）
///     default_type: 未在 schema 中声明的列按什么类型解析（默认 "float64"）
///         未声明的列如果非空值全部解析失败，会直接报错而不是静默产出全 null
///     trim: 是否剥离字符串字段的首尾空白（默认 True）
///         数值/日期列不受影响，始终按 trim 后的值解析
///     quoting: 是否处理双引号（默认 True）
///         - True: `"a,b"` 是一个字段，`""` 是转义引号，引号内换行不断行；
///                 闭合引号后不是分隔符时按字面引号处理（不丢字符），
///                 适配没有按 RFC4180 转义的数据源
///         - False: 引号是普通字面字符，只按逗号和换行切分
///
/// Returns:
///     polars.DataFrame
///
/// Example:
///     df = d2_loader.read_csvs(
///         paths,
///         columns=["股票代码", "交易日期", "收盘价"],
///         schema={"股票代码": "str", "交易日期": "date:%Y-%m-%d"}
///     )
#[pyfunction]
#[pyo3(signature = (paths, columns=None, skip_rows=1, schema=None, io_threads=256, quoting=true, default_type="float64", trim=true))]
fn read_csvs(
    py: Python<'_>,
    paths: &Bound<PyList>,
    columns: Option<Vec<String>>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    io_threads: usize,
    quoting: bool,
    default_type: &str,
    trim: bool,
) -> PyResult<PyObject> {
    let paths: Vec<String> = paths.extract()?;
    let schema_spec = parse_schema(schema, default_type)?;
    let opts = ParseOptions {
        skip_rows,
        quoting,
        trim,
    };

    let (schema, batches) = guard(|| {
        stock_reader::read_csvs_to_batches(
            &paths,
            columns.as_deref(),
            &schema_spec,
            &opts,
            io_threads,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    })?;

    batches_to_py(py, schema, batches)
}

/// 批量读取异构 schema 的 GBK CSV 文件（如财务数据）
///
/// 不同文件列可以不同，自动 diagonal concat 补 null。
///
/// Args:
///     paths: CSV 文件路径列表
///     skip_rows: 跳过文件开头的行数（默认 1）
///     schema: 可选，列类型定义 dict（同 read_csvs）
///     renames: 可选，列重命名 dict（如 {"stock_code": "code"}）
///     quoting / default_type / trim: 同 read_csvs
///
/// Returns:
///     polars.DataFrame
#[pyfunction]
#[pyo3(signature = (paths, skip_rows=1, schema=None, renames=None, io_threads=256, quoting=true, default_type="float64", trim=true))]
fn read_csvs_diagonal(
    py: Python<'_>,
    paths: &Bound<PyList>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    renames: Option<&Bound<PyDict>>,
    io_threads: usize,
    quoting: bool,
    default_type: &str,
    trim: bool,
) -> PyResult<PyObject> {
    let paths: Vec<String> = paths.extract()?;
    let schema_spec = parse_schema(schema, default_type)?;
    let opts = ParseOptions {
        skip_rows,
        quoting,
        trim,
    };

    let rename_map: HashMap<String, String> = if let Some(r) = renames {
        r.iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<String>()?)))
            .collect::<PyResult<_>>()?
    } else {
        HashMap::new()
    };

    let (schema, batches) = guard(|| {
        fina_reader::read_fina_csvs_to_batches(
            &paths,
            &schema_spec,
            &rename_map,
            &opts,
            io_threads,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    })?;

    batches_to_py(py, schema, batches)
}

// === 向后兼容旧 API ===

/// 向后兼容：read_stock_csvs
#[pyfunction]
#[pyo3(signature = (paths, columns=None))]
fn read_stock_csvs(
    py: Python<'_>,
    paths: &Bound<PyList>,
    columns: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // 使用框架默认 schema
    let schema_dict = PyDict::new(py);
    for col in &[
        "股票代码", "股票名称",
        "申万一级行业名称", "申万二级行业名称", "申万三级行业名称",
        "新版申万一级行业名称", "新版申万二级行业名称", "新版申万三级行业名称",
        "沪深300成分股", "上证50成分股", "中证500成分股",
        "中证1000成分股", "中证2000成分股", "创业板指成分股",
    ] {
        schema_dict.set_item(*col, "str")?;
    }
    schema_dict.set_item("交易日期", "date:%Y-%m-%d")?;

    read_csvs(py, paths, columns, 1, Some(&schema_dict), 256, true, "float64", true)
}

/// 向后兼容：read_fina_csvs
#[pyfunction]
fn read_fina_csvs(py: Python<'_>, paths: &Bound<PyList>) -> PyResult<PyObject> {
    let schema_dict = PyDict::new(py);
    for col in &["code", "stock_code", "statement_format"] {
        schema_dict.set_item(*col, "str")?;
    }
    schema_dict.set_item("report_date", "date:%Y%m%d")?;
    schema_dict.set_item("publish_date", "date:%Y-%m-%d")?;

    let renames = PyDict::new(py);
    renames.set_item("stock_code", "code")?;

    read_csvs_diagonal(py, paths, 1, Some(&schema_dict), Some(&renames), 256, true, "float64", true)
}

/// d2_loader - Rust 加速的 GBK CSV 批量加载器
#[pymodule]
fn d2_loader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_csvs, m)?)?;
    m.add_function(wrap_pyfunction!(read_csvs_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(read_stock_csvs, m)?)?;
    m.add_function(wrap_pyfunction!(read_fina_csvs, m)?)?;
    Ok(())
}
