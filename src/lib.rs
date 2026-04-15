pub mod fina_reader;
pub mod gbk;
pub mod stock_reader;

use std::collections::{HashMap, HashSet};

use arrow::ipc::writer::StreamWriter;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use stock_reader::SchemaSpec;

/// 从 Python dict 构建 SchemaSpec
///
/// schema_overrides 格式：
///     {"列名": "str", "列名": "date:%Y-%m-%d", "列名": "float64"}
///     未指定的列默认当 float64
fn parse_schema(overrides: Option<&Bound<PyDict>>) -> PyResult<SchemaSpec> {
    let mut string_cols = HashSet::new();
    let mut date_cols = HashMap::new();

    if let Some(d) = overrides {
        for (key, val) in d.iter() {
            let col: String = key.extract()?;
            let typ: String = val.extract()?;
            if typ == "str" {
                string_cols.insert(col);
            } else if let Some(fmt) = typ.strip_prefix("date:") {
                date_cols.insert(col, fmt.to_string());
            }
            // "float64" 或其他 → 走默认
        }
    }

    Ok(SchemaSpec {
        string_cols,
        date_cols,
    })
}

/// Arrow RecordBatch → IPC bytes → pyarrow → polars DataFrame
fn batch_to_py(py: Python<'_>, batch: arrow::record_batch::RecordBatch) -> PyResult<PyObject> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC writer: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC write: {e}")))?;
        writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC finish: {e}")))?;
    }

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
///     columns: 可选，要读取的列名列表
///     skip_rows: 跳过文件开头的行数（默认 1，跳过注释行）
///     schema: 可选，列类型定义 dict
///         - "str": 字符串
///         - "date:%Y-%m-%d": 日期（指定格式）
///         - "float64": 浮点数（默认）
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
#[pyo3(signature = (paths, columns=None, skip_rows=1, schema=None, io_threads=256))]
fn read_csvs(
    py: Python<'_>,
    paths: &Bound<PyList>,
    columns: Option<Vec<String>>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    io_threads: usize,
) -> PyResult<PyObject> {
    let paths: Vec<String> = paths.extract()?;
    let schema_spec = parse_schema(schema)?;
    let cols_ref = columns.as_deref();

    let batch = stock_reader::read_csvs_to_batch(&paths, cols_ref, skip_rows, &schema_spec, io_threads)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    batch_to_py(py, batch)
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
///
/// Returns:
///     polars.DataFrame
#[pyfunction]
#[pyo3(signature = (paths, skip_rows=1, schema=None, renames=None, io_threads=256))]
fn read_csvs_diagonal(
    py: Python<'_>,
    paths: &Bound<PyList>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    renames: Option<&Bound<PyDict>>,
    io_threads: usize,
) -> PyResult<PyObject> {
    let paths: Vec<String> = paths.extract()?;
    let schema_spec = parse_schema(schema)?;

    let rename_map: HashMap<String, String> = if let Some(r) = renames {
        r.iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<String>()?)))
            .collect::<PyResult<_>>()?
    } else {
        HashMap::new()
    };

    let batch =
        fina_reader::read_fina_csvs_to_batch(&paths, skip_rows, &schema_spec, &rename_map, io_threads)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    batch_to_py(py, batch)
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

    read_csvs(py, paths, columns, 1, Some(&schema_dict), 256)
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

    read_csvs_diagonal(py, paths, 1, Some(&schema_dict), Some(&renames), 256)
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
