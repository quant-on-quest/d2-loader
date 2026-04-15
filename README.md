# gbk-csv-loader

Rust 加速的 GBK 编码 CSV 批量加载器。

使用 tokio 异步 I/O + rayon 并行 CPU 处理 + encoding_rs SIMD GBK 解码，比 Polars Python 层快 4.8x。

## 安装

```bash
pip install gbk-csv-loader
# 或
uv add gbk-csv-loader
```

预编译 wheel 支持：Linux (x86_64/aarch64)、macOS (Intel/ARM)、Windows (x64)、Python 3.11-3.13。

## 使用

```python
import d2_loader

# 批量读取 GBK CSV
df = d2_loader.read_csvs(
    paths=["file1.csv", "file2.csv", ...],
    columns=["col1", "col2"],          # 可选列筛选
    skip_rows=1,                        # 跳过注释行
    schema={"col1": "str", "col2": "date:%Y-%m-%d"},  # 列类型
    io_threads=256,                     # I/O 并发线程数
)

# 异构 schema（不同文件列不同，自动 diagonal concat）
df = d2_loader.read_csvs_diagonal(
    paths=["a.csv", "b.csv"],
    renames={"stock_code": "code"},     # 列重命名
)
```

Schema 类型：`"str"` 字符串、`"date:%Y-%m-%d"` 日期、`"float64"` 浮点数（默认）。

## License

MIT
