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
    quoting=True,                       # 是否处理双引号
)

# 异构 schema（不同文件列不同，自动 diagonal concat）
df = d2_loader.read_csvs_diagonal(
    paths=["a.csv", "b.csv"],
    renames={"stock_code": "code"},     # 列重命名
)
```

Schema 类型：`"str"` 字符串、`"date:%Y-%m-%d"` 日期、`"float64"` 浮点数（默认）。

## 引号处理

`quoting=True`（默认）：

| 输入 | 结果 |
| --- | --- |
| `"a,b"` | `a,b` —— 字段内的逗号不切分 |
| `"a""b"` | `a"b` —— RFC4180 双写转义 |
| `"a\nb"` | `a\nb` —— 引号内的换行不断行 |
| `"a"b"c"` | `a"b"c` —— 闭合引号后不是分隔符时按字面引号处理 |

最后一条是给没有按 RFC4180 转义的数据源兜底：这种输入下 polars 会把内部引号直接丢掉，
这里选择一个字符都不丢。

`quoting=False`：引号只是普通字面字符，只按逗号和换行切分（0.1.x 的行为）。数据源自己有一套
非标准引号规则、希望原样读进来时用它。

## 错误处理

所有数据错误都是普通的 `RuntimeError`，可以被 `except Exception` 捕获，错误信息带文件路径。
Rust 侧的 panic 也会被转成 `RuntimeError`，不会再以 `PanicException`（继承自 `BaseException`）
的形式打死调用方进程。

各文件列数、列序不一致时按列名取并集对齐，缺失列补 null，不会中断整批读取。

## 变更

### 0.2.0

- 支持引号：`"a,b"` 不再被逗号切碎，跨行引号字段不再被拆成多行，首尾包裹引号会被剥掉
- 新增 `quoting` 参数
- 修复列数不齐时 `concat_batches` 越界 panic（`PanicException` 穿透 `except Exception`）
- 错误信息带上文件路径；`read_csvs_diagonal` 不再把单文件解析失败静默跳过

行情数据（2000 个文件 / 671 万行 / 38 列）实测与 0.1.0 持平，财务数据（5824 个文件 / 372 列）
快约 15%，输出逐值一致。

## License

MIT
