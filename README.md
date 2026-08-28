# gbk-csv-loader

Rust 加速的 GBK 编码 CSV 批量加载器。

使用 tokio 异步 I/O + rayon 并行 CPU 处理 + encoding_rs SIMD GBK 解码，比 Polars Python 层快 4.8x。

## 安装

```bash
pip install gbk-csv-loader
# 或
uv add gbk-csv-loader
```

预编译 wheel 支持：Linux (x86_64/aarch64)、macOS (Intel/ARM)、Windows (x64)。

wheel 是 abi3 的（`cp311-abi3`），一个文件覆盖 CPython 3.11 及之后的所有版本，包括 3.14 —— 新 Python 发布时不需要等新 wheel。

## 使用

```python
import d2_loader

# 批量读取 GBK CSV
df = d2_loader.read_csvs(
    paths=["file1.csv", "file2.csv", ...],
    columns=["col1", "col2"],          # 可选列筛选，按列名映射，顺序任意
    skip_rows=1,                        # 跳过注释行
    schema={"col1": "str", "col2": "date:%Y-%m-%d"},  # 列类型
    io_threads=256,                     # I/O 并发线程数
    quoting=True,                       # 是否处理双引号
    default_type="float64",             # 未声明列的类型
    trim=True,                          # 是否剥离字符串字段首尾空白
)

# 异构 schema（不同文件列不同，自动 diagonal concat）
df = d2_loader.read_csvs_diagonal(
    paths=["a.csv", "b.csv"],
    renames={"stock_code": "code"},     # 列重命名
)
```

Schema 类型：`"str"` 字符串、`"int64"` 整数、`"date:%Y-%m-%d"` 日期、`"float64"` 浮点数（默认）。

大整数（股本、金额）用 `"int64"`，float64 超过 2^53 会丢精度。

## 列筛选

`columns` 按**列名**映射，顺序随便写，输出列顺序与 `columns` 一致，跟文件里的列序无关。
某个文件里没有这一列时输出为整列 null，不会被悄悄丢掉。

## 未声明列的类型

没在 `schema` 里声明的列走 `default_type`（默认 `"float64"`）。

如果一个**未声明**的列非空值全部解析失败，会直接报错并指出列名、文件和示例值——
这种情况基本都是漏声明了字符串列，以前会静默产出一整列 null：

```
列 '公告标题' 未在 schema 中声明，按 float64 解析，但 1138 个非空值全部解析失败
（示例值: '某公司2024年年度报告'）。请在 schema 里把它声明为 "str"，或调用时传 default_type="str"
```

显式声明成 `"float64"` / `"int64"` 的列不受影响，解析失败照旧填 null。

想要 polars 那样"未声明就当字符串"的语义，传 `default_type="str"`。

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

## 与 polars 的已知差异

| | 本库 | polars `read_csv` |
| --- | --- | --- |
| 字符串字段首尾空白 | 默认剥离，`trim=False` 保留字面值 | 保留字面值 |
| 空行 | 跳过 | 单列文件里当作一行 null |
| 未按 RFC4180 转义的内部引号 | 原样保留 | 丢弃 |

## 错误处理

所有数据错误都是普通的 `RuntimeError`，可以被 `except Exception` 捕获，错误信息带文件路径。
Rust 侧的 panic 也会被转成 `RuntimeError`，不会再以 `PanicException`（继承自 `BaseException`）
的形式打死调用方进程。

各文件列数、列序不一致时按列名取并集对齐，缺失列补 null，不会中断整批读取。

## 变更

### 0.4.0

- wheel 改为 abi3（`cp311-abi3`）：一个文件覆盖 CPython 3.11+，**新增 3.14 支持**。
  此前每个 Python 版本一个 wheel，3.14 发布后装不上；现在新 Python 版本不需要重新发版。
- PyO3 0.24 → 0.29（0.24 本身不支持 3.14）。`PyObject` 别名在 0.29 移除，改用 `Py<PyAny>`，
  仅此一处破坏性改动，无行为变化。
- 发布流程新增 sdist。

无 API 变更，输出与 0.3.0 逐值一致（30 个测试在 3.12 与 3.14 上均通过）。

### 0.3.0

- `columns` 改为按列名映射：以前只要请求顺序不是文件列序的子序列，"回退"的那一列就会静默变成全 null
- 请求的列在文件里不存在时输出整列 null，不再被静默丢弃
- 未声明列如果非空值全部解析失败改为报错，新增 `default_type` 参数
- 新增 `"int64"` 列类型
- 新增 `trim` 参数；`trim=True` 现在对引号字段一视同仁（0.2.0 只剥离非引号字段）
- 大批量读取改为分块流水线（读下一块的同时解析当前块），并省掉合并时的整份拷贝

6080 个行情文件、1763 万行、投影 12 列的单次调用：**13.9s → 3.9s**。
调用方不再需要为了绕开性能问题自己分批。输出与 0.2.0 逐值一致。

### 0.2.0

- 支持引号：`"a,b"` 不再被逗号切碎，跨行引号字段不再被拆成多行，首尾包裹引号会被剥掉
- 新增 `quoting` 参数
- 修复列数不齐时 `concat_batches` 越界 panic（`PanicException` 穿透 `except Exception`）
- 错误信息带上文件路径；`read_csvs_diagonal` 不再把单文件解析失败静默跳过

行情数据（2000 个文件 / 671 万行 / 38 列）实测与 0.1.0 持平，财务数据（5824 个文件 / 372 列）
快约 15%，输出逐值一致。

## License

MIT
