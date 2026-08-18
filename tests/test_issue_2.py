"""issue #2：列映射/类型的静默数据丢失，以及 int64 / trim / 内部分块。

这些测试是先写的——实现之前它们应该全部失败。

关于 P0-2 的契约选择（未在 schema 中声明的列）：
    issue 建议"未声明列默认按字符串解析"。这里没有直接采纳，因为 d2 的
    read_stock_csvs 只声明字符串列和日期列，价格/成交量全靠 float64 默认，
    默认值一改那边就全变字符串。改成两条：
      1. 新增 default_type 参数，想要 polars 语义的调用方显式传 "str"
      2. 未声明的列若"非空值 100% 解析失败"则报错，指出列名/文件/示例值
    显式声明成 float64 的列不受影响，仍然填 null（尊重调用方的选择）。
"""
import pathlib
import tempfile

import polars as pl
import pytest

import d2_loader


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


def write_gbk(path: pathlib.Path, text: str) -> str:
    path.write_bytes(text.encode("gbk"))
    return str(path)


# ─── P0-1：columns 顺序必须与文件列序无关 ──────────────────────────

STR3 = {"A": "str", "B": "str", "C": "str"}


def test_columns_order_independent_of_file_order(tmpdir):
    """文件列序 C,A,B，请求 [A,B,C]：0.2.0 会让 C 整列静默变 null"""
    f = write_gbk(tmpdir / "a.csv", "免责行\nC,A,B\nC0,A0,B0\nC1,A1,B1\n")

    df = d2_loader.read_csvs([f], columns=["A", "B", "C"], skip_rows=1, schema=STR3)

    assert df["C"].to_list() == ["C0", "C1"]
    assert df["A"].to_list() == ["A0", "A1"]
    assert df["B"].to_list() == ["B0", "B1"]


def test_output_column_order_follows_request(tmpdir):
    f = write_gbk(tmpdir / "a.csv", "免责行\nC,A,B\nC0,A0,B0\n")

    assert d2_loader.read_csvs([f], columns=["A", "B", "C"], skip_rows=1, schema=STR3).columns == ["A", "B", "C"]
    assert d2_loader.read_csvs([f], columns=["B", "C", "A"], skip_rows=1, schema=STR3).columns == ["B", "C", "A"]


def test_fully_reversed_column_request(tmpdir):
    f = write_gbk(tmpdir / "a.csv", "免责行\nA,B,C\na0,b0,c0\n")

    df = d2_loader.read_csvs([f], columns=["C", "B", "A"], skip_rows=1, schema=STR3)

    assert df.row(0) == ("c0", "b0", "a0")


def test_real_world_base_plus_extra_column_layout(tmpdir):
    """issue 里描述的真实形态：基础列 + 扩展列，而扩展列在文件中段"""
    header = "扩展1,基础1,基础2,扩展2"
    f = write_gbk(tmpdir / "a.csv", f"免责行\n{header}\ne1,b1,b2,e2\n")
    schema = {c: "str" for c in ("基础1", "基础2", "扩展1", "扩展2")}

    df = d2_loader.read_csvs(
        [f], columns=["基础1", "基础2", "扩展1", "扩展2"], skip_rows=1, schema=schema
    )

    assert df.row(0) == ("b1", "b2", "e1", "e2")


def test_requested_column_missing_from_file_is_null_column(tmpdir):
    """请求的列文件里没有：0.2.0 会把这一列从结果里静默删掉"""
    f = write_gbk(tmpdir / "a.csv", "免责行\nA,B\na0,b0\n")

    df = d2_loader.read_csvs(
        [f], columns=["A", "不存在的列", "B"], skip_rows=1,
        schema={"A": "str", "B": "str", "不存在的列": "str"},
    )

    assert df.columns == ["A", "不存在的列", "B"]
    assert df["不存在的列"].to_list() == [None]


def test_column_missing_from_only_some_files(tmpdir):
    a = write_gbk(tmpdir / "a.csv", "免责行\nA,B\na0,b0\n")
    b = write_gbk(tmpdir / "b.csv", "免责行\nB,A\nb1,a1\n")
    c = write_gbk(tmpdir / "c.csv", "免责行\nA\na2\n")

    df = d2_loader.read_csvs([a, b, c], columns=["A", "B"], skip_rows=1, schema={"A": "str", "B": "str"})

    assert df["A"].to_list() == ["a0", "a1", "a2"]
    assert df["B"].to_list() == ["b0", "b1", None]


# ─── P0-2：未声明列的类型 ──────────────────────────────────────────

UNDECL = "免责行\n代码,标题,数值\nsh600000,某公告,1.5\nsh600001,另一公告,2.5\n"


def test_undeclared_text_column_raises_instead_of_silent_null(tmpdir):
    """0.2.0 在这里返回 dtype=f64 的全 null 列，不报错也不警告"""
    f = write_gbk(tmpdir / "a.csv", UNDECL)

    with pytest.raises(RuntimeError) as exc:
        d2_loader.read_csvs([f], columns=["代码", "标题"], skip_rows=1, schema={"代码": "str"})

    msg = str(exc.value)
    assert "标题" in msg           # 指出是哪一列
    assert "a.csv" in msg          # 指出是哪个文件
    assert "某公告" in msg          # 给出示例值
    assert "str" in msg            # 提示怎么修


def test_default_type_str_reads_undeclared_columns_as_text(tmpdir):
    f = write_gbk(tmpdir / "a.csv", UNDECL)

    df = d2_loader.read_csvs(
        [f], columns=["代码", "标题"], skip_rows=1, schema={"代码": "str"}, default_type="str"
    )

    assert df["标题"].to_list() == ["某公告", "另一公告"]
    assert df["标题"].dtype == pl.String


def test_default_type_float64_is_the_default(tmpdir):
    """默认值不变：未声明的数值列仍然是 float64（d2 依赖这个）"""
    f = write_gbk(tmpdir / "a.csv", "免责行\n代码,数值\nsh600000,1.5\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"代码": "str"})

    assert df["数值"].dtype == pl.Float64
    assert df["数值"].to_list() == [1.5]


def test_explicitly_declared_float64_keeps_nulls_without_raising(tmpdir):
    """显式声明 float64 = 调用方自己的选择，解析失败照旧填 null"""
    f = write_gbk(tmpdir / "a.csv", "免责行\nv\n--\nN/A\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"v": "float64"})

    assert df["v"].to_list() == [None, None]


def test_undeclared_column_with_some_valid_values_stays_float(tmpdir):
    """只有部分值解析失败 → 不报错，失败的填 null（保持原行为）"""
    f = write_gbk(tmpdir / "a.csv", "免责行\nv\n1.5\n--\n2.5\n")

    df = d2_loader.read_csvs([f], skip_rows=1)

    assert df["v"].to_list() == [1.5, None, 2.5]


def test_undeclared_all_empty_column_does_not_raise(tmpdir):
    """整列都是空值 → 没有"解析失败"，不该报错"""
    f = write_gbk(tmpdir / "a.csv", "免责行\na,v\nx,\ny,\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"a": "str"})

    assert df["v"].to_list() == [None, None]


# ─── P1：int64 ────────────────────────────────────────────────────

def test_int64_preserves_large_integers(tmpdir):
    f = write_gbk(tmpdir / "a.csv", "免责行\nv\n9007199254740993\n123456789012345678\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"v": "int64"})

    assert df["v"].dtype == pl.Int64
    assert df["v"].to_list() == [9007199254740993, 123456789012345678]


def test_int64_handles_negative_empty_and_invalid(tmpdir):
    # 单列文件里的空行与"空值"无法区分，扫描器按空行跳过，所以用两列表达空值
    f = write_gbk(tmpdir / "a.csv", "免责行\na,v\nx,-42\nx,\nx,不是数字\nx,1.5\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"a": "str", "v": "int64"})

    assert df["v"].to_list() == [-42, None, None, None]


def test_default_type_int64(tmpdir):
    f = write_gbk(tmpdir / "a.csv", "免责行\na,v\nx,7\n")

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"a": "str"}, default_type="int64")

    assert df["v"].dtype == pl.Int64
    assert df["v"].to_list() == [7]


def test_diagonal_reader_supports_int64(tmpdir):
    a = write_gbk(tmpdir / "a.csv", "免责行\ncode,shares\n600000,9007199254740993\n")
    b = write_gbk(tmpdir / "b.csv", "免责行\ncode,other\n600001,1\n")

    df = d2_loader.read_csvs_diagonal(
        [a, b], skip_rows=1, schema={"code": "str", "shares": "int64"}
    )

    assert df.filter(pl.col("code") == "600000")["shares"][0] == 9007199254740993


# ─── P2-1：trim ───────────────────────────────────────────────────

TRIM_CSV = '免责行\na,b\n  空格  ,"  引号内空格  "\n'


def test_trim_default_true_trims_both_quoted_and_unquoted(tmpdir):
    """默认保持 0.2.0 的 trim 行为，但要对引号字段一视同仁（0.2.0 只 trim 非引号字段）"""
    f = write_gbk(tmpdir / "a.csv", TRIM_CSV)

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"a": "str", "b": "str"})

    assert df.row(0) == ("空格", "引号内空格")


def test_trim_false_preserves_literal_whitespace(tmpdir):
    """trim=False 与 polars read_csv 的字面值语义一致"""
    f = write_gbk(tmpdir / "a.csv", TRIM_CSV)

    df = d2_loader.read_csvs([f], skip_rows=1, schema={"a": "str", "b": "str"}, trim=False)
    expected = pl.read_csv(f, encoding="gbk", skip_rows=1)

    assert df.row(0) == ("  空格  ", "  引号内空格  ")
    assert df["a"][0] == expected["a"][0]
    assert df["b"][0] == expected["b"][0]


def test_trim_false_still_parses_numbers_with_surrounding_spaces(tmpdir):
    """数值列不该因为 trim=False 就解析不出来"""
    f = write_gbk(tmpdir / "a.csv", "免责行\nv\n  1.5  \n")

    df = d2_loader.read_csvs([f], skip_rows=1, trim=False)

    assert df["v"].to_list() == [1.5]


def test_trim_applies_to_headers_regardless(tmpdir):
    """表头无论如何都要 trim，否则列名匹配不上"""
    f = write_gbk(tmpdir / "a.csv", "免责行\n a , b \nx,y\n")

    df = d2_loader.read_csvs([f], columns=["a", "b"], skip_rows=1, schema={"a": "str", "b": "str"}, trim=False)

    assert df.columns == ["a", "b"]
    assert df.row(0) == ("x", "y")


# ─── P2-2：内部分块不改变结果 ──────────────────────────────────────

def test_many_files_single_call_matches_batched_calls(tmpdir):
    """内部分块读取后，一次性调用的结果必须与分批调用完全一致"""
    paths = []
    for i in range(300):
        p = write_gbk(tmpdir / f"f{i:04d}.csv", f"免责行\ncode,v\ns{i:04d},{i}.5\n")
        paths.append(p)

    one = d2_loader.read_csvs(paths, skip_rows=1, schema={"code": "str"})
    batched = pl.concat(
        [d2_loader.read_csvs(paths[i:i + 50], skip_rows=1, schema={"code": "str"}) for i in range(0, 300, 50)]
    )

    assert one.shape == (300, 2)
    assert one.equals(batched)


def test_many_files_preserve_input_order(tmpdir):
    paths = [write_gbk(tmpdir / f"f{i:03d}.csv", f"免责行\ncode\ns{i:03d}\n") for i in range(250)]

    df = d2_loader.read_csvs(paths, skip_rows=1, schema={"code": "str"})

    assert df["code"].to_list() == [f"s{i:03d}" for i in range(250)]
