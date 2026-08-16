"""issue #1 的三个案例：引号处理 + 列数不齐不再 panic。

    uv run --python .venv/bin/python -m pytest tests/ -q
"""
import pathlib
import tempfile

import polars as pl
import pytest

import d2_loader

ANN_SCHEMA = {
    "公告日期": "date:%Y-%m-%d",
    "股票代码": "str",
    "股票名称": "str",
    "公告标题": "str",
}
HEADER = "公告日期,股票代码,股票名称,公告标题"


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


def write_gbk(path: pathlib.Path, body: str) -> str:
    path.write_bytes(f"免责声明行\n{HEADER}\n{body}".encode("gbk"))
    return str(path)


def test_case1_unescaped_inner_quotes_lose_nothing(tmpdir):
    """数据源把字段用引号包住，但内部引号没有翻倍转义。

    首尾的包裹引号要剥掉，内部引号原样保留——一个字符都不丢。
    polars 1.43 在这种输入上会把内部引号直接吞掉，所以这里不跟它对齐。
    """
    row = '2024-05-08,bj920000,安徽凤凰,"安徽凤凰:2023年年度报告业绩说明会预告暨落实"提质守信重回报"行动公告"\n'
    f = write_gbk(tmpdir / "a.csv", row)

    got = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA)

    assert got["公告标题"][0] == "安徽凤凰:2023年年度报告业绩说明会预告暨落实\"提质守信重回报\"行动公告"

    # polars 的结果是我们的子集：把引号删掉就一样
    expected = pl.read_csv(f, encoding="gbk", skip_rows=1, truncate_ragged_lines=True)
    assert got["公告标题"][0].replace('"', "") == expected["公告标题"][0]


def test_case1b_quoted_comma_does_not_shift_columns(tmpdir):
    """引号内的逗号不能把行切碎（0.1.x 会整体错位）"""
    f = write_gbk(tmpdir / "b.csv", '2024-05-08,bj920000,安徽凤凰,"关于变更公司名称,注册地址的公告"\n')

    got = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA)

    assert got["公告标题"][0] == "关于变更公司名称,注册地址的公告"
    assert got["股票名称"][0] == "安徽凤凰"


def test_case2_multiline_quoted_field_row_count_matches_polars(tmpdir):
    """跨行引号字段算一条记录，行数要和 polars 一致"""
    body = '2024-05-08,bj920000,安徽凤凰,"第一行\n第二行的续行"\n2024-05-09,bj920001,测试,普通标题\n'
    f = write_gbk(tmpdir / "c.csv", body)

    got = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA)
    expected = pl.read_csv(f, encoding="gbk", skip_rows=1)

    assert len(got) == len(expected) == 2
    assert got["公告标题"][0] == "第一行\n第二行的续行"
    assert got["股票代码"][1] == "bj920001"


def test_case3_ragged_columns_raise_normal_exception_not_panic(tmpdir):
    """列数不齐：0.1.x 在 concat_batches 里越界 panic，PanicException 穿透 except Exception"""
    wide = tmpdir / "wide.csv"
    wide.write_bytes("注释\na,b,c\n1,2,3\n".encode("gbk"))
    narrow = tmpdir / "narrow.csv"
    narrow.write_bytes("注释\na,b\n4,5\n".encode("gbk"))

    df = d2_loader.read_csvs([str(wide), str(narrow)], skip_rows=1)

    assert df.shape == (2, 3)
    assert df["c"].to_list() == [3.0, None]


def test_missing_file_raises_catchable_exception(tmpdir):
    """错误信息要带上文件路径，且必须是普通 Exception"""
    with pytest.raises(Exception) as exc:
        d2_loader.read_csvs([str(tmpdir / "不存在.csv")])
    assert "不存在.csv" in str(exc.value)


def test_quoting_false_keeps_quotes_literal(tmpdir):
    """显式关掉引号处理 = 0.1.x 的行为"""
    f = write_gbk(tmpdir / "d.csv", '2024-05-08,bj920000,安徽凤凰,"带引号的标题"\n')

    on = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA)
    off = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA, quoting=False)

    assert on["公告标题"][0] == "带引号的标题"
    assert off["公告标题"][0] == '"带引号的标题"'


def test_diagonal_reader_handles_quotes_and_missing_columns(tmpdir):
    a = tmpdir / "a.csv"
    a.write_bytes('注释\ncode,name\n"600000","浦发,银行"\n'.encode("gbk"))
    b = tmpdir / "b.csv"
    b.write_bytes("注释\ncode,extra\n600001,7\n".encode("gbk"))

    df = d2_loader.read_csvs_diagonal(
        [str(a), str(b)], skip_rows=1, schema={"code": "str", "name": "str"}
    )

    assert df.shape == (2, 3)
    assert df.filter(pl.col("code") == "600000")["name"][0] == "浦发,银行"
    assert df.filter(pl.col("code") == "600001")["extra"][0] == 7.0


def test_plain_numeric_csv_unchanged(tmpdir):
    """没有引号的普通数据：结果与 polars 完全一致（回归保护）"""
    body = "".join(f"2024-01-{d:02d},sz00000{d},股票{d},{d}.5\n" for d in range(1, 10))
    f = write_gbk(tmpdir / "e.csv", body)

    got = d2_loader.read_csvs([f], skip_rows=1, schema=ANN_SCHEMA)
    expected = pl.read_csv(f, encoding="gbk", skip_rows=1, try_parse_dates=True)

    assert got["公告日期"].to_list() == expected["公告日期"].to_list()
    assert got["股票名称"].to_list() == expected["股票名称"].to_list()
