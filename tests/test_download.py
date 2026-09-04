"""parquet 索引条目解析自测（离线）。"""
from voxft.data.download import _resolve_parquet_ref


def test_resolve_parquet_ref():
    f, rev = _resolve_parquet_ref(
        "https://huggingface.co/api/datasets/google/fleurs/parquet/af_za/test/0.parquet")
    assert (f, rev) == ("af_za/test/0.parquet", "refs/convert/parquet")

    f, rev = _resolve_parquet_ref(
        "https://hf-mirror.com/api/datasets/x/y/parquet/tl/train/3.parquet")
    assert (f, rev) == ("tl/train/3.parquet", "refs/convert/parquet")

    f, rev = _resolve_parquet_ref(
        "https://huggingface.co/datasets/x/y/resolve/refs%2Fconvert%2Fparquet/a/0.parquet")
    assert rev == "refs/convert/parquet" and f == "a/0.parquet"

    f, rev = _resolve_parquet_ref("data/train-00000-of-00002.parquet")
    assert (f, rev) == ("data/train-00000-of-00002.parquet", "main")
