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


def test_xet_disabled_only_on_mirror(monkeypatch):
    """镜像不代理 xet 的 CAS 服务器，不关会 401；直连官方时不该动它。"""
    import os

    from voxft import paths

    def _run(endpoint, preset=None):
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        if preset is not None:
            monkeypatch.setenv("HF_HUB_DISABLE_XET", preset)
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        paths._disable_xet_on_mirror()
        return os.environ.get("HF_HUB_DISABLE_XET")

    assert _run("https://hf-mirror.com") == "1"
    assert _run("https://huggingface.co") is None
    assert _run("https://hf-mirror.com", preset="0") == "0"   # 显式设置优先
