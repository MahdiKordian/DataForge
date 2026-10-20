# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import io
import socket
import zipfile

import numpy as np
import pandas as pd
import pytest

from dataforge import benchmark as BM, lab as LB, projects as PR, scale as SC, security as SE
from dataforge import build_pipeline, split_frame
from dataforge.pipeline import Pipeline
from tests.test_core import CFG, toy


# ---- security: uploads -----------------------------------------------------------------------------------------
def test_upload_checks_type_size_and_content():
    assert SE.check_upload("a.csv", b"a,b\n1,2\n") == "csv"
    for name, raw in [("a.exe", b"x"), ("a.csv", b""), ("a.csv", b"MZ\x90\x00binary"), ("a.csv", b"a\x00b"),
                      ("a.xlsx", b"not a zip"), ("a.parquet", b"nope"), ("a.json", b"hello"), ("a.xls", b"nope")]:
        with pytest.raises(SE.UnsafeInput):
            SE.check_upload(name, raw)
    with pytest.raises(SE.UnsafeInput, match="limit"):
        SE.check_upload("a.csv", b"x" * 2048, max_mb=0.001)


def test_zip_bomb_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/big.xml", b"0" * 50_000_000)
    with pytest.raises(SE.UnsafeInput, match="unreasonable"):
        SE.check_upload("x.xlsx", buf.getvalue())


def test_real_parquet_and_xlsx_pass(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_parquet(tmp_path / "p.parquet"); df.to_excel(tmp_path / "x.xlsx", index=False)
    assert SE.check_upload("p.parquet", (tmp_path / "p.parquet").read_bytes()) == "parquet"
    assert SE.check_upload("x.xlsx", (tmp_path / "x.xlsx").read_bytes()) == "xlsx"


def test_filename_is_sanitised():
    assert SE.safe_filename("../../etc/passwd") == "passwd" and SE.safe_filename("a\x00b.csv") == "ab.csv"


# ---- security: URLs --------------------------------------------------------------------------------------------
def fake_dns(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.csv", "gopher://x", "http://user:pw@example.com/a.csv",
                                 "http://example.com:22/a.csv", "http:///nohost"])
def test_bad_urls_refused(url, monkeypatch):
    fake_dns(monkeypatch, "93.184.216.34")
    with pytest.raises(SE.UnsafeInput):
        SE.check_url(url)


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "172.16.0.1", "0.0.0.0", "::1", "fe80::1", "::ffff:127.0.0.1"])
def test_internal_addresses_refused(ip, monkeypatch):
    fake_dns(monkeypatch, ip)
    with pytest.raises(SE.UnsafeInput, match="private|internal"):
        SE.check_url("https://looks-public.example.com/a.csv")


def test_public_address_allowed_and_redirect_to_internal_blocked(monkeypatch):
    fake_dns(monkeypatch, "93.184.216.34")
    assert SE.check_url("https://example.com/a.csv")
    import requests

    class R:
        is_redirect, status_code, headers = True, 302, {"Location": "http://169.254.169.254/latest/meta-data"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
    calls = []
    def fake_get(url, **k):
        calls.append(url)
        return R()
    monkeypatch.setattr(requests, "get", fake_get)
    state = {"n": 0}
    def dns(host, *a, **k):
        ip = "93.184.216.34" if host == "example.com" else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    with pytest.raises(SE.UnsafeInput, match="internal"):
        SE.safe_fetch("https://example.com/a.csv")
    assert len(calls) == 1                                       # the internal address was never requested


# ---- security: personal data and names --------------------------------------------------------------------------
def test_pii_scan_and_redaction():
    df = pd.DataFrame({"mail": ["a@x.com", "b@y.org"] * 20, "tel": ["+1 415 555 0100", "+44 20 7946 0958"] * 20,
                       "card": ["4539 1488 0343 6467"] * 40, "ip": ["8.8.8.8"] * 40, "FirstName": ["Ann"] * 40, "color": ["red", "blue"] * 20})
    got = SE.pii_scan(df).groupby("column")["looks like"].apply(list).to_dict()
    assert "email" in got["mail"] and "phone number" in got["tel"] and "card number" in got["card"] and "IP address" in got["ip"] and "FirstName" in got
    assert "color" not in got
    assert "a@x.com" not in SE.redact("contact a@x.com or 4539 1488 0343 6467")


def test_project_names_cannot_escape(tmp_path):
    for bad in ["../x", "a/b", "..", "a\\b", ".hidden", "", "x" * 80, "a..b"]:
        with pytest.raises(SE.UnsafeInput):
            SE.safe_project_name(bad)
    assert SE.safe_project_name("My project 1") == "My project 1"
    with pytest.raises(SE.UnsafeInput):
        SE.inside(tmp_path, tmp_path / ".." / "other")


def test_saved_pipeline_rejects_tampered_fields():
    pipe = build_pipeline(CFG, "y"); pipe.fit_transform(toy(80))
    st = pipe.state(); st["steps"][2]["learned"]["__class__"] = 1
    with pytest.raises(ValueError, match="Unexpected field"):
        Pipeline.from_state(st)


# ---- scale ------------------------------------------------------------------------------------------------------------
def test_optimize_dtypes_saves_memory_without_changing_values():
    df = BM.make_dataset(20_000, 20)
    df["n"] = np.arange(len(df)) % 100
    out, rep = SC.optimize_dtypes(df)
    assert rep["after_mb"] < rep["before_mb"] and rep["saved %"] > 20
    pd.testing.assert_frame_equal(out.astype({c: df[c].dtype for c in df.columns if str(out[c].dtype) != str(df[c].dtype)}), df, check_dtype=False)
    assert str(out["n"].dtype) == "uint8"


def test_pipeline_gives_the_same_result_on_optimised_data():
    df = BM.make_dataset(3000, 12, seed=2)
    opt, _ = SC.optimize_dtypes(df, float32=False)
    a, _, b, _ = split_frame(df, "y", 20, 0, 1)
    c, _, d, _ = split_frame(opt, "y", 20, 0, 1)
    p1, p2 = build_pipeline(CFG, "y"), build_pipeline(CFG, "y")
    pd.testing.assert_frame_equal(p1.fit_transform(a), p2.fit_transform(c), check_dtype=False)
    pd.testing.assert_frame_equal(p1.transform(b), p2.transform(d), check_dtype=False)


def test_smart_sample_keeps_rare_classes():
    df = pd.DataFrame({"x": range(100_000), "y": ["a"] * 99_000 + ["b"] * 1000})
    s = SC.smart_sample(df, 5000, "y", seed=1)
    assert len(s) <= 5500 and (s.y == "b").sum() >= 50


def test_chunked_profile_equals_full_profile(tmp_path):
    df = BM.make_dataset(30_000, 8, seed=3)
    f = tmp_path / "d.csv"; df.to_csv(f, index=False)
    got = SC.profile_in_chunks(f, chunksize=4_000).set_index("column")
    num = df.select_dtypes("number")
    for c in ["num_0", "num_1"]:
        assert got.loc[c, "mean"] == pytest.approx(num[c].mean(), rel=1e-9)
        assert got.loc[c, "std"] == pytest.approx(num[c].std(), rel=1e-9)
        assert got.loc[c, "min"] == num[c].min() and got.loc[c, "missing"] == num[c].isna().sum()


def test_fingerprint_changes_with_data_and_is_stable():
    df = toy(500)
    assert SC.fingerprint(df) == SC.fingerprint(df.copy())
    d2 = df.copy(); d2.loc[250, "age"] += 1
    assert SC.fingerprint(df) != SC.fingerprint(d2.iloc[:-1])


def test_duckdb_and_polars_when_present(tmp_path):
    df = BM.make_dataset(2000, 6); f = tmp_path / "d.csv"; df.to_csv(f, index=False)
    eng = SC.engines()
    if eng["duckdb"]:
        s = SC.duckdb_summary(str(f))
        assert set(df.columns) <= set(s["column_name"])
    if eng["polars"]:
        assert len(SC.polars_head(str(f), 100)) == 100


# ---- projects ---------------------------------------------------------------------------------------------------------
def test_project_roundtrip_and_integrity(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("DATAFORGE_HOME", str(tmp_path))
    df = toy(200)
    state = {"target": "y", "var_types": {"age": "continuous"}, "contract": {"columns": {}}, "decisions": {"missing:age": "median"},
             "steps": [{"step": "Split", "params": {"test": 20, "val": 0, "seed": 1, "stratify": True, "balance": "None"}}]}
    PR.save_project("demo", df, state, True, "hello")
    assert [p["name"] for p in PR.list_projects()] == ["demo"]
    meta, back = PR.load_project("demo")
    pd.testing.assert_frame_equal(back, df)
    assert meta["settings"]["target"] == "y" and meta["steps"][0]["step"] == "Split" and meta["notes"] == "hello"
    z = zipfile.ZipFile(io.BytesIO(PR.export_zip("demo")))
    assert {"demo/project.json", "demo/data.parquet"} <= set(z.namelist())
    # tamper with the stored data -> refused
    pd.concat([df, df.iloc[:1]]).to_parquet(tmp_path / "projects" / "demo" / "data.parquet")
    with pytest.raises(ValueError, match="does not match"):
        PR.load_project("demo")
    PR.delete_project("demo")
    assert PR.list_projects() == []


def test_project_without_data_and_bad_names(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFORGE_HOME", str(tmp_path))
    PR.save_project("settings only", toy(50), {"target": "y"}, store_data=False)
    meta, df = PR.load_project("settings only")
    assert df is None and not (tmp_path / "projects" / "settings only" / "data.parquet").exists()
    with pytest.raises(SE.UnsafeInput):
        PR.save_project("../evil", toy(5), {})
    with pytest.raises(FileNotFoundError):
        PR.load_project("missing")


# ---- lab ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not LB.sklearn_available(), reason="scikit-learn not installed")
def test_lab_compares_variants_without_a_test_set():
    import inspect
    assert "test" not in " ".join(inspect.signature(LB.run_lab).parameters)      # the test set cannot be passed in
    df = BM.make_dataset(1500, 10, seed=4)
    v = LB.variants_for("Missing numbers", ["Median", "Mean", "Drop rows"])
    res = LB.run_lab(df, "y", v, "Logistic regression", folds=3, metric="macro F1")
    T = res["table"].set_index("variant")
    assert list(T.index) == ["Median", "Mean", "Drop rows"] and T["failed folds"].sum() == 0
    assert T.loc["Drop rows", "rows kept %"] < 100 and T.loc["Median", "rows kept %"] == pytest.approx(100)
    assert T.loc["Median", "verdict"] == "baseline" and res["notes"]


@pytest.mark.skipif(not LB.sklearn_available(), reason="scikit-learn not installed")
def test_lab_reports_a_failing_variant_instead_of_crashing():
    df = BM.make_dataset(800, 8, seed=5)
    v = LB.variants_for("Missing numbers", ["Median"])
    v["No encoding"] = {"Clean": CFG["Clean"]}
    res = LB.run_lab(df, "y", v, "Logistic regression", folds=3)
    assert res["table"].set_index("variant").loc["No encoding", "failed folds"] == 3


# ---- benchmark correctness suite -----------------------------------------------------------------------------------
def test_benchmark_correctness_suite_all_pass():
    r = BM.correctness()
    assert len(r) >= 14 and r["pass"].all(), r[~r["pass"]]
