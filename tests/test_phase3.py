# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
import numpy as np
import pandas as pd

from dataforge import analysis as A
from dataforge import build_pipeline, dumps, loads, make_spec, apply_saved, split_frame, target as T
from dataforge.reporting import key_points
from dataforge.types import infer_types
from tests.test_core import CFG


def types_df(n=120):
    r = np.random.default_rng(0)
    return pd.DataFrame({
        "CustomerID": [f"C{i:04d}" for i in range(n)],
        "PostalCode": r.choice(["02115", "02116", "10001", "90210", "00501"], n),
        "Charges": [f"{x:.2f}" for x in r.random(n) * 100],
        "Size": r.choice(["small", "medium", "large"], n),
        "Joined": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Flag": r.integers(0, 2, n), "Age": r.integers(18, 80, n).astype(float), "y": r.choice(["a", "b"], n)})


def test_type_inference_catches_the_classic_traps():
    t = infer_types(types_df(), "y")
    assert t["CustomerID"]["type"] == "id" and t["CustomerID"]["confidence"] == "high"
    assert t["PostalCode"]["type"] == "categorical"
    assert t["Charges"]["type"] == "continuous"
    assert t["Size"]["type"] == "ordinal" and t["Size"]["order"] == ["small", "medium", "large"]
    assert t["Joined"]["type"] == "datetime"
    assert t["Flag"]["type"] == "categorical" and t["Age"]["type"] == "continuous"
    assert t["y"]["type"] == "target"
    assert all(v["reason"] for v in t.values())


def test_confirmed_types_drive_the_pipeline():
    df = types_df()
    types = {c: v["type"] for c, v in infer_types(df, "y").items() if c != "y"}
    cfg = {**CFG, "Types": {"types": types, "orders": {"Size": ["small", "medium", "large"]}}}
    tr, _, te, _ = split_frame(df, "y", 25, 0, 1)
    pipe = build_pipeline(cfg, "y")
    out = pipe.fit_transform(tr)
    assert "CustomerID" not in out.columns
    assert any(c.startswith("PostalCode_") for c in out.columns)
    assert any("0" == c.split("PostalCode_")[1][0] for c in out.columns if c.startswith("PostalCode_"))
    assert set(out["Size"].dropna().unique()) <= {0.0, 1.0, 2.0} or out["Size"].nunique() > 1
    assert any(c.startswith("Joined_") for c in out.columns)
    te_out = pipe.transform(te)
    assert list(te_out.columns) == list(out.columns)


def test_ordinal_follows_the_users_order_and_warns_on_unseen():
    from dataforge.transformers import TypeCaster
    tc = TypeCaster(types={"s": "ordinal"}, orders={"s": ["low", "mid", "high"]}).fit(pd.DataFrame({"s": ["low", "mid", "high"]}))
    out = tc.transform(pd.DataFrame({"s": ["High", "low", "weird", None]}))
    assert out["s"].iloc[:2].tolist() == [2.0, 0.0] and pd.isna(out["s"].iloc[2]) and tc.warnings_


def test_types_survive_save_and_load():
    df = types_df(); types = {c: v["type"] for c, v in infer_types(df, "y").items() if c != "y"}
    cfg = {**CFG, "Types": {"types": types, "orders": {}}}
    tr, _, te, _ = split_frame(df, "y", 25, 0, 1)
    pipe = build_pipeline(cfg, "y"); pipe.fit_transform(tr)
    spec = loads(dumps(make_spec(pipe, cfg, {"test_pct": 25, "val_pct": 0, "seed": 1, "balance": "None"}, tr)))
    pd.testing.assert_frame_equal(apply_saved(spec, te)[0], pipe.transform(te))


def test_association_measures_behave():
    r = np.random.default_rng(1)
    a = pd.Series(r.choice(list("xyz"), 600)); b = a.map({"x": "p", "y": "q", "z": "r"})
    assert A.cramers_v(a, b) > 0.95 and A.cramers_v(a, pd.Series(r.choice(list("pq"), 600))) < 0.1
    v = pd.Series(a.map({"x": 0.0, "y": 5.0, "z": 10.0}) + r.normal(0, .1, 600))
    assert A.eta_squared(a, v) > 0.95
    x = pd.Series(r.uniform(-3, 3, 800)); y = x ** 2 + r.normal(0, .1, 800)
    assert abs(x.corr(y)) < 0.15 and 0.2 < A.nmi(x, y) <= 1.0
    assert A.nmi(pd.Series([1.0] * 100), pd.Series(r.random(100))) == 0.0


def test_nonlinear_and_vif():
    r = np.random.default_rng(2)
    x = r.uniform(-3, 3, 800)
    df = pd.DataFrame({"x": x, "sq": x ** 2 + r.normal(0, .1, 800), "n": r.normal(size=800)})
    nl = A.nonlinear_candidates(df, ["x", "sq", "n"])
    assert {"x", "sq"} == set(nl.iloc[0][["column A", "column B"]])
    c = pd.DataFrame({"a": r.normal(size=300)}); c["b"] = c.a * 2 + r.normal(0, .01, 300); c["c"] = r.normal(size=300)
    v = A.vif_table(c, ["a", "b", "c"]).set_index("column")["VIF"]
    assert v["a"] > 100 and v["c"] < 2


def test_missing_analysis_finds_dependent_missingness():
    r = np.random.default_rng(3); n = 800
    age = r.normal(40, 10, n); inc = r.normal(50, 10, n)
    inc[age > 45] = np.where(r.random((age > 45).sum()) < 0.7, np.nan, inc[age > 45])
    m = A.missing_analysis(pd.DataFrame({"age": age, "inc": inc, "z": r.normal(size=n)}))
    assert m["n_dependent"] >= 1 and m["relations"].iloc[0]["related column"] == "age"
    assert "random" in m["limits"].lower()


def test_quality_checks_find_the_usual_suspects():
    df = pd.DataFrame({"age": [25, -3, 40, 31] * 10, "city": ["NY", "ny ", "N/A", "LA"] * 10,
                       "mix": ["1", "2", "abc", "4"] * 10, "const": 1, "e": np.nan})
    kinds = {(q["column"], q["check"]) for q in A.quality_checks(df)}
    assert {("age", "implausible negatives"), ("city", "disguised missing values"), ("city", "extra spaces"),
            ("mix", "mixed types"), ("const", "constant"), ("e", "empty column")} <= kinds


def test_time_profile_sees_gaps():
    t = list(pd.date_range("2024-01-01", periods=40)) + list(pd.date_range("2024-06-01", periods=10))
    P = A.time_profile(pd.DataFrame({"t": t, "v": range(50)}), "t", ["v"])
    assert P["irregular gaps"] == 1 and P["empty periods"] > 50 and len(P["trends"]) == 1


def test_classification_flags_imbalance_leakage_and_split_drift():
    r = np.random.default_rng(4); n = 600
    y = pd.Series(np.where(r.random(n) < 0.08, "pos", "neg"))
    df = pd.DataFrame({"y": y, "leak": (y == "pos").astype(float) + r.normal(0, .01, n), "x": r.normal(size=n)})
    res = T.analyze_target(df, "y", "Classification")
    msgs = " ".join(f["message"] for f in res["flags"])
    assert "Imbalanced" in msgs and "leakage" in msgs and res["features"].iloc[0]["feature"] == "leak"
    drift = T.analyze_target(df, "y", "Classification", splits={"train": df.iloc[:300], "test": df[df.y == "pos"]})
    assert "Class shares differ" in " ".join(f["message"] for f in drift["flags"])


def test_small_samples_are_marked_unreliable():
    r = np.random.default_rng(5); n = 400
    x = pd.Series(np.nan, index=range(n)); x.iloc[:8] = [0, 1] * 4
    df = pd.DataFrame({"y": r.choice(["a", "b"], n), "sparse": x, "ok": r.normal(size=n)})
    f = T.analyze_target(df, "y", "Classification")["features"].set_index("feature")
    assert not f.loc["sparse", "reliable"] and f.loc["ok", "reliable"]


def test_regression_and_timeseries_and_unsupervised():
    r = np.random.default_rng(6); n = 500
    x = r.normal(size=n); y = np.exp(x + r.normal(0, .3, n))
    reg = T.analyze_target(pd.DataFrame({"x": x, "y": y}), "y", "Regression")
    assert any("skewed" in f["message"] for f in reg["flags"]) and reg["features"].iloc[0]["feature"] == "x"
    d = pd.DataFrame({"t": pd.date_range("2022-01-01", periods=300), "v": np.arange(300) * 0.1 + np.sin(np.arange(300) * 2 * np.pi / 7)})
    ts = T.analyze_target(d, "v", "Time-series forecasting", time_col="t")
    m = " ".join(f["message"] for f in ts["flags"])
    assert "trend" in m and "Split by time" in m and 7 in ts["seasonal_lags"]
    un = T.analyze_target(pd.DataFrame({"a": r.normal(size=200), "b": r.normal(size=200) * 1000}), None, "Unsupervised")
    assert any("scales" in f["message"] for f in un["flags"])


def test_key_points_are_quantified():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4] * 10, "y": ["u", "v"] * 20})
    pts = key_points(df, "y", None, None, None, False)
    assert any("missing" in p for p in pts) and any("not been split" in p for p in pts)


def test_blank_text_counts_as_missing_everywhere():
    df = pd.DataFrame({"TotalCharges": ["1.5", " ", "2", "3", ""] * 20, "y": ["a", "b"] * 50})
    assert int(A.missing_analysis(df)["per_column"].iloc[0]["missing"]) == 40
    assert any("TotalCharges" in p and "40%" in p for p in key_points(df, "y", None, None, None, False))
