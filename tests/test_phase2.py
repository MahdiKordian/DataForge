# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from dataforge import (SpecError, apply_saved, build_pipeline, compare, diagnose, dumps, infer_contract, loads,
                       make_spec, merge, refit, split_frame, to_script, validate_contract, what_if_missing)
from dataforge.pipeline import Pipeline
from tests.test_core import CFG, toy


def contract_df():
    return pd.DataFrame({"CustomerID": [f"c{i}" for i in range(30)], "Age": np.arange(20, 50),
                         "Income": np.linspace(1000, 5000, 30), "Type": ["a", "b", "c"] * 10})


def test_contract_passes_on_its_own_data():
    df = contract_df()
    assert validate_contract(df, infer_contract(df))["status"] == "passed"


def test_contract_catches_rename_text_range_category_duplicates():
    df = contract_df(); c = infer_contract(df)
    new = df.rename(columns={"Income": "MonthlyIncome"})
    new["Age"] = new["Age"].astype(object); new.loc[0, "Age"] = "abc"; new.loc[1, "Age"] = 500
    new.loc[2, "Type"] = "zzz"; new.loc[3, "CustomerID"] = "c4"
    rep = validate_contract(new, c)
    assert rep["status"] == "failed"
    rules = {(i["column"], i["rule"]) for i in rep["issues"]}
    assert {("Income", "exists"), ("Age", "type"), ("Age", "max"), ("Type", "allowed"), ("CustomerID", "unique")} <= rules
    assert "MonthlyIncome" in next(i for i in rep["issues"] if i["rule"] == "exists")["message"]
    assert rep["bad_rows"]


def test_contract_numeric_text_with_blanks_is_numeric():
    df = pd.DataFrame({"x": ["1.5", " ", "3", "4"]})
    c = infer_contract(df)
    assert c["columns"]["x"]["kind"] == "numeric" and validate_contract(df, c)["status"] == "passed"

def test_diff_detects_drift_and_schema_changes():
    r = np.random.default_rng(0)
    a = pd.DataFrame({"age": r.normal(35, 5, 500), "city": r.choice(["x", "y"], 500), "k": 1.0})
    b = pd.DataFrame({"age": r.normal(55, 5, 500), "city": r.choice(["x", "y", "new"], 500), "extra": 1})
    d = compare(a, b, "train", "prod")
    assert d["removed"] == ["k"] and d["added"] == ["extra"]
    assert d["numeric"].set_index("column").loc["age", "drift"] == "large"
    assert d["categorical"].set_index("column").loc["city", "n new"] == 1
    assert any("age" in f["message"] for f in d["flags"])


def test_diff_identical_is_stable():
    a = toy(300); d = compare(a, a.copy())
    assert (d["numeric"]["drift"] == "stable").all() and d["summary"]["change"].eq(0).all()


def test_advisor_explains_and_gives_options_without_a_winner():
    r = np.random.default_rng(1)
    df = pd.DataFrame({"cid": [f"id{i}" for i in range(100)], "inc": np.where(r.random(100) < .2, np.nan, r.lognormal(10, 1, 100)),
                       "c": r.choice(["a", "b"], 100), "y": r.choice(["n"] * 9 + ["p"], 100)})
    f = {x["id"]: x for x in diagnose(df, "y")}
    assert {"missing:inc", "id:cid", "imbalance"} <= set(f)
    m = f["missing:inc"]
    assert m["reason"] and m["evidence"]["missing %"] > 0 and len(m["options"]) >= 3
    assert all(o["pros"] and o["cons"] for o in m["options"])
    assert not any("best" in (o["label"] + o["pros"]).lower() for o in m["options"])


def test_advisor_decision_changes_the_result():
    df = toy(80); df.loc[:9, "income"] = np.nan
    opt = next(o for o in next(x for x in diagnose(df, "y") if x["id"] == "missing:income")["options"] if o["key"] == "zero")
    cfg = {"Clean": merge(CFG, opt["patch"])["Clean"]}
    out = build_pipeline(cfg, "y").fit_transform(df)
    assert (out["income"].iloc[:10] == 0).all()


def test_what_if_returns_one_row_per_strategy():
    df = toy(80); df.loc[:9, "income"] = np.nan
    t = what_if_missing(df, "income", None)
    assert list(t["strategy"]) == ["Median", "Mean", "Fill with 0", "Drop rows"]
    assert t.set_index("strategy").loc["Drop rows", "rows lost"] == 10


@pytest.mark.parametrize("file,target", [("Telco_Customer_Churn.csv", "Churn"), ("Palmer_Penguins.csv", "species"),
                                         ("Ames_Housing.csv", "SalePrice")])
def test_saved_pipeline_gives_identical_output_after_roundtrip(file, target):
    path = Path(__file__).resolve().parents[1] / "sample_data" / file
    if not path.exists():
        pytest.skip(f"sample_data/{file} is not in this checkout")
    df = pd.read_csv(path)
    tr, va, te, _ = split_frame(df, target, 20, 0, 42)
    pipe = build_pipeline(CFG, target); pipe.fit_transform(tr)
    info = {"test_pct": 20, "val_pct": 0, "seed": 42, "balance": "None", "stratify": True}
    text = dumps(make_spec(pipe, CFG, info, tr, "t"))
    spec = loads(text)
    out, _ = apply_saved(spec, te)
    pd.testing.assert_frame_equal(out, pipe.transform(te))


def test_saved_pipeline_works_on_data_without_target_and_fails_readably():
    df = toy(100); tr, _, te, _ = split_frame(df, "y", 20, 0, 1)
    pipe = build_pipeline(CFG, "y"); pipe.fit_transform(tr)
    spec = loads(dumps(make_spec(pipe, CFG, {"test_pct": 20, "val_pct": 0, "seed": 1, "balance": "None"}, tr)))
    out, _ = apply_saved(spec, te.drop(columns="y"))
    assert "y" not in out.columns and len(out) == len(te)
    with pytest.raises(Exception, match="income"):
        apply_saved(spec, te.drop(columns="income"))


def test_refit_reproduces_training_output():
    df = toy(120)
    tr, _, _, _ = split_frame(df, "y", 20, 10, 5)
    pipe = build_pipeline(CFG, "y"); a = pipe.fit_transform(tr)
    spec = loads(dumps(make_spec(pipe, CFG, {"test_pct": 20, "val_pct": 10, "seed": 5, "balance": "None", "stratify": True}, tr)))
    pd.testing.assert_frame_equal(refit(spec, df)["train"], a)


def test_bad_spec_files_are_rejected():
    for bad in ("not json", "{}", json.dumps({"format": "dataforge-pipeline", "version": 99})):
        with pytest.raises(SpecError):
            loads(bad)



def test_unknown_step_class_is_refused():
    df = toy(60); pipe = build_pipeline(CFG, "y"); pipe.fit_transform(df)
    st = pipe.state(); st["steps"][1]["class"] = "os.system"
    with pytest.raises(ValueError, match="Unknown step"):
        Pipeline.from_state(st)


def test_generated_script_runs(tmp_path, monkeypatch):
    df = toy(100); df.to_csv(tmp_path / "d.csv", index=False)
    tr, _, _, _ = split_frame(df, "y", 20, 0, 1)
    pipe = build_pipeline(CFG, "y"); pipe.fit_transform(tr)
    code = to_script(make_spec(pipe, CFG, {"test_pct": 20, "val_pct": 0, "seed": 1, "balance": "None", "stratify": True}, tr), "d.csv")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "gen.py", "exec"), {})
    assert (tmp_path / "train_prepared.csv").exists() and (tmp_path / "test_prepared.csv").exists()


def test_advisor_sees_blank_text_and_numbers_stored_as_text():
    df = pd.DataFrame({"TotalCharges": ["10.5", " ", "20", "30", "40.2"] * 10, "y": [0, 1] * 25})
    f = {x["id"]: x for x in diagnose(df, "y")}
    assert "numtext:TotalCharges" in f and "missing:TotalCharges" in f
    assert f["missing:TotalCharges"]["evidence"]["missing cells"] == 10
