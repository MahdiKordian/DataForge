import html
import re
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PAGES = {"dataset": "pages/dataset.py"}
SECTIONS = ["Summary", "Findings", "Column profile", "Statistics", "Relationships",
            "Pipeline log", "Before vs after", "Recommendations"]
PIPELINE = ["Load", "Clean", "Transform", "Encode", "Scale", "Split"]
SEV = {3: "High", 2: "Medium", 1: "Low"}
ADVICE = {
    "Clean": "Clean: fill or drop missing values, remove duplicate rows and drop constant columns.",
    "Transform": "Transform: treat outliers, reduce skew and balance the target classes if needed.",
    "Encode": "Encode: turn categorical columns into numbers before training.",
    "Scale": "Scale: put numeric columns on a similar range so no column dominates.",
    "Split": "Split: create the train and test sets last, so no information leaks between them.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&display=swap');
.block-container{max-width:1080px;padding-top:2.5rem}
.dfd{font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:30px 0 10px;padding:0}
.dfd p{margin:8px 0 0;opacity:.7;max-width:60ch;font-size:17px}
</style>
"""

LIGHT = "--bg:#ffffff;--fg:#1c2330;--mut:#667085;--line:#dfe3ea;--row:#edf0f5;--card:#f4f6fa;--code:#eef1f6;--acc:#3b6cff;"
DARK = "--bg:#0e1117;--fg:#e6e9ef;--mut:#9aa4b5;--line:#2c3446;--row:#1f2635;--card:#171c27;--code:#232b3b;--acc:#7b97ff;"
REPORT_CSS = (
    ":root{" + LIGHT + "color-scheme:light}"
    ":root[data-theme=dark]{" + DARK + "color-scheme:dark}"
    "@media (prefers-color-scheme:dark){:root:not([data-theme]){" + DARK + "color-scheme:dark}}"
    """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--fg);background:var(--bg);margin:0;line-height:1.55}
main{max-width:920px;margin:0 auto;padding:40px 28px 60px}
h1{font-size:34px;letter-spacing:-.02em;margin:0 0 4px}
h2{font-size:21px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
.meta,.note{color:var(--mut)}.meta{font-size:14px}.note{font-size:13px}
p,li{font-size:15px}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:13px}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0}
.kv div{background:var(--card);border-radius:8px;padding:10px 12px}
.kv span{display:block;font-size:12px;color:var(--mut)}.kv b{font-size:19px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0;break-inside:avoid}
th{text-align:left;color:var(--mut);font-weight:600;border-bottom:2px solid var(--line);padding:6px 8px}
td{border-bottom:1px solid var(--row);padding:6px 8px;vertical-align:middle}
.bar{display:inline-block;height:7px;background:var(--acc);border-radius:4px;margin-left:8px;vertical-align:middle}
"""
    "@media print{:root:root,:root[data-theme]{" + LIGHT + "color-scheme:light}main{padding:0}h2{break-after:avoid}}"
)


def nums(df):
    return list(df.select_dtypes("number").columns)


def cats(df):
    n = set(nums(df))
    return [c for c in df.columns if c not in n and not pd.api.types.is_datetime64_any_dtype(df[c])]


def is_class(s):
    return not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20


def outlier_info(df):
    total, rate = 0, {}
    for c in nums(df):
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        i = q3 - q1
        if i > 0:
            k = int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
            total += k
            rate[c] = k / len(s) * 100
    return total, rate


def readiness(df, split_done):
    out, _ = outlier_info(df), None
    n_cells = max(df.select_dtypes("number").size, 1)
    checks = {
        "No missing values": int(df.isna().sum().sum()) == 0,
        "No duplicate rows": int(df.duplicated().sum()) == 0,
        "No constant columns": int((df.nunique(dropna=True) <= 1).sum()) == 0,
        "Outliers under 1% of values": out[0] <= 0.01 * n_cells,
        "All columns numeric": len(cats(df)) == 0,
        "Train/test split done": bool(split_done),
    }
    return round(100 * sum(checks.values()) / len(checks)), checks


def find_issues(df, target):
    out = []
    for c, p in (df.isna().mean() * 100).items():
        if p > 0:
            out.append((3 if p > 30 else 2, f"`{c}` has {p:.1f}% missing values", "Clean"))
    d = int(df.duplicated().sum())
    if d:
        out.append((2, f"{d} duplicate rows", "Clean"))
    const = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        out.append((2, f"Constant columns: {', '.join(const)}", "Clean"))
    for c, r in outlier_info(df)[1].items():
        if r > 5:
            out.append((2, f"`{c}` has {r:.1f}% outliers (IQR rule)", "Transform"))
    for c in nums(df):
        s = df[c].dropna()
        if len(s) > 2 and abs(s.skew()) > 1:
            out.append((1, f"`{c}` is skewed (skew {s.skew():.1f})", "Transform"))
    cc = cats(df)
    if cc:
        out.append((2, f"{len(cc)} categorical columns are not encoded: {', '.join(map(str, cc[:5]))}", "Encode"))
    if target in df.columns:
        s = df[target].dropna()
        if df[target].isna().any():
            out.append((3, f"Target `{target}` has {int(df[target].isna().sum())} missing values", "Clean"))
        if len(s) and is_class(s) and s.value_counts(normalize=True).min() < 0.2:
            out.append((2, f"Target `{target}` is imbalanced", "Transform"))
    return sorted(out, key=lambda x: -x[0])


def cramers(x, y):
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    r, k = ct.shape
    if n == 0 or min(r, k) < 2:
        return 0.0
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    return float(np.sqrt(((ct.values - exp) ** 2 / exp).sum() / n / (min(r, k) - 1)))


def eta(g, v):
    tot = ((v - v.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    grp = v.groupby(g)
    return float(np.sqrt((grp.count() * (grp.mean() - v.mean()) ** 2).sum() / tot))


def target_strength(df, t):
    N, rows = set(nums(df)), []
    for x in df.columns:
        d = df[[x, t]].dropna() if x != t else None
        if d is None or len(d) < 5:
            continue
        if x in N and t in N:
            v = abs(d[x].corr(d[t]))
        elif x in N:
            v = eta(d[t], d[x])
        elif t in N:
            v = eta(d[x], d[t])
        else:
            v = cramers(d[x], d[t])
        rows.append({"column": x, "strength (0 to 1)": 0.0 if pd.isna(v) else v})
    return pd.DataFrame(rows).sort_values("strength (0 to 1)", ascending=False).head(10)


def facts(d):
    return {"Rows": len(d), "Columns": d.shape[1], "Missing cells": int(d.isna().sum().sum()),
            "Duplicate rows": int(d.duplicated().sum()), "Numeric columns": len(nums(d)),
            "Categorical columns": len(cats(d))}


def build_blocks(df, ss, chosen):
    B, n, t = [], nums(df), ss.target
    log = ss.log
    issues = find_issues(df, t)
    if "Summary" in chosen:
        score, checks = readiness(df, ss.split_done)
        B += [("h2", "Summary"),
              ("kv", [("Rows", f"{len(df):,}"), ("Columns", df.shape[1]), ("Numeric", len(n)),
                      ("Categorical", len(cats(df))), ("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%"),
                      ("Duplicate rows", int(df.duplicated().sum())),
                      ("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"),
                      ("Target", t or "none"), ("ML readiness", f"{score} / 100")]),
              ("p", f"{len(issues)} issue(s) found. Readiness counts these checks: "
                    + "; ".join(f"{k.lower()} ({'yes' if v else 'no'})" for k, v in checks.items()) + ".")]
    if "Findings" in chosen:
        B.append(("h2", "Findings"))
        if issues:
            B.append(("table", pd.DataFrame([{"priority": SEV[s], "finding": x, "fixed in": st_} for s, x, st_ in issues]), None, 1))
        else:
            B.append(("p", "No issues found."))
    if "Column profile" in chosen:
        rows = []
        for c in df.columns[:60]:
            s = df[c]
            isn = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            rows.append({"column": c, "type": str(s.dtype), "missing %": s.isna().mean() * 100,
                         "unique": s.nunique(),
                         "mean / most common": s.mean() if isn else (str(s.mode().iloc[0]) if s.notna().any() else "–")})
        B += [("h2", "Column profile"), ("table", pd.DataFrame(rows), "missing %", 100)]
        if df.shape[1] > 60:
            B.append(("note", "Only the first 60 columns are listed."))
    if "Statistics" in chosen and n:
        d = df[n[:40]].describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
        B += [("h2", "Statistics"), ("table", d.reset_index().rename(columns={"index": "column"}), None, 1)]
    if "Relationships" in chosen:
        B.append(("h2", "Relationships"))
        if len(n) >= 2:
            m = df[n[:40]].corr()
            p = pd.DataFrame([(a, b, m.loc[a, b]) for a, b in combinations(m.columns, 2) if pd.notna(m.loc[a, b])],
                             columns=["column A", "column B", "correlation"])
            p = p.reindex(p.correlation.abs().sort_values(ascending=False).index).head(8)
            B += [("p", "Strongest Pearson correlations between numeric columns:"), ("table", p, None, 1)]
        if t in df.columns:
            B += [("p", f"Columns most related to the target `{t}` (a quick screen; measures differ by column type):"),
                  ("table", target_strength(df, t), "strength (0 to 1)", 1)]
        if len(n) < 2 and t not in df.columns:
            B.append(("p", "Not enough numeric columns, and no target selected."))
    if "Pipeline log" in chosen:
        B.append(("h2", "Pipeline log"))
        if log:
            L = pd.DataFrame(log)
            B.append(("table", L[[c for c in ["time", "step", "detail", "rows", "cols"] if c in L]], None, 1))
        else:
            B.append(("p", "No steps recorded yet."))
    if "Before vs after" in chosen:
        B.append(("h2", "Before vs after"))
        if ss.snapshots:
            a, b = facts(ss.snapshots[0]["df"]), facts(df)
            tb = pd.DataFrame({"measure": list(a), "as loaded": list(a.values()), "now": list(b.values())})
            tb["change"] = tb["now"] - tb["as loaded"]
            B.append(("table", tb, None, 1))
        else:
            B.append(("p", "No snapshot of the original data is available."))
    if "Recommendations" in chosen:
        done = {l.get("step") for l in log}
        steps = {s for _, _, s in issues}
        sd = [df[c].std() for c in n if df[c].std() > 0]
        if sd and max(sd) / min(sd) > 10:
            steps.add("Scale")
        if not ss.split_done:
            steps.add("Split")
        recs = [ADVICE[s] for s in PIPELINE if s in steps and s in ADVICE and (s in {x for _, _, x in issues} or s not in done)]
        B += [("h2", "Recommendations"),
              ("list", recs) if recs else ("p", "The data looks ready. Export it when you are.")]
    return B


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.3g}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def inline_html(s):
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(str(s)))


LIVE_JS = r"""<script>
(function () {
  function solid(c) { return c && c !== "transparent" && !/,\s*0\)$/.test(c); }
  try {
    var P = window.parent, d = P.document;
    var cs = P.getComputedStyle(d.querySelector(".stApp") || d.body);
    var bg = cs.backgroundColor;
    if (!solid(bg)) bg = P.getComputedStyle(d.body).backgroundColor;
    var fg = cs.color, m = (bg.match(/[\d.]+/g) || [255, 255, 255]).map(Number);
    var dark = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) < 128;
    var r = document.documentElement;
    r.setAttribute("data-theme", dark ? "dark" : "light");
    if (solid(bg)) r.style.setProperty("--bg", bg);
    if (fg) r.style.setProperty("--fg", fg);
  } catch (e) {
    var q = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", q ? "dark" : "light");
  }
})();
</script>"""


def to_html(B, meta, theme="auto", live=False):
    attr = "" if theme == "auto" else f' data-theme="{theme}"'
    body = ""
    for b in B:
        k = b[0]
        if k == "h2":
            body += f"<h2>{html.escape(b[1])}</h2>"
        elif k in ("p", "note"):
            body += f'<p class="{"note" if k == "note" else ""}">{inline_html(b[1])}</p>'
        elif k == "kv":
            body += '<div class="kv">' + "".join(f"<div><span>{html.escape(a)}</span><b>{html.escape(str(v))}</b></div>" for a, v in b[1]) + "</div>"
        elif k == "list":
            body += "<ul>" + "".join(f"<li>{inline_html(x)}</li>" for x in b[1]) + "</ul>"
        elif k == "table":
            df, bar, mx = b[1], b[2], b[3]
            body += "<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
            for row in df.to_dict("records"):
                cells = ""
                for c, v in row.items():
                    extra = ""
                    if c == bar and isinstance(v, (int, float, np.number)) and not pd.isna(v):
                        extra = f'<span class="bar" style="width:{min(max(v / mx, 0), 1) * 80:.0f}px"></span>'
                    cells += f"<td>{inline_html(fmt(v))}{extra}</td>"
                body += f"<tr>{cells}</tr>"
            body += "</table>"
    return (f'<!doctype html><html lang="en"{attr}><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(meta['title'])}</title><style>{REPORT_CSS}</style></head><body><main>"
            f"<h1>{html.escape(meta['title'])}</h1><div class=\"meta\">{html.escape(meta['line'])}</div>{body}"
            f'<p class="note" style="margin-top:40px">Generated by DataForge.</p></main>' + (LIVE_JS if live else "") + '</body></html>')


def to_md(B, meta):
    out = [f"# {meta['title']}", f"*{meta['line']}*", ""]
    for b in B:
        k = b[0]
        if k == "h2":
            out += [f"## {b[1]}", ""]
        elif k in ("p", "note"):
            out += [str(b[1]), ""]
        elif k == "kv":
            out += [f"- **{a}:** {v}" for a, v in b[1]] + [""]
        elif k == "list":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "table":
            df = b[1]
            out.append("| " + " | ".join(map(str, df.columns)) + " |")
            out.append("|" + "---|" * len(df.columns))
            for row in df.to_dict("records"):
                out.append("| " + " | ".join(fmt(v).replace("|", "\\|") for v in row.values()) + " |")
            out.append("")
    return "\n".join(out)


def app_theme():
    """Best guess of the Streamlit app theme, used for the preview."""
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return st.get_option("theme.base") or "light"


def note_report(fmt_name):
    ss = st.session_state
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "step": "Report",
                   "detail": f"Downloaded report as {fmt_name}", "rows": len(ss.df), "cols": ss.df.shape[1]})


ss = st.session_state
for k, v in {"df": None, "df_name": "untitled", "target": None, "log": [], "snapshots": [], "split_done": False}.items():
    ss.setdefault(k, v)
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="dfd"><h1>Report</h1><p>A complete write-up of your project that you can '
            "download, share or print.</p></div>", unsafe_allow_html=True)

df = ss.df
if df is None:
    st.write("")
    st.info("No dataset yet. Add one first and the report will be built from it.")
    if st.button("Go to Dataset", type="primary"):
        try:
            st.switch_page(PAGES["dataset"])
        except Exception:
            st.toast(f"Page not found: {PAGES['dataset']}")
    st.stop()

st.markdown('<div class="dfd"><h2>Options</h2></div>', unsafe_allow_html=True)
a, b = st.columns(2)
title = a.text_input("Title", value=f"{ss.df_name} data report")
author = b.text_input("Prepared by (optional)")
chosen = st.multiselect("Sections", SECTIONS, default=SECTIONS)
theme_choice = st.radio("Report theme", ["Auto", "Light", "Dark"], horizontal=True,
                        help="Auto: the downloaded file follows the reader's device setting, "
                             "and the preview follows this app. Printing is always light.")

if not chosen:
    st.warning("Pick at least one section.")
    st.stop()

with st.spinner("Building the report..."):
    blocks = build_blocks(df, ss, chosen)
line = " · ".join(x for x in [f"Dataset: {ss.df_name}", f"{len(df):,} rows, {df.shape[1]} columns",
                              f"Prepared by {author}" if author.strip() else "",
                              datetime.now().strftime("%Y-%m-%d %H:%M")] if x)
meta = {"title": title.strip() or "Data report", "line": line}
dl_theme = theme_choice.lower()
html_doc = to_html(blocks, meta, dl_theme)
md_doc = to_md(blocks, meta)
preview_doc = to_html(blocks, meta, app_theme() if dl_theme == "auto" else dl_theme, live=dl_theme == "auto")
safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(ss.df_name).rsplit(".", 1)[0]) or "dataset"

st.markdown('<div class="dfd"><h2>Download</h2></div>', unsafe_allow_html=True)
c1, c2, _ = st.columns([1, 1, 2])
c1.download_button("HTML report", html_doc, f"{safe}_report.html", "text/html", type="primary",
                   use_container_width=True, on_click=note_report, args=("HTML",))
c2.download_button("Markdown", md_doc, f"{safe}_report.md", "text/markdown",
                   use_container_width=True, on_click=note_report, args=("Markdown",))
st.caption("To get a PDF, open the HTML file in your browser and choose Print, then Save as PDF. "
           "Downloading adds a Report entry to the log.")

st.markdown('<div class="dfd"><h2>Preview</h2></div>', unsafe_allow_html=True)
components.html(preview_doc, height=780, scrolling=True)