import html
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

PAGES = {
    "dataset": "pages/dataset.py",
    "prepare": "pages/prepare.py",
    "analysis": "pages/analysis.py",
    "export": "pages/export.py",
}
PIPELINE = ["Load", "Clean", "Transform", "Encode", "Scale", "Split"]
NONE = "None (unsupervised)"
DEMO = True

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
.block-container{max-width:1180px;padding-top:2.5rem}
.dfd{--amber:#e8960c;--blue:#3b6cff;--red:#e5484d;
--line:color-mix(in srgb,currentColor 16%,transparent);
--soft:color-mix(in srgb,currentColor 6%,transparent);
font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:34px 0 12px;padding:0}
.dfd p{margin:0}.dfd .mono{font-family:'IBM Plex Mono',monospace;font-size:13px;opacity:.65}
.dfd-head{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:center;
background:var(--soft);border-radius:14px;padding:22px 26px;margin:10px 0 6px}
.dfd-head .name{font-size:26px;font-weight:800;letter-spacing:-.02em}
.dfd-score{display:flex;align-items:center;gap:16px}
.ring{--c:var(--blue);width:92px;height:92px;border-radius:50%;display:grid;place-items:center;position:relative;
background:conic-gradient(var(--c) calc(var(--p)*1%),var(--line) 0)}
.ring::before{content:"";position:absolute;inset:9px;border-radius:50%;background:var(--background-color,Canvas)}
.ring b{position:relative;font-size:26px}
.dfd-score .lbl{font-weight:600;font-size:17px}
.steps{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(6,1fr);border-top:1px solid var(--line)}
.steps li{position:relative;padding:16px 10px 0 0}
.steps li::before{content:"";position:absolute;top:-6px;left:0;width:11px;height:11px;border-radius:50%;
background:var(--background-color,Canvas);box-shadow:inset 0 0 0 2px var(--line)}
.steps li.done::before{background:var(--blue);box-shadow:none}
.steps li.next::before{background:var(--amber);box-shadow:0 0 0 4px color-mix(in srgb,var(--amber) 25%,transparent)}
.steps b{display:block;font-size:16px}.steps span{font-size:12px;opacity:.6}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:10px}
.s3{background:var(--red)}.s2{background:var(--amber)}.s1{background:var(--blue)}
.tag{font-family:'IBM Plex Mono',monospace;font-size:12px;padding:2px 8px;border-radius:5px;background:var(--soft);margin-left:8px}
.issue{padding:6px 0}
@media(max-width:820px){.dfd-head{grid-template-columns:1fr}.steps{grid-template-columns:repeat(3,1fr);row-gap:24px}}
</style>
"""


def init_state():
    ss = st.session_state
    for k, v in {"df": None, "df_name": "untitled", "target": None, "log": [],
                 "snapshots": [], "modified_at": None, "split_done": False}.items():
        ss.setdefault(k, v)


def go(page, **state):
    st.session_state.update(state)
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def load_demo():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.lognormal(10.5, 0.6, n),
        "city": rng.choice(["Tehran", "Shiraz", "Tabriz", "Isfahan"], n),
        "plan": rng.choice(["free", "pro"], n, p=[0.85, 0.15]),
        "country": "IR",
        "churned": rng.choice([0, 1], n, p=[0.88, 0.12]),
    })
    df.loc[rng.choice(n, 50, replace=False), "age"] = np.nan
    df = pd.concat([df, df.head(12)], ignore_index=True)
    now = datetime.now().strftime("%H:%M:%S")
    ss = st.session_state
    ss.df, ss.df_name, ss.modified_at = df, "demo.csv", now
    ss.snapshots = [{"name": "Loaded", "df": df.copy()}]
    ss.log = [{"time": now, "step": "Load", "detail": "Loaded demo.csv",
               "rows": len(df), "cols": df.shape[1]}]


def health(df):
    num = df.select_dtypes("number")
    cat = df.select_dtypes(exclude=["number", "bool", "datetime", "timedelta"])
    outliers = 0
    for c in num:
        s = num[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            outliers += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    return {
        "missing": float(df.isna().sum().sum() / max(df.size, 1) * 100),
        "dups": int(df.duplicated().sum()),
        "constant": int((df.nunique(dropna=True) <= 1).sum()),
        "outliers": outliers,
        "out_ok": outliers <= 0.01 * max(num.size, 1),
        "unencoded": int(cat.shape[1]),
        "high_card": int(sum(df[c].nunique() > 20 and df[c].nunique() / len(df) > 0.5 for c in cat)),
    }


def readiness(h):
    checks = {
        "No missing values": h["missing"] == 0,
        "No duplicate rows": h["dups"] == 0,
        "No constant columns": h["constant"] == 0,
        "Outliers under 1% of values": h["out_ok"],
        "All columns numeric": h["unencoded"] == 0,
        "Train/test split done": bool(st.session_state.split_done),
    }
    return round(100 * sum(checks.values()) / len(checks)), checks


def is_classification(s):
    return not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 20


def find_issues(df, target):
    out = []
    e = html.escape
    for c, p in (df.isna().mean() * 100).items():
        if p > 0:
            out.append((3 if p > 30 else 2, f"<code>{e(str(c))}</code> has {p:.1f}% missing values", "Clean"))
    d = int(df.duplicated().sum())
    if d:
        out.append((2, f"{d} duplicate rows", "Clean"))
    const = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        out.append((2, f"Constant columns: {e(', '.join(const))}", "Clean"))
    for c in df.select_dtypes("number"):
        s = df[c].dropna()
        if len(s) >= 4:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                r = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
                if r > 5:
                    out.append((2, f"<code>{e(str(c))}</code> has {r:.1f}% outliers", "Transform"))
    cat = [c for c in df.select_dtypes(exclude=["number", "bool", "datetime", "timedelta"])]
    if cat:
        out.append((2, f"{len(cat)} categorical columns are not encoded: {e(', '.join(map(str, cat[:4])))}", "Encode"))
    for c in cat:
        if df[c].nunique() > 20 and df[c].nunique() / len(df) > 0.5:
            out.append((1, f"<code>{e(str(c))}</code> has very high cardinality", "Encode"))
    if target in df.columns:
        s = df[target]
        if s.isna().any():
            out.append((3, f"Target <code>{e(str(target))}</code> has {int(s.isna().sum())} missing values", "Clean"))
        s = s.dropna()
        if len(s) and is_classification(s) and s.value_counts(normalize=True).min() < 0.2:
            out.append((2, f"Target <code>{e(str(target))}</code> is imbalanced", "Transform"))
    return sorted(out, key=lambda x: -x[0])


def overview(df):
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().sum() > 1:
            dist = np.histogram(s.dropna(), bins=10)[0].tolist()
        else:
            dist = s.value_counts().head(10).tolist()
        rows.append({"column": str(c), "type": str(s.dtype), "missing %": float(s.isna().mean() * 100),
                     "unique": int(s.nunique()), "distribution": dist or [0]})
    return pd.DataFrame(rows)


def fmt_bytes(n):
    return f"{n / 1024:.0f} KB" if n < 1024**2 else f"{n / 1024**2:.1f} MB"


def section(title):
    st.markdown(f'<div class="dfd"><h2>{title}</h2></div>', unsafe_allow_html=True)


init_state()
ss = st.session_state
st.markdown(CSS, unsafe_allow_html=True)
df = ss.df

if df is None:
    st.markdown('<div class="dfd"><h1>Dashboard</h1><p style="margin-top:10px;opacity:.7">'
                "No dataset yet. Add one to see its health, issues and next steps here.</p></div>",
                unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 3])
    if a.button("Go to Dataset", type="primary", use_container_width=True):
        go("dataset")
    if DEMO and b.button("Load demo data", use_container_width=True):
        load_demo()
        st.rerun()
    st.stop()

left, right = st.columns([3, 1.4], vertical_alignment="bottom")
left.markdown('<div class="dfd"><h1>Dashboard</h1></div>', unsafe_allow_html=True)
opts = [NONE] + [str(c) for c in df.columns]
cur = ss.target if ss.target in df.columns else NONE
choice = right.selectbox("Target column", opts, index=opts.index(cur))
ss.target = None if choice == NONE else choice

h = health(df)
score, checks = readiness(h)
color = "var(--blue)" if score >= 60 else "var(--amber)"
label = "Ready for ML" if score >= 90 else "Almost there" if score >= 60 else "Needs work"
st.markdown(
    f'<div class="dfd"><div class="dfd-head"><div>'
    f'<div class="name">{html.escape(str(ss.df_name))}</div>'
    f'<div class="mono">{len(df):,} rows &middot; {df.shape[1]} columns &middot; '
    f'{fmt_bytes(int(df.memory_usage(deep=True).sum()))} &middot; '
    f'last change {ss.modified_at or "-"}</div></div>'
    f'<div class="dfd-score"><div class="ring" style="--p:{score};--c:{color}"><b>{score}</b></div>'
    f'<div><div class="lbl">{label}</div><div class="mono">ML readiness</div></div></div>'
    f"</div></div>",
    unsafe_allow_html=True,
)
with st.expander("What counts toward the score"):
    for name, ok in checks.items():
        st.markdown(f"{'✅' if ok else '⬜'} {name}")

section("Data health")
base = health(ss.snapshots[0]["df"]) if len(ss.snapshots) > 1 else None


def delta(key, pct=False):
    if base is None or base[key] == h[key]:
        return None
    d = h[key] - base[key]
    return f"{d:+.1f}%" if pct else f"{d:+d}"


cards = [
    ("Missing", f"{h['missing']:.1f}%", delta("missing", True)),
    ("Duplicates", h["dups"], delta("dups")),
    ("Constant cols", h["constant"], delta("constant")),
    ("Outliers", h["outliers"], delta("outliers")),
    ("Not encoded", h["unencoded"], delta("unencoded")),
    ("High cardinality", h["high_card"], delta("high_card")),
]
for col, (name, val, d) in zip(st.columns(6), cards):
    col.metric(name, val, d, delta_color="inverse")
if base is not None:
    st.caption("Arrows show the change since the data was first loaded.")

section("Pipeline")
done = {l["step"] for l in ss.log} | {"Load"}
if ss.split_done:
    done.add("Split")
nxt = next((s for s in PIPELINE if s not in done), None)
items = ""
for s in PIPELINE:
    cls, txt = ("done", "done") if s in done else ("next", "next") if s == nxt else ("todo", "to do")
    items += f'<li class="{cls}"><b>{s}</b><span>{txt}</span></li>'
st.markdown(f'<div class="dfd"><ul class="steps">{items}</ul></div>', unsafe_allow_html=True)
if nxt:
    if st.button(f"Continue with {nxt}", type="primary"):
        go("prepare", prepare_focus=nxt)
else:
    st.success("Every pipeline step is done. Review the log and export your data.")

issues = find_issues(df, ss.target)
section(f"Issues ({len(issues)})")
if not issues:
    st.success("No issues found.")
for i, (sev, text, step) in enumerate(issues[:8]):
    a, b = st.columns([6, 1], vertical_alignment="center")
    a.markdown(f'<div class="dfd issue"><span class="dot s{sev}"></span>{text}'
               f'<span class="tag">{step}</span></div>', unsafe_allow_html=True)
    if b.button("Fix", key=f"fix_{i}", use_container_width=True):
        go("prepare", prepare_focus=step)
if len(issues) > 8:
    st.caption(f"+ {len(issues) - 8} more issues in Prepare.")

section("Columns")
st.dataframe(
    overview(df), hide_index=True, use_container_width=True,
    column_config={
        "missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%"),
        "distribution": st.column_config.BarChartColumn("distribution"),
    },
)

c1, c2 = st.columns(2)
with c1:
    section("Missing values")
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values.")
    else:
        st.bar_chart(miss.rename("missing %"))
with c2:
    section("Target distribution")
    if not ss.target:
        st.caption("Choose a target column above to see its distribution.")
    else:
        s = df[ss.target].dropna()
        if is_classification(s):
            vc = s.value_counts()
            vc.index = vc.index.astype(str)
            st.bar_chart(vc.rename("count"))
        else:
            counts, edges = np.histogram(s, bins=20)
            st.bar_chart(pd.Series(counts, index=np.round(edges[:-1], 3), name="count"))

section("Recent activity")
if ss.log:
    recent = pd.DataFrame(ss.log[-5:][::-1])[["time", "step", "detail"]]
    st.dataframe(recent, hide_index=True, use_container_width=True)
else:
    st.caption("Nothing has happened yet.")
if st.button("Open full log"):
    go("export")