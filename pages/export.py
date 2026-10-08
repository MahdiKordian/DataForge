import html
import io
import json
from datetime import datetime
import pandas as pd
import streamlit as st

PAGES = {"dataset": "pages/dataset.py", "dashboard": "pages/dashboard.py"}
STEP_COLORS = {"Load": "#3b6cff", "Clean": "#e8960c", "Transform": "#14b8a6",
               "Encode": "#8b5cf6", "Scale": "#ec4899", "Split": "#22c55e", "Export": "#64748b"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
.block-container{max-width:1080px;padding-top:2.5rem}
.dfd{--line:color-mix(in srgb,currentColor 16%,transparent);--soft:color-mix(in srgb,currentColor 6%,transparent);
font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.dfd h1{font-size:44px;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.dfd h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:32px 0 10px;padding:0}
.dfd .mono{font-family:'IBM Plex Mono',monospace;font-size:13px;opacity:.65}
.tl{max-height:560px;overflow:auto;border-top:1px solid var(--line);padding-top:6px}
.tl-item{display:grid;grid-template-columns:18px 1fr;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}
.tl-dot{width:10px;height:10px;border-radius:50%;margin-top:7px}
.tl-top{display:flex;align-items:center;gap:10px;margin-bottom:2px}
.tl-step{font-weight:600;font-size:13px;padding:1px 9px;border-radius:5px;
color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent)}
</style>
"""


def go(page):
    try:
        st.switch_page(PAGES[page])
    except Exception:
        st.toast(f"Page not found: {PAGES[page]}")


def section(t):
    st.markdown(f'<div class="dfd"><h2>{t}</h2></div>', unsafe_allow_html=True)


def enrich(log):
    """Adds row_change / col_change versus the previous entry."""
    out, prev = [], (None, None)
    for e in log:
        r, c = e.get("rows"), e.get("cols")
        d = dict(e)
        d["row_change"] = r - prev[0] if r is not None and prev[0] is not None else None
        d["col_change"] = c - prev[1] if c is not None and prev[1] is not None else None
        out.append(d)
        prev = (r, c)
    return out


def shape_text(e):
    if e.get("rows") is None:
        return ""
    s = f"{e['rows']:,} rows &middot; {e.get('cols', '?')} columns"
    ch = []
    if e.get("row_change"):
        ch.append(f"{e['row_change']:+,} rows")
    if e.get("col_change"):
        ch.append(f"{e['col_change']:+} columns")
    return s + (f" ({', '.join(ch)})" if ch else "")


def timeline_html(entries):
    items = ""
    for e in entries:
        step = str(e.get("step", ""))
        col = STEP_COLORS.get(step, "#64748b")
        items += (f'<div class="tl-item"><span class="tl-dot" style="background:{col}"></span><div>'
                  f'<div class="tl-top"><span class="mono">{html.escape(str(e.get("time", "")))}</span>'
                  f'<span class="tl-step" style="--c:{col}">{html.escape(step)}</span></div>'
                  f'<div>{html.escape(str(e.get("detail", "")))}</div>'
                  f'<div class="mono">{shape_text(e)}</div></div></div>')
    return f'<div class="dfd"><div class="tl">{items}</div></div>'


def log_json(log, name, target, shape):
    return json.dumps({"dataset": name, "target": target, "exported_at": datetime.now().isoformat(timespec="seconds"),
                       "final_shape": {"rows": shape[0], "columns": shape[1]}, "steps": log},
                      indent=2, ensure_ascii=False, default=str)


def log_text(log, name, target, shape):
    lines = ["DataForge log", f"Dataset:  {name}", f"Target:   {target or 'none'}",
             f"Exported: {datetime.now():%Y-%m-%d %H:%M:%S}", f"Steps:    {len(log)}",
             f"Final:    {shape[0]:,} rows x {shape[1]} columns", "-" * 56]
    for e in enrich(log):
        line = f"[{e.get('time', '')}] {str(e.get('step', '')):<10} {e.get('detail', '')}"
        if e.get("rows") is not None:
            line += f"  ({e['rows']:,} x {e.get('cols', '?')})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def data_bytes(df, fmt, index, bom):
    if fmt == "CSV":
        return df.to_csv(index=index).encode("utf-8-sig" if bom else "utf-8"), "text/csv", "csv"
    if fmt == "JSON":
        return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"), "application/json", "json"
    buf = io.BytesIO()
    if fmt == "Excel":
        df.to_excel(buf, index=index)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    df.to_parquet(buf, index=index)
    return buf.getvalue(), "application/octet-stream", "parquet"


def note_export(fmt):
    ss = st.session_state
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "step": "Export",
                   "detail": f"Downloaded data as {fmt}", "rows": len(ss.df), "cols": ss.df.shape[1]})


ss = st.session_state
for k, v in {"df": None, "df_name": "untitled", "target": None, "log": []}.items():
    ss.setdefault(k, v)
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="dfd"><h1>Export</h1></div>', unsafe_allow_html=True)

df = ss.df
if df is None:
    st.info("No dataset yet. Add one first, then come back to review and download your work.")
    if st.button("Go to Dataset", type="primary"):
        go("dataset")
    st.stop()

name = str(ss.df_name)
safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0]) or "dataset"
log = enrich(ss.log)
shape = (len(df), df.shape[1])

m = st.columns(5)
m[0].metric("Steps", len(log))
m[1].metric("Started", log[0].get("time", "-") if log else "-")
m[2].metric("Last step", log[-1].get("time", "-") if log else "-")
first_rows = log[0].get("rows") if log else None
first_cols = log[0].get("cols") if log else None
m[3].metric("Rows now", f"{shape[0]:,}", f"{shape[0] - first_rows:+,}" if first_rows is not None else None, delta_color="off")
m[4].metric("Columns now", shape[1], f"{shape[1] - first_cols:+}" if first_cols is not None else None, delta_color="off")

section("Project log")
if not log:
    st.info("The log is empty. Every step you run in Prepare is recorded here.")
else:
    a, b = st.columns([2, 2])
    all_steps = list(dict.fromkeys(str(e.get("step", "")) for e in log))
    picked = a.multiselect("Steps", all_steps, default=all_steps)
    query = b.text_input("Search details", placeholder="e.g. age")
    view = st.radio("View", ["Timeline", "Table"], horizontal=True, label_visibility="collapsed")
    shown = [e for e in log if str(e.get("step", "")) in picked
             and query.lower() in str(e.get("detail", "")).lower()]
    if not shown:
        st.warning("No entries match these filters.")
    elif view == "Timeline":
        st.markdown(timeline_html(shown), unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame(shown), hide_index=True, use_container_width=True)
    st.caption(f"Showing {len(shown)} of {len(log)} entries. Downloads below always include the full log.")

    section("Download the log")
    raw = [{k: v for k, v in e.items() if k not in ("row_change", "col_change")} for e in log]
    c1, c2, c3 = st.columns(3)
    c1.download_button("CSV", pd.DataFrame(raw).to_csv(index=False).encode("utf-8-sig"),
                       f"{safe}_log.csv", "text/csv", use_container_width=True)
    c2.download_button("JSON", log_json(raw, name, ss.target, shape), f"{safe}_log.json",
                       "application/json", use_container_width=True)
    c3.download_button("Text report", log_text(raw, name, ss.target, shape), f"{safe}_log.txt",
                       "text/plain", use_container_width=True)

section("Download the prepared data")
st.caption(f"This is the data as it is right now: {shape[0]:,} rows and {shape[1]} columns.")
a, b = st.columns([1, 2])
fmt = a.selectbox("Format", ["CSV", "Excel", "Parquet", "JSON"])
index = b.checkbox("Include the row index", value=False)
bom = fmt == "CSV" and b.checkbox("Excel-friendly (UTF-8 with BOM, keeps Persian text readable)", value=True)
if fmt == "Excel" and shape[0] > 1_048_575:
    st.error("Excel cannot hold more than 1,048,575 rows. Choose CSV or Parquet.")
else:
    try:
        data, mime, ext = data_bytes(df, fmt, index, bom)
        st.download_button(f"Download {fmt}", data, f"{safe}_prepared.{ext}", mime, type="primary",
                           on_click=note_export, args=(fmt,))
        st.caption(f"File size: {len(data) / 1024:,.0f} KB. Downloading adds an Export entry to the log.")
    except ImportError as e:
        pkg = "openpyxl" if fmt == "Excel" else "pyarrow"
        st.error(f"{fmt} export needs the `{pkg}` package. Install it or choose another format. ({e})")
    except Exception as e:
        st.error(f"Could not create the {fmt} file: {e}")