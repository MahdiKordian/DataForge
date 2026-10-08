import html
import streamlit as st

from core import (KINDS, PAGES, SAMPLES, VERSION, commit, find_column, go,
                  init_state, load_sample, read_upload)

STEPS = [
    ("Load", "Upload a file or pick a sample."),
    ("Clean", "Fix missing values, duplicates and messy text."),
    ("Transform", "Tame outliers and skewed columns."),
    ("Encode", "Turn categories into numbers."),
    ("Scale", "Bring columns to a similar range."),
    ("Split", "Create train, validation and test sets."),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
.block-container{max-width:1080px;padding-top:2.5rem}
.lp{--amber:#e8960c;--blue:#3b6cff;
--line:color-mix(in srgb,currentColor 16%,transparent);
--soft:color-mix(in srgb,currentColor 6%,transparent);
font-family:'Bricolage Grotesque',system-ui,sans-serif;line-height:1.5}
.lp h1,.lp h2,.lp h3,.lp p{margin:0;padding:0;font-family:inherit}
.lp .mono{font-family:'IBM Plex Mono',monospace}

.lp-hero{padding:4px 0 26px}
.lp-brand{display:flex;align-items:center;gap:0;font-weight:600;font-size:18px;line-height:1;margin-bottom:26px}
.lp-brand span.fire{font-size:22px;line-height:1;display:inline-block;transform:translateY(-2px)}
.lp-chip{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:400;padding:2px 8px;
border-radius:5px;background:var(--soft);opacity:.8;margin-left:10px}
.lp-hero h1{font-size:clamp(38px,6.4vw,70px);font-weight:800;line-height:1.02;letter-spacing:-.04em;max-width:15ch}
.lp-hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--amber),var(--blue));
-webkit-background-clip:text;background-clip:text;color:transparent}
.lp-hero p{margin-top:20px;max-width:54ch;font-size:18px;opacity:.72}

.lp h2{font-size:26px;font-weight:800;letter-spacing:-.03em;margin:46px 0 6px}
.lp-sub{font-size:16px;opacity:.68;margin-bottom:14px}

.lp-cur{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;
background:var(--soft);border-radius:12px;border-left:4px solid var(--blue);padding:14px 18px;margin:6px 0 12px}
.lp-cur b{font-size:18px}
.lp-cur .mono{font-size:13px;opacity:.65}

.lp-up{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.lp-up b{font-size:18px}
.lp-fmt{font-family:'IBM Plex Mono',monospace;font-size:12px;opacity:.6}

.lp-sample{min-height:218px;display:flex;flex-direction:column;gap:8px}
.lp-top{display:flex;align-items:center;justify-content:space-between}
.lp-emoji{font-size:28px;line-height:1}
.lp-task{font-family:'IBM Plex Mono',monospace;font-size:12px;padding:2px 8px;border-radius:5px}
.lp-task.reg{background:color-mix(in srgb,var(--amber) 18%,transparent);color:var(--amber)}
.lp-task.clf{background:color-mix(in srgb,var(--blue) 16%,transparent);color:var(--blue)}
.lp-sample h3{font-size:21px;font-weight:600;letter-spacing:-.01em}
.lp-sample p{font-size:14.5px;opacity:.72}
.lp-meta{font-family:'IBM Plex Mono',monospace;font-size:13px;margin-top:auto}
.lp-meta span{opacity:.55;margin-right:8px}
.lp-tags{display:flex;flex-wrap:wrap;gap:6px}
.lp-tags span{font-size:12px;padding:2px 9px;border-radius:999px;border:1px solid var(--line);opacity:.8}

.lp-steps{list-style:none;margin:18px 0 0;padding:0;display:grid;grid-template-columns:repeat(6,1fr);
border-top:1px solid var(--line)}
.lp-steps li{position:relative;padding:18px 14px 0 0}
.lp-steps li::before{content:"";position:absolute;top:-5px;left:0;width:9px;height:9px;border-radius:50%;background:var(--blue)}
.lp-steps .n{font-family:'IBM Plex Mono',monospace;font-size:12px;opacity:.5;display:block;margin-bottom:2px}
.lp-steps b{font-size:17px}
.lp-steps p{margin-top:4px;font-size:13.5px;opacity:.7}

@media(max-width:900px){.lp-steps{grid-template-columns:repeat(3,1fr);row-gap:26px}}
@media(max-width:520px){.lp-steps{grid-template-columns:repeat(2,1fr)}}
</style>
"""

HIDE_SIDEBAR = """
<style>
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
</style>
"""


def sample_card(s: dict) -> str:
    e = html.escape
    tone = "reg" if s["task"] == "Regression" else "clf"
    tags = "".join(f"<span>{e(t)}</span>" for t in s["tags"])
    return (
        '<div class="lp"><div class="lp-sample">'
        f'<div class="lp-top"><span class="lp-emoji">{s["icon"]}</span>'
        f'<span class="lp-task {tone}">{e(s["task"])}</span></div>'
        f'<h3>{e(s["title"])}</h3><p>{e(s["blurb"])}</p>'
        f'<div class="lp-meta"><span>target</span><b>{e(s["target"])}</b></div>'
        f'<div class="lp-tags">{tags}</div>'
        "</div></div>"
    )


def steps_html() -> str:
    items = "".join(f'<li><span class="n">{i:02d}</span><b>{t}</b><p>{d}</p></li>'
                    for i, (t, d) in enumerate(STEPS, 1))
    return f'<div class="lp"><ol class="lp-steps">{items}</ol></div>'


init_state()
ss = st.session_state
has_data = ss.df is not None

st.markdown(CSS, unsafe_allow_html=True)
if not has_data:
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)

st.markdown(
    '<div class="lp"><section class="lp-hero">'
    f'<div class="lp-brand"><span class="fire">🔥</span>DataForge<span class="lp-chip">{html.escape(VERSION)}</span></div>'
    "<h1>Get your data <em>ready for machine learning.</em></h1>"
    "<p>Load a dataset, see what is wrong with it, fix it step by step "
    "and export something your model can use.</p>"
    "</section></div>",
    unsafe_allow_html=True,
)

if has_data:
    st.markdown(
        '<div class="lp"><div class="lp-cur"><div>'
        '<div class="mono">current dataset</div>'
        f"<b>{html.escape(str(ss.df_name))}</b></div>"
        f'<div class="mono">{len(ss.df):,} rows &middot; {ss.df.shape[1]} columns</div></div></div>',
        unsafe_allow_html=True,
    )
    a, _ = st.columns([1, 3])
    if a.button("Open dashboard", type="primary", width="stretch"):
        go("dashboard")

st.markdown('<div class="lp"><h2>Bring your data</h2></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        '<div class="lp"><div class="lp-up"><b>Upload a file</b>'
        '<span class="lp-fmt">CSV &middot; TSV &middot; Excel &middot; JSON &middot; Parquet</span></div></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload a dataset", type=list(KINDS), label_visibility="collapsed")
    if has_data:
        st.caption("Loading a new dataset replaces the current one and clears its log.")
    st.page_link(PAGES["dataset"], icon=":material/arrow_forward:",
                 label="More options: paste text, load from a URL or choose an Excel sheet")

if uploaded is not None:
    try:
        with st.spinner("Reading your file..."):
            df, name = read_upload(uploaded)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
    else:
        commit(df, name, f"file {uploaded.name}", None)
        go("dashboard")

st.markdown(
    '<div class="lp"><h2>Or start with a sample</h2>'
    '<p class="lp-sub">Real datasets with the usual problems, so every step has something to do.</p></div>',
    unsafe_allow_html=True,
)
for col, s in zip(st.columns(len(SAMPLES)), SAMPLES):
    with col.container(border=True):
        st.markdown(sample_card(s), unsafe_allow_html=True)
        if st.button("Use this sample", key=f"sample_{s['key']}", width="stretch"):
            try:
                with st.spinner(f"Loading {s['title']}..."):
                    df, name = load_sample(s)
            except Exception as e:
                st.error(f"Could not load this sample: {e}")
            else:
                commit(df, name, "sample", find_column(df, s["target"]))
                go("dashboard")

st.markdown('<div class="lp"><h2>How it works</h2></div>', unsafe_allow_html=True)
st.markdown(steps_html(), unsafe_allow_html=True)

st.write("")
st.page_link(PAGES["about"], icon=":material/auto_awesome:", label="About DataForge")
st.caption("Your data stays in the current session and is not stored anywhere.")