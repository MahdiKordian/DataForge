import streamlit as st

HEADER = ("name", "age", "city")

DIRTY = [
    [("ali reza", "bad"), ("34", ""), ("Tehran", "")],
    [("ALI REZA", "bad"), ("34", ""), ("tehran ", "bad")],
    [("Sara", ""), ("NaN", "bad"), ("Shiraz", "")],
    [("Nima", ""), ("29", ""), ("Tabriz", "")],
]

CLEAN = [
    [("Ali Reza", "fix"), ("34", ""), ("Tehran", "fix")],
    [("Sara", ""), ("31.5", "fix"), ("Shiraz", "")],
    [("Nima", ""), ("29", ""), ("Tabriz", "")],
]

STEPS = [
    ("Clean", "Fix missing values, duplicates and messy text."),
    ("Transform", "Encode, scale and reshape columns."),
    ("Analyze", "See distributions, correlations and outliers."),
    ("Train", "Export a dataset your model can use."),
]

FEATURES = [
    ("Data cleaning", "Handle missing values, duplicates and inconsistent entries without writing a line of pandas."),
    ("Data transformation", "Turn raw columns into a cleaner, more useful format in a few clicks."),
    ("Data analysis", "Explore your dataset and spot patterns, trends and problems early."),
    ("ML-ready output", "Get a structured, clean dataset that goes straight into your training workflow."),
    ("Fast workflow", "Skip the repetitive notebook code and spend your time on the model."),
    ("Simple interface", "One clean screen per task, so data preparation never feels like a project of its own."),
]



def table_html(rows, title, caption, kind):
    head = "".join(f"<span>{h}</span>" for h in HEADER)
    body = ""
    for row in rows:
        cells = "".join(f'<span class="c {flag}">{text}</span>' for text, flag in row)
        body += f'<div class="r">{cells}</div>'
    return (
        f'<div class="df-table {kind}">'
        f'<div class="df-table-title">{title}</div>'
        f'<div class="r h">{head}</div>{body}'
        f'<div class="df-table-cap">{caption}</div>'
        f"</div>"
    )


steps_html = "".join(
    f'<li><span class="n">{i}</span><b>{t}</b><p>{d}</p></li>'
    for i, (t, d) in enumerate(STEPS, 1)
)

features_html = "".join(
    f'<div class="df-feat"><h3>{t}</h3><p>{d}</p></div>' for t, d in FEATURES
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&display=swap');

.block-container { max-width: 1080px; padding-top: 3rem; }

.df {
    --amber: #e8960c;
    --amber-bg: rgba(232, 150, 12, 0.16);
    --blue: #3b6cff;
    --blue-bg: rgba(59, 108, 255, 0.14);
    --line: color-mix(in srgb, currentColor 16%, transparent);
    --soft: color-mix(in srgb, currentColor 6%, transparent);
    font-family: 'Bricolage Grotesque', system-ui, sans-serif;
    color: inherit;
    line-height: 1.55;
}
.df p { margin: 0; }
.df h1, .df h2, .df h3 { margin: 0; padding: 0; font-family: inherit; }

/* ---------- Hero ---------- */
.df-hero { padding: 8px 0 44px; }
.df-hero h1 {
    font-size: clamp(40px, 7vw, 76px);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    max-width: 14ch;
}
.df-hero p {
    margin-top: 22px;
    max-width: 52ch;
    font-size: 18px;
    opacity: 0.72;
}
.df-brand {
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.df-brand {
    font-weight: 600;
    font-size: 18px;
    line-height: 1;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    gap: 0;
}
.df-brand span { font-size: 22px; line-height: 1; display: block; transform: translateY(-2px); }

/* ---------- Before / After ---------- */
.df-stage {
    display: grid;
    grid-template-columns: 1fr 64px 1fr;
    align-items: center;
    gap: 10px;
    margin-bottom: 72px;
}
.df-arrow { text-align: center; font-size: 26px; opacity: 0.45; }

.df-table {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 16px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
}
.df-table.before { border-top: 3px solid var(--amber); }
.df-table.after  { border-top: 3px solid var(--blue); }
.df-table-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 600; font-size: 15px; margin-bottom: 10px;
}
.df-table .r {
    display: grid;
    grid-template-columns: 1.3fr 0.6fr 1fr;
    gap: 6px;
    padding: 3px 0;
}
.df-table .r.h { opacity: 0.5; font-size: 12px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.df-table .c { padding: 3px 7px; border-radius: 5px; }
.df-table .c.bad { background: var(--amber-bg); box-shadow: inset 0 0 0 1px var(--amber); }
.df-table .c.fix { background: var(--blue-bg); box-shadow: inset 0 0 0 1px var(--blue); }
.df-table-cap {
    margin-top: 12px;
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 13px; opacity: 0.7;
}
.df-table.before .c.bad { animation: flag 1.4s ease-out 0.4s 1 both; }
@keyframes flag {
    from { background: transparent; box-shadow: inset 0 0 0 1px transparent; }
}

/* ---------- Pipeline ---------- */
.df h2 { font-size: 30px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 22px; }
.df-steps {
    list-style: none; margin: 0 0 72px; padding: 0;
    display: grid; grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid var(--line);
}
.df-steps li { padding: 18px 18px 0 0; position: relative; }
.df-steps li::before {
    content: ""; position: absolute; top: -5px; left: 0;
    width: 9px; height: 9px; border-radius: 50%; background: var(--blue);
}
.df-steps .n { font-family: 'IBM Plex Mono', monospace; font-size: 12px; opacity: 0.5; display: block; margin-bottom: 4px; }
.df-steps b { font-size: 18px; }
.df-steps p { margin-top: 4px; font-size: 14px; opacity: 0.7; max-width: 24ch; }

/* ---------- Features ---------- */
.df-feats { margin-bottom: 72px; border-top: 1px solid var(--line); }
.df-feat {
    display: grid; grid-template-columns: 1fr 2fr; gap: 24px;
    padding: 20px 0; border-bottom: 1px solid var(--line);
}
.df-feat h3 { font-size: 19px; font-weight: 600; }
.df-feat p { opacity: 0.72; max-width: 60ch; }

/* ---------- Creator ---------- */
.df-creator {
    display: grid; grid-template-columns: 1.6fr 1fr; gap: 40px;
    background: var(--soft); border-radius: 14px; padding: 32px;
    margin-bottom: 40px;
}
.df-creator h2 { margin-bottom: 4px; }
.df-creator .role { font-size: 14px; opacity: 0.6; margin-bottom: 14px; }
.df-creator p { opacity: 0.8; max-width: 58ch; }
.df-meta { font-family: 'IBM Plex Mono', monospace; font-size: 13px; align-self: end; }
.df-meta div { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--line); }
.df-meta span:first-child { opacity: 0.55; }

.df-foot { text-align: left; font-size: 14px; opacity: 0.6; padding-bottom: 24px; }

/* ---------- Responsive ---------- */
@media (max-width: 820px) {
    .df-stage { grid-template-columns: 1fr; }
    .df-arrow { transform: rotate(90deg); }
    .df-steps { grid-template-columns: 1fr 1fr; row-gap: 28px; }
    .df-feat { grid-template-columns: 1fr; gap: 6px; }
    .df-creator { grid-template-columns: 1fr; padding: 24px; }
}
@media (prefers-reduced-motion: reduce) {
    .df-table.before .c.bad { animation: none; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


PAGE = (
    '<div class="df">'
    '<section class="df-hero">'
    '<div class="df-brand"><span>🔥</span>DataForge</div>'
    "<h1>Raw data in. Model-ready data out.</h1>"
    "<p>DataForge cleans, transforms and explores your dataset in one place, "
    "so you can spend your time on the model instead of the spreadsheet.</p>"
    "</section>"
    '<section class="df-stage">'
    + table_html(DIRTY, "Before", "Duplicate row, missing age, inconsistent text.", "before")
    + '<div class="df-arrow">&rarr;</div>'
    + table_html(CLEAN, "After", "Duplicate removed, age filled with the median, text standardized.", "after")
    + "</section>"
    "<h2>How it works</h2>"
    f'<ol class="df-steps">{steps_html}</ol>'
    "<h2>What you can do</h2>"
    f'<div class="df-feats">{features_html}</div>'
    '<section class="df-creator">'
    "<div>"
    "<h2>Mahdi Kordian</h2>"
    '<div class="role">AI Engineer</div>'
    "<p>I focus on machine learning, deep learning, generative AI and LLMs, "
    "and I build practical, production-oriented AI systems with Python and modern ML frameworks. "
    "DataForge is one of my projects, made to keep data preparation and exploration simple.</p>"
    "</div>"
    '<div class="df-meta">'
    "<div><span>version</span><span>v1.0.0</span></div>"
    "<div><span>built with</span><span>Python, Streamlit</span></div>"
    "<div><span>project</span><span>DataForge</span></div>"
    "</div>"
    "</section>"
    '<div class="df-foot">Better data, brighter insights.</div>'
    "</div>"
)

st.markdown(PAGE, unsafe_allow_html=True)