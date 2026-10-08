import html
import streamlit as st
from core import APP_NAME, PAGES, VERSION, clear_all, go, init_state

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MahdiKordian/DataForge",
        "Report a bug": "https://github.com/MahdiKordian/DataForge/issues",
        "About": f"# 🔥 {APP_NAME}\n\nAutomated Data Preparation for Machine Learning.\n\nBuilt with Python and Streamlit.",
    },
)

NAV = [
    ("landing", "Home", ":material/home:"),
    ("dataset", "Dataset", ":material/dataset:"),
    ("dashboard", "Dashboard", ":material/dashboard:"),
    ("prepare", "Prepare", ":material/tune:"),
    ("analysis", "Analysis", ":material/analytics:"),
    ("export", "Export", ":material/file_download:"),
    ("report", "Report", ":material/summarize:"),
    ("about", "About", ":material/auto_awesome:"),
]

pages = {key: st.Page(PAGES[key], title=title, icon=icon, default=(key == "landing"))
         for key, title, icon in NAV}
pg = st.navigation(list(pages.values()), position="hidden")

init_state()

st.markdown("<style>.block-container{max-width:1100px}</style>", unsafe_allow_html=True)


def sidebar_menu() -> None:
    ss = st.session_state
    with st.sidebar:
        st.title(f"🔥 {APP_NAME}")
        st.badge(VERSION, icon=":material/new_releases:")
        for p in pages.values():
            st.page_link(p)
        st.divider()

        if ss.df is None:
            st.caption("No dataset loaded yet.")
            return

        target = f" · target: {html.escape(str(ss.target))}" if ss.target else ""
        st.markdown(
            f"<div style='font-size:12px;opacity:.6;letter-spacing:.04em'>CURRENT DATASET</div>"
            f"<div style='font-weight:600'>{html.escape(str(ss.df_name))}</div>"
            f"<div style='font-size:13px;opacity:.7'>{len(ss.df):,} rows · {ss.df.shape[1]} columns{target}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.popover("Remove dataset", icon=":material/delete:", width="stretch"):
            st.write("This clears the dataset, its log and every prepared version.")
            if st.button("Yes, remove it", type="primary", key="sidebar_remove"):
                clear_all()
                go("landing")


sidebar_menu()
pg.run()