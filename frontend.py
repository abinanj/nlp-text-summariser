import streamlit as st
import requests

st.set_page_config(page_title="Text Summarizer", page_icon="📰")

# Minimal black & white theme
st.markdown(
	"""
	<style>
	:root { --bg: #000; --fg: #fff; --muted: #bbb; --card:#111; --border:#222; }
	[data-testid="stAppViewContainer"] { background: var(--bg); color: var(--fg); }
	[data-testid="stHeader"] { background: transparent; }
	.stTextArea textarea { background: var(--bg); color: var(--fg); border: 1px solid var(--fg); border-radius: 0; }
	.stButton > button { background: var(--bg); color: var(--fg); border: 1px solid var(--fg); border-radius: 0; }
	.stButton > button:hover { filter: invert(1); color:#000 !important; }
	code, pre { background: var(--card); color: var(--fg); border: 1px solid var(--border); }
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("Text Summarizer")

text = st.text_area("Paste text to summarize", height=240, placeholder="Paste or type long text here...")
col1, col2 = st.columns([1,1])
with col1:
	max_len = st.number_input("Max length", 50, 300, 130)
with col2:
	min_len = st.number_input("Min length", 10, 150, 30)

if st.button("Summarize"):
	if not text.strip():
		st.warning("Please provide some text.")
	else:
		try:
			resp = requests.post("http://localhost:8000/summarize", json={"text": text, "max_length": int(max_len), "min_length": int(min_len)}, timeout=60)
			resp.raise_for_status()
			st.subheader("Summary")
			st.write(resp.json().get("summary", ""))
		except Exception as e:
			st.error(f"Error: {e}")

