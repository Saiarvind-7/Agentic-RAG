"""Streamlit front end.  Run:  streamlit run ui/streamlit_app.py"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")
st.title("Agentic RAG with hybrid retrieval")
st.caption("Ask questions about your documents. The agent searches, checks what it found, and checks its own answer.")


def call(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with: uvicorn app.api:app --reload")
        return None
    if response.status_code != 200:
        try:
            error = response.json()
        except requests.exceptions.JSONDecodeError:
            error = response.text.strip() or "The API returned an empty, non-JSON response."
        if isinstance(error, dict):
            error = error.get("detail", response.text.strip() or str(error))
        st.error(f"API error ({response.status_code}): {error}")
        return None
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        body = response.text.strip() or "empty response"
        st.error(f"API returned invalid JSON (HTTP {response.status_code}): {body}")
        return None


with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add .txt, .md or .pdf files", accept_multiple_files=True, type=["txt", "md", "pdf"])
    if uploads and st.button("Upload"):
        files = [("files", (f.name, f.getvalue())) for f in uploads]
        result = call("POST", "/upload", files=files)
        if result:
            st.success(f"Saved {len(result['saved'])} file(s). Now rebuild the index.")
    if st.button("Rebuild index"):
        with st.spinner("Indexing..."):
            result = call("POST", "/ingest")
        if result:
            st.success(f"Indexed {result['documents']} documents ({result['chunks']} chunks).")

    st.header("Settings")
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"])
    top_k = st.slider("Chunks to use", 1, 10, 5)

question = st.text_input("Your question", placeholder="e.g. How many days of annual leave do I get?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        data = call("POST", "/ask", json={"question": question, "top_k": top_k, "mode": mode})
    if data:
        st.subheader("Answer")
        st.write(data["answer"])

        check = data.get("self_eval")
        if check:
            label = "passed" if check["passed"] else "below threshold"
            st.metric("Self-check score", f"{check['score']:.2f}", label)
            if check["unsupported_claims"]:
                st.warning("Claims the checker could not find in the sources:\n\n- " + "\n- ".join(check["unsupported_claims"]))

        st.subheader("Sources")
        for source in data["sources"]:
            with st.expander(f"[{source['n']}] {source['source']}  ({', '.join(source['found_by']) or 'n/a'})"):
                st.write(source["text"])

        with st.expander("Agent trace"):
            st.json(data["trace"])
