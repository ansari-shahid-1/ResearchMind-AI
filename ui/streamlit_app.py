"""Streamlit front end.

Runs against the FastAPI backend over HTTP rather than importing the services
directly, so the two can be run, deployed or restarted independently.

    uvicorn app.main:app --reload
    streamlit run ui/streamlit_app.py
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st

API = os.environ.get("RESEARCHMIND_API", "http://localhost:8000")
TIMEOUT = httpx.Timeout(300.0, connect=5.0)

st.set_page_config(page_title="ResearchMind AI", page_icon="◫", layout="wide")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def api_get(path: str):
    response = httpx.get(f"{API}{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs):
    response = httpx.post(f"{API}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def backend_status() -> tuple[bool, str]:
    try:
        health = api_get("/health")
    except Exception as exc:
        return False, str(exc)
    ollama = health.get("ollama", {})
    if not ollama.get("ok"):
        return True, "API is up but Ollama is unreachable - generation will fail."
    if not ollama.get("chat_model_available"):
        return True, f"Ollama is up but {health['chat_model']} is not pulled."
    return True, ""


def show_sources(sources: list[dict], citations: list[dict]) -> None:
    cited = {c["marker"]: c for c in citations}
    st.markdown("**Sources**")
    for source in sources:
        marker = next(
            (
                k
                for k, v in cited.items()
                if v["page"] == source["page"] and v["doc_title"] == source["doc_title"]
            ),
            None,
        )
        tag = f"[{marker}] " if marker else ""
        label = f"{tag}{source['doc_title']} - p.{source['page']}  (score {source['score']:.4f})"
        with st.expander(label):
            if source.get("dense_rank") or source.get("sparse_rank"):
                st.caption(
                    f"dense rank {source.get('dense_rank') or '-'} / "
                    f"sparse rank {source.get('sparse_rank') or '-'}"
                )
            st.write(source["text"])


def render_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result.get("citations"):
        st.caption(
            "Citations: "
            + ", ".join(f"[{c['marker']}] {c['doc_title']} p.{c['page']}" for c in result["citations"])
        )
    else:
        st.warning("No citations were produced for this answer.")

    if result.get("steps"):
        with st.expander(f"Agent trace ({len(result['steps'])} steps)"):
            for i, step in enumerate(result["steps"], start=1):
                kind = step.get("type")
                if kind == "tool":
                    st.markdown(
                        f"**{i}. {step['tool']}** `{json.dumps(step.get('args', {}))}` "
                        f"-> {step.get('results', 0)} chunks in {step.get('ms', 0):.0f} ms"
                    )
                    st.code(step.get("observation", "")[:600], language=None)
                elif kind == "seed":
                    st.markdown(f"**{i}. seed retrieval** -> {step.get('results', 0)} chunks")
                else:
                    tools = step.get("tools")
                    label = f" ({', '.join(t for t in tools if t)})" if tools else ""
                    st.markdown(f"**{i}. {kind}**{label}")

    with st.expander(f"Retrieved context ({len(result['sources'])} passages)"):
        show_sources(result["sources"], result.get("citations", []))

    st.caption(f"{result['mode']} mode - {result['latency_ms']:.0f} ms")


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

st.sidebar.title("ResearchMind AI")

ok, warning = backend_status()
if not ok:
    st.sidebar.error("Backend unreachable")
    st.error(
        f"Cannot reach the API at {API}.\n\n"
        "Start it with `uvicorn app.main:app --reload`, or point "
        "`RESEARCHMIND_API` at wherever it is running."
    )
    st.stop()
if warning:
    st.sidebar.warning(warning)
else:
    st.sidebar.success("Backend ready")

st.sidebar.subheader("Add papers")
uploads = st.sidebar.file_uploader(
    "PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
)
if uploads and st.sidebar.button("Ingest", type="primary"):
    for upload in uploads:
        with st.sidebar.status(f"{upload.name}", expanded=True) as status:
            try:
                result = api_post(
                    "/documents",
                    files={"file": (upload.name, upload.getvalue(), "application/pdf")},
                )
                if result["duplicates"]:
                    status.write(f"Already indexed ({result['doc_id']})")
                else:
                    status.write(f"{result['pages']} pages -> {result['chunks']} chunks")
                status.update(label=upload.name, state="complete")
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", exc.response.text)
                status.update(label=f"{upload.name}: {detail}", state="error")
    st.rerun()

st.sidebar.subheader("Corpus")
try:
    documents = api_get("/documents")
except Exception:
    documents = []

if not documents:
    st.sidebar.caption("No papers yet.")
for doc in documents:
    row = st.sidebar.columns([4, 1])
    row[0].markdown(f"**{doc['title']}**  \n<sub>{doc['pages']}p · {doc['chunks']} chunks</sub>",
                    unsafe_allow_html=True)
    if row[1].button("✕", key=f"del-{doc['doc_id']}", help="Remove from the index"):
        httpx.delete(f"{API}/documents/{doc['doc_id']}", timeout=TIMEOUT)
        st.rerun()

st.sidebar.subheader("Answer mode")
mode = st.sidebar.radio(
    "mode",
    options=["agent", "rag"],
    format_func=lambda m: "Agent (tools)" if m == "agent" else "Plain RAG",
    label_visibility="collapsed",
)
top_k = st.sidebar.slider("Passages to retrieve", 2, 12, 6, disabled=(mode == "agent"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])

with ask_tab:
    st.title("Ask across your papers")
    if not documents:
        st.info("Upload a few PDFs in the sidebar to get started.")
    else:
        st.caption("Give me a hard one — the agent will search several times and do its own arithmetic.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["result"])

    question = st.chat_input("e.g. which method has the best F1 on multi-hop questions?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                try:
                    result = api_post(
                        "/query",
                        json={"question": question, "mode": mode, "top_k": top_k},
                    )
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.json().get("detail", exc.response.text)
                    st.error(detail)
                    result = None
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    result = None
            if result:
                render_answer(result)
                st.session_state.history.append({"question": question, "result": result})

with eval_tab:
    st.title("Evaluation")
    results_dir = Path(__file__).resolve().parent.parent / "eval" / "results"
    runs = sorted(results_dir.glob("*.json"), reverse=True) if results_dir.exists() else []

    if not runs:
        st.info("No evaluation runs yet. Run `python -m eval.run` and reload this tab.")
    else:
        chosen = st.selectbox("Run", options=runs, format_func=lambda p: p.stem)
        payload = json.loads(chosen.read_text(encoding="utf-8"))
        meta = payload["meta"]
        st.caption(
            f"{meta['chat_model']} · chunk {meta['chunk_size']}/{meta['chunk_overlap']} · "
            f"top_k {meta['top_k']} · judge {'on' if meta['judge'] else 'off'}"
        )

        summaries = payload["summaries"]
        columns = st.columns(len(summaries))
        for column, (label, summary) in zip(columns, summaries.items(), strict=False):
            with column:
                st.metric(f"{label} - Recall@{meta['k']}", summary.get("recall@k", "-"))
                st.metric("LLM judge", summary.get("judge_score") or "-")
                st.metric("Citation precision", summary.get("citation_precision") or "-")
                st.metric("p50 latency", f"{summary.get('p50_ms', '-')} ms")

        st.dataframe(
            [
                {
                    "id": row["id"],
                    "mode": row["mode"],
                    "type": row["type"],
                    "recall@k": row["recall"],
                    "citation P": row["citation_precision"],
                    "token F1": row["token_f1"],
                    "judge": row.get("judge_score"),
                    "ms": round(row["latency_ms"]),
                }
                for row in payload["rows"]
            ],
            use_container_width=True,
        )

        report = chosen.with_suffix(".md")
        if report.exists():
            with st.expander("Markdown report"):
                st.markdown(report.read_text(encoding="utf-8"))
