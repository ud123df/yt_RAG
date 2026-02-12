

#################################################################################################

import streamlit as st
import os
import re
from yt_dlp import YoutubeDL

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title(" Chat with YouTube Video")


import os
os.environ["GROQ_API_KEY"] ="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"

# 🔐 API Key (use env var or sidebar input)
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = st.sidebar.text_input(
#         "Enter Groq API Key", type="password"
#     )

# if not os.environ.get("GROQ_API_KEY"):
#     st.warning("⚠️ Please enter your Groq API key to continue.")

# ---------------- FUNCTIONS ---------------- #

def download_subtitles(video_url):
    ydl_opts = {
    "skip_download": True,          # do not download video
    "writesubtitles": True,         # write subtitles
    "writeautomaticsub": True,      # write auto-generated subtitles
    "subtitleslangs": ["en"],       # English only
    "subtitlesformat": "vtt",       # output format
    "outtmpl": "yt_text.%(ext)s",    # filename template
}

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

print("✅ Auto-generated English subtitles downloaded")


def vtt_to_text(path):
    text = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = re.sub(r"<[^>]+>", "", line).strip()
            if line and not line[0].isdigit() and "WEBVTT" not in line:
                text.append(line)
    return " ".join(text)


@st.cache_resource(show_spinner=False)
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceBgeEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# ---------------- UI ---------------- #

video_url = st.text_input("📎 Enter YouTube Video URL")

if st.button("📥 Process Video"):
    if not video_url:
        st.warning("Please enter a YouTube URL.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("Groq API key is required.")
    else:
        with st.spinner("Downloading subtitles..."):
            download_subtitles(video_url)

        vtt_file = "yt_text.en.vtt"
        if not os.path.exists(vtt_file):
            st.error("❌ English subtitles not found for this video.")
        else:
            with st.spinner("Processing transcript & building embeddings..."):
                text = vtt_to_text(vtt_file)
                os.remove(vtt_file)

                vector_store = build_vector_store(text)
                st.session_state.retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                st.session_state.transcript = text

            st.success("✅ Video processed! You can now ask questions.")
            st.text_area(
                "📄 Transcript Preview",
                st.session_state.transcript[:2000],
                height=200
            )

# ---------------- Q&A Section ---------------- #

if "retriever" in st.session_state:
    st.subheader("💬 Ask a question about the video")

    question = st.text_input("Your question", key="question_input")

    if st.button("🤖 Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            retriever = st.session_state.retriever
            docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in docs)

            prompt = PromptTemplate(
                template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
                input_variables=["context", "question"],
            )

            final_prompt = prompt.invoke({
                "context": context_text,
                "question": question,
            })

            llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.1,
            )

            with st.spinner("Thinking..."):
                answer = llm.invoke(final_prompt)

            st.success("✅ Answer")
            st.write(answer.content)


