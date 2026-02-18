# 5_chatbot_ui.py
# =============================================
# CHATBOT UI - Streamlit se banayenge
# Run command: streamlit run 5_chatbot_ui.py
# =============================================

import streamlit as st
# streamlit = Website banane ki library
# st = shortcut naam

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
import os
import tempfile
# tempfile = temporary files banane ke liye
# PDF upload hone par temporarily save karna

# =============================================
# PAGE SETUP
# =============================================

st.set_page_config(
    page_title="RAG Chatbot",
    # Browser tab mein ye naam dikhega

    page_icon="🤖",
    # Browser tab mein ye icon dikhega

    layout="wide"
    # Wide = Full screen layout use karo
)

# =============================================
# TITLE
# =============================================

st.title("🤖 RAG Document Chatbot")
# st.title = Sabse bada heading

st.markdown("### PDF upload karo aur AI se questions puchho!")
# st.markdown = Formatted text
# ### = Medium heading

st.divider()
# Horizontal line draw karo (visual separator)

# =============================================
# SIDEBAR - Settings
# =============================================
# Sidebar = Left side panel
# Settings ke liye use hota hai

with st.sidebar:
    # "with" = is block ke andar sab
    #          sidebar mein jayega

    st.header("⚙️ Setup")
    # st.header = Medium heading

    # ---- HuggingFace Token ----
    st.subheader("1️⃣ HuggingFace Token")
    hf_token = st.text_input(
        label="Token paste karo:",
        type="password",
        # type=password = dots dikhenge
        # Security ke liye
        placeholder="hf_xxxxxxxxxxxxx"
        # placeholder = hint text
    )

    # ---- PDF Upload ----
    st.subheader("2️⃣ PDF Upload")
    uploaded_file = st.file_uploader(
        label="Apni PDF yahan upload karo",
        type=['pdf']
        # Sirf PDF files allow karo
    )

    # ---- Process Button ----
    st.subheader("3️⃣ Process Karo")
    process_btn = st.button(
        "🚀 Process PDF",
        use_container_width=True
        # Button full width ka hoga
    )

    # ---- Status ----
    st.divider()
    st.subheader("📊 Status")

    if "processed" not in st.session_state:
        # session_state = page reload tak data save
        # "processed" key nahi hai matlab abhi process nahi hua
        st.warning("⚠️ PDF process nahi hua abhi")
    else:
        st.success("✅ PDF ready hai!")
        st.info(f"📄 Chunks: {st.session_state.get('chunk_count', 0)}")

# =============================================
# PDF PROCESSING LOGIC
# =============================================

if process_btn:
    # Button click hua!

    # Check: Token diya?
    if not hf_token:
        st.error("❌ HuggingFace token daalo pehle!")
        st.stop()
           # Check: PDF upload hua?
    if not uploaded_file:
        st.error("❌ PDF upload karo pehle!")
        st.stop()

    # Sab theek hai - process karo!
    with st.spinner("⏳ PDF process ho raha hai... Please wait!"):
        # st.spinner = Loading animation
        # User ko pata chale kuch ho raha hai
        try:
            # Token set karo
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            # PDF temporarily save karo
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.pdf'
                # suffix = file ka extension
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                # getvalue() = uploaded file ka content
                tmp_path = tmp_file.name
                # tmp_path = temporary file ka location
            # PDF load karo
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            # Chunks banao
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)
            # Embeddings banao
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # FAISS database banao
            vectordb = FAISS.from_documents(chunks, embeddings)
            # LLM load karo
            hf_pipe = pipeline(
                task="text-generation",
                model="google/flan-t5-small",
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
            llm = HuggingFacePipeline(pipeline=hf_pipe)
            # RAG Chain banao
            def format_docs(docs):
                return "\n\n".join(
                    doc.page_content for doc in docs
                )
            prompt = PromptTemplate.from_template("""
Answer based on context only.
If unknown, say "I don't know".
Context: {context}
Question: {question}
Answer:""")
            retriever = vectordb.as_retriever(
                search_kwargs={"k": 2}
            )

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            # Session state mein save karo
            # (Page reload tak data rahega)
            st.session_state.rag_chain = rag_chain
            st.session_state.processed = True
            st.session_state.chunk_count = len(chunks)
            st.session_state.vectordb = vectordb
            st.success(f"✅ PDF processed! {len(chunks)} chunks ready!")
            st.balloons()
            # st.balloons = Celebration animation! 🎈
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
# =============================================
# CHAT INTERFACE
# =============================================

st.header("💬 Chat")

# Chat history initialize karo
if "messages" not in st.session_state:
    st.session_state.messages = []
    # messages = list of all chat messages
    # Format: [{"role": "user", "content": "..."}]

# Purani messages display karo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # role = "user" ya "assistant"
        # Alag alag style mein dikhega
        st.write(message["content"])

# New question input
question = st.chat_input(
    "Question type karo... (PDF process karo pehle!)"
)
# st.chat_input = Bottom mein input box

if question:
    # User ne question likha!

    # User message display karo
    with st.chat_message("user"):
        st.write(question)

    # History mein add karo
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Check: PDF process hua?
    if "rag_chain" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("⚠️ Pehle PDF upload karke Process karo!")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Pehle PDF upload karke Process karo!"
        })
    else:
        # Answer generate karo
        with st.chat_message("assistant"):
            with st.spinner("🤔 Soch raha hai..."):

                # RAG chain se answer lo
                answer = st.session_state.rag_chain.invoke(question)

                # Answer display karo
                st.write(answer)

                # Sources bhi dikhaao
                with st.expander("📄 Sources dekho"):
                    # expander = click karne par khulta hai
                    docs = st.session_state.vectordb.similarity_search(
                        question, k=2
                    )
                    for i, doc in enumerate(docs):
                        st.write(f"*Source {i+1}:*")
                        st.write(doc.page_content)
                        st.divider()

        # History mein add karo
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# =============================================
# FOOTER
# =============================================

st.divider()
st.markdown(
    "Made with ❤️ using LangChain + FAISS + Streamlit"
)