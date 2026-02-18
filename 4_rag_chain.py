# 4_rag_chain.py - FINAL VERSION
# Local model use kar rahe hain - no API issues!

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import os

print("=" * 50)
print("   DAY 4 - RAG CHAIN (LOCAL MODEL)")
print("=" * 50)

# ---- Token ----
HF_TOKEN = "apana huggingface token use karo "
# ☝️ Apna token paste karo!
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ---- Step 1: PDF Load ----
print("\n📄 Step 1: PDF load kar raha hun...")
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(f"✅ {len(documents)} pages loaded!")

# ---- Step 2: Chunks ----
print("\n✂️ Step 2: Chunks bana raha hun...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks ready!")

# ---- Step 3: Embeddings + FAISS ----
print("\n🤖 Step 3: Embeddings bana raha hun...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb = FAISS.from_documents(chunks, embeddings)
print("✅ FAISS ready!")

# ---- Step 4: Local LLM Pipeline ----
print("\n🧠 Step 4: Local LLM load kar raha hun...")
print("(Model pehli baar download hoga - 2-3 mins wait karo!)")

# Local pipeline banao
# Ye model directly tumhare laptop pe chalega
# Koi API call nahi hogi!
hf_pipeline = pipeline(
    task="text-generation",
    # text2text = input text do, output text lo
    # Question answering ke liye perfect

    model="google/flan-t5-small",
    # flan-t5-small = Chhota fast model
    # Size: 80MB (8GB RAM perfect)
    # Locally chalega - no internet needed after download!

    max_new_tokens=200,
    # Maximum 200 tokens ka answer

    temperature=0.3,
    # Low = accurate answers

    do_sample=True
    # do_sample = varied responses
)

# HuggingFacePipeline = Local pipeline ko
# LangChain ke saath use karne ka tarika
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("✅ Local LLM ready!")

# ---- Step 5: Retriever ----
print("\n🔍 Step 5: Retriever set kar raha hun...")
retriever = vectordb.as_retriever(
    search_kwargs={"k": 2}
)
print("✅ Retriever ready!")

# ---- Step 6: Prompt ----
print("\n📝 Step 6: Prompt bana raha hun...")
prompt = PromptTemplate.from_template("""Answer the question based on the context below.
If you don't know, say "I don't know".

Context: {context}

Question: {question}

Answer:""")
print("✅ Prompt ready!")

# ---- Step 7: RAG Chain ----
print("\n⛓️ Step 7: RAG chain bana raha hun...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG Chain ready!")

# ---- Step 8: Questions! ----
print("\n" + "=" * 50)
print("   AI SE QUESTIONS PUCHH RAHE HAIN!")
print("=" * 50)

questions = [
    "What is machine learning?",
    "What are types of machine learning?",
    "What are applications of machine learning?"
]

for question in questions:
    print(f"\n❓ Question: {question}")
    print("⏳ Soch raha hai...")

    answer = rag_chain.invoke(question)
    print(f"🤖 Answer: {answer}")
    print("-" * 40)

print("\n" + "=" * 50)
print("✅ DAY 4 COMPLETE!")
print("Pehla AI Answer mil gaya!")
print("Next: Chatbot UI! (Day 5)")
print("=" * 50)