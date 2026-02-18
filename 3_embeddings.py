# 3_embeddings.py
# =============================================
# KYA KARTA HAI YE FILE:
# 1. PDF chunks ko numbers mein convert karta hai
#    (Embeddings = text ka numerical representation)
# 2. FAISS database mein store karta hai
# 3. Search test karta hai
# =============================================

# ---- Libraries Import ----

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader = PDF se text nikalna
# Pehle use kar chuke hain (Day 2)

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Text ko chunks mein todna
# Day 2 mein fix kiya tha import

from langchain_huggingface import HuggingFaceEmbeddings
# HuggingFaceEmbeddings = Free embedding model
# Text ko numbers mein convert karta hai
# HuggingFace ke server se model download hoga

from langchain_community.vectorstores import FAISS
# FAISS = Vector database
# Numbers store karta hai
# Similar numbers dhundta hai (smart search)

import os
# os = Operating System library
# Environment variables set karne ke liye
# Token safely store karne ke liye

print("=" * 50)
print("   DAY 3 - EMBEDDINGS + FAISS")
print("=" * 50)

# ---- STEP 1: HuggingFace Token Set Karo ----

print("\n🔑 Step 1: Token set kar raha hun...")

# IMPORTANT: Apna token yahan paste karo!
HF_TOKEN = "upload your huggingface token"
# ☝️ YE CHANGE KARO - apna actual token paste karo

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
# os.environ = system level variable
# Libraries automatically yahan se token padhti hain
# Direct code mein token likhne se better hai

print("✅ Token set hua!")

# ---- STEP 2: PDF Load + Chunks Banao ----
# (Same as Day 2 - repeat kar rahe hain)

print("\n📄 Step 2: PDF load kar raha hun...")

loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(f"✅ PDF loaded - {len(documents)} pages")

print("\n✂️ Step 3: Chunks bana raha hun...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks ready!")

# ---- STEP 3: Embedding Model Load Karo ----

print("\n🤖 Step 4: Embedding model load kar raha hun...")
print("(Pehli baar 1-2 minutes lagenge - model download hoga)")
print("Please wait...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # all-MiniLM-L6-v2 = Chhota fast model
    # Size: 80MB (8GB RAM ke liye perfect)
    # 384 dimensions ka vector banata hai
    # "sentence-transformers/" = organization name
    # "all-MiniLM-L6-v2" = model name
)

print("✅ Embedding model ready!")

# ---- STEP 4: Test - Ek Text Ka Embedding Dekho ----

print("\n🔢 Step 5: Embedding test kar raha hun...")

test_text = "What is machine learning?"
vector = embeddings.embed_query(test_text)
# embed_query = ek text ko numbers mein convert karo

print(f"Text: '{test_text}'")
print(f"Vector length: {len(vector)}")
# 384 numbers honge

print(f"Pehle 5 numbers: {[round(v, 3) for v in vector[:5]]}")
# round(v, 3) = 3 decimal places tak
# Poore 384 numbers nahi dikha rahe - bahut lamba hoga!

# ---- STEP 5: FAISS Database Banao ----

print("\n💾 Step 6: FAISS database bana raha hun...")
print("(Thoda time lagega - sabhi chunks ka embedding ban raha hai)")

vectordb = FAISS.from_documents(
    documents=chunks,
    # Ye chunks store honge database mein
    
    embedding=embeddings
    # Har chunk ka embedding bhi store hoga
    # Matlab: text + numbers dono store honge
)

print("✅ FAISS database ready!")
print(f"Total vectors stored: {vectordb.index.ntotal}")
# ntotal = kitne vectors stored hain
# Ye number chunks ke barabar hoga

# ---- STEP 6: Database Save Karo ----

print("\n💿 Step 7: Database save kar raha hun...")

vectordb.save_local("faiss_db")
# "faiss_db" = folder ka naam
# D:\ppchat\faiss_db\ mein save hoga
# Computer band karo - data safe rahega!
# Kal dobara load kar sakte ho

print("✅ Database saved in 'faiss_db' folder!")

# ---- STEP 7: Search Test Karo ----

print("\n🔍 Step 8: Search test kar raha hun...")
print("(Ye sabse important step hai!)")

# 3 alag questions test karenge
test_questions = [
    "What is machine learning?",
    "What are types of machine learning?",
    "What are applications of machine learning?"
]

for question in test_questions:
    print(f"\n❓ Question: {question}")
    
    results = vectordb.similarity_search(
        question,
        k=2
        # k=2 = top 2 most similar chunks do
        # k badha sakte ho zyada context ke liye
    )
    # similarity_search kya karta hai:
    # 1. Question ko embedding mein convert karta hai
    # 2. FAISS mein similar embeddings dhundta hai
    # 3. Top k chunks return karta hai
    
    print(f"📄 Top {len(results)} relevant chunks mile:")
    
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"  Page: {result.metadata['page']}")
        print(f"  Content: {result.page_content[:150]}...")
        # Sirf 150 characters dikha rahe hain
    
    print("-" * 40)

# ---- SUMMARY ----

print("\n" + "=" * 50)
print("   SUMMARY")
print("=" * 50)
print(f"✅ PDF Chunks: {len(chunks)}")
print(f"✅ Vectors Stored: {vectordb.index.ntotal}")
print(f"✅ Embedding Dimensions: {len(vector)}")
print(f"✅ Database Saved: faiss_db/")
print("=" * 50)
print("✅ Day 3 Complete!")
print("Next: RAG Chain - AI se jawab lenge! (Day 4)")
print("=" * 50)