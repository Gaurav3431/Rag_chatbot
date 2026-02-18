print("=" * 50)
print("   RAG CHATBOT - SETUP CHECK")
print("=" * 50)

print("\n✓ Test 1: Python is working!")
print("\n✓ Test 2: Checking libraries...")

try:
    import langchain
    print("  ✓ langchain imported")
except:
    print("  ✗ langchain NOT found")

try:
    import faiss
    print("  ✓ faiss imported")
except:
    print("  ✗ faiss NOT found")

try:
    from sentence_transformers import SentenceTransformer
    print("  ✓ sentence-transformers imported")
except:
    print("  ✗ sentence-transformers NOT found")

try:
    import pypdf
    print("  ✓ pypdf imported")
except:
    print("  ✗ pypdf NOT found")

try:
    import streamlit
    print("  ✓ streamlit imported")
except:
    print("  ✗ streamlit NOT found")

print("\n" + "=" * 50)
print("   SETUP COMPLETE!")
print("=" * 50)