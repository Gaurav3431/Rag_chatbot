# 2_pdf_loader.py
# =============================================
# KYA KARTA HAI YE FILE:
# PDF se text nikalti hai
# Text ko chhote chunks mein todti hai
# Taaki AI easily padh sake
# =============================================

# ---- PART 1: Libraries Import ----

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader = PDF padhne wala tool
# langchain_community = extra tools ka collection
# Ye tool PDF binary data ko readable text mein convert karta hai

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter = smart text cutter
# "Recursive" = paragraphs → sentences → words order mein kata hai
# Intelligent cutting - beech mein sentence nahi katega

print("=" * 50)
print("   DAY 2 - PDF LOADER TEST")
print("=" * 50)

# ---- PART 2: PDF Load Karo ----

print("\n📄 Step 1: PDF load kar raha hun...")

# PDF ka path batao
pdf_path = "sample.pdf"
# Ye file D:\ppchat\ mein honi chahiye
# Tumne jo PDF banai thi wahi

loader = PyPDFLoader(pdf_path)
# loader = PDF reader ready kiya
# Abhi tak PDF nahi padhi
# Sirf reader ready hua

documents = loader.load()
# AB PDF padhi!
# documents = list of pages
# Har page ek alag item hai list mein

print(f"✅ PDF load hua!")
print(f"📊 Total pages: {len(documents)}")
# len() = count karo kitne pages hain

# ---- PART 3: Pages Ka Content Dekho ----

print("\n📖 Step 2: Pages ka content dekh raha hun...")

for i, page in enumerate(documents):
    # enumerate = page number bhi do saath mein
    # i = page number (0, 1, 2...)
    # page = page ka content
    
    print(f"\n--- Page {i+1} ---")
    # i+1 kyunki counting 0 se shuru hoti hai
    # But humans 1 se count karte hain
    
    print(f"Characters: {len(page.page_content)}")
    # page.page_content = us page ka poora text
    # len() = kitne characters hain
    
    print(f"Content Preview:")
    print(page.page_content[:200])
    # [:200] = sirf pehle 200 characters dikhaao
    # Poora text bahut lamba hoga screen pe

# ---- PART 4: Text Ko Chunks Mein Todo ----

print("\n✂️ Step 3: Text ko chunks mein tod raha hun...")

text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=300,
    # Har chunk maximum 300 characters ka hoga
    # Kyun 300?
    # - Chhota enough = AI easily process kare
    # - Bada enough = Meaning samajh aaye
    # - 8GB RAM ke liye safe
    
    chunk_overlap=50,
    # Consecutive chunks mein 50 characters same honge
    # Kyun overlap?
    # Chunk 1: "...Machine learning is powerful"
    # Chunk 2: "powerful tool used in..."
    # Overlap se sentence toot ke context nahi jaata!
    
    length_function=len,
    # Length kaise measure karein?
    # len() = characters count karta hai
    
    separators=["\n\n", "\n", " ", ""]
    # Kahan se katein priority order mein:
    # 1. "\n\n" = Double newline (paragraph break) - pehle try karo
    # 2. "\n" = Single newline (line break)
    # 3. " " = Space (word boundary)
    # 4. "" = Character (last resort)
    # Smart cutting - meaningful boundaries pe!
)

chunks = text_splitter.split_documents(documents)
# split_documents = documents list lo, chunks list do
# Saari pages ka text ek saath chunk hoga

print(f"✅ Chunking complete!")
print(f"📊 Total chunks banaye: {len(chunks)}")

# ---- PART 5: Chunks Dekho ----

print("\n🔍 Step 4: Chunks ka preview...")
print(f"Pehle 3 chunks dikhata hun:\n")

for i, chunk in enumerate(chunks[:3]):
    # chunks[:3] = sirf pehle 3 chunks
    # Poore chunks bahut zyada honge screen pe
    
    print(f"--- Chunk {i+1} ---")
    print(f"Size: {len(chunk.page_content)} characters")
    # Chunk kitna bada hai
    
    print(f"Page: {chunk.metadata['page']}")
    # metadata = extra information
    # 'page' = ye chunk kis page se aaya
    
    print(f"Content:\n{chunk.page_content}")
    # Poora chunk content dikhaao
    print()

# ---- PART 6: Summary ----

print("=" * 50)
print("   SUMMARY")
print("=" * 50)
print(f"PDF Pages: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
# sum() = sab chunks ke characters jodo
# // = integer division (decimal nahi chahiye)
# Average = total / count
print("=" * 50)
print("✅ PDF Loading Complete!")
print("Next: Embeddings banayenge (Day 3)")
print("=" * 50)