# Rag_chatbot
Rag -based document Q&amp;A chatbot using Langchain, Faiss and streamlit
# 🤖 RAG Document Chatbot

An intelligent document question-answering chatbot built using Retrieval Augmented Generation (RAG) architecture.

## 🌟 Features

- PDF document upload and processing
- Semantic search using FAISS vector database
- Local LLM integration (flan-t5)
- Beautiful interactive UI with Streamlit
- Source citation for answers

## 🛠️ Technologies Used

- *LangChain* - RAG pipeline framework
- *FAISS* - Vector database for semantic search
- *Sentence Transformers* - Text embeddings
- *HuggingFace* - Free LLM models
- *Streamlit* - Web UI
- *Python 3.14*

## 📋 Requirements
bash
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
faiss-cpu
pypdf
streamlit
transformers


## 🚀 Installation

1. Clone the repository:
bash
git clone https://github.com/Gaurav3431/rag-chatbot.git
cd rag-chatbot


2. Create virtual environment:
bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux


3. Install dependencies:
bash
pip install -r requirements.txt


## 💻 Usage

1. Get HuggingFace token from [huggingface.co](https://huggingface.co/settings/tokens)

2. Run the chatbot:
bash
streamlit run 5_chatbot_ui.py


3. Open browser at http://localhost:8501

4. Upload PDF, enter token, and start asking questions!

## 📊 How It Works

1. *Document Processing*: PDF is loaded and split into chunks
2. *Embeddings*: Chunks are converted to vector embeddings
3. *Vector Storage*: Embeddings stored in FAISS database
4. *Retrieval*: User query finds similar chunks via semantic search
5. *Generation*: LLM generates answer based on retrieved context

## 🎯 Example Questions

- "What is machine learning?"
- "Explain the types of ML"
- "What are the applications mentioned?"

## 📝 Project Structure

rag-chatbot/
├── test1_basic.py          # Setup verification
├── 2_pdf_loader.py         # PDF loading test
├── 3_embeddings.py         # Embeddings + FAISS test
├── 4_rag_chain.py          # RAG chain test
├── 5_chatbot_ui.py         # Main chatbot UI
├── sample.pdf              # Sample document
└── README.md               # This file


## 🔮 Future Enhancements

- [ ] Multiple PDF support
- [ ] Chat history persistence
- [ ] Better LLM models
- [ ] Deployment on cloud
- [ ] Multi-language support

## 👨‍💻 Author

*Gaurav Kumar*
- GitHub: github.com/Gaurav3431
- LinkedIn: linkedin.com/in/gaurav-kumar-87a00a394
## 📄 License

MIT License

---

Made with ❤️ using LangChain + FAISS + Streamlit


**Commit changes** click karo!
