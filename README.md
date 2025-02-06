# 📚 RAG App with Streamlit, MongoDB & LlamaIndex

🚀 **A Retrieval-Augmented Generation (RAG) app to upload PDFs, index their content, and answer user queries.**  
Built using **Streamlit, MongoDB Atlas Vector Search, and LlamaIndex**.

---

## ✨ Features

✅ **PDF Upload & Processing** – Extracts text from PDFs and creates vector embeddings  
✅ **MongoDB Atlas Vector Store** – Stores document vectors for efficient retrieval  
✅ **Natural Language Search** – Ask questions and retrieve relevant document content  
✅ **Top-3 Source Extraction** – Shows the most relevant document excerpts  

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/rag-app.git
cd rag-app
```


2️⃣ Install Dependencies
Make sure you have Python 3.9+ installed, then run:

```
pip install -r requirements.txt
```

3️⃣ Configure Environment Variables
Create a .env file and add the following:
```
MONGODB_URI="your-mongodb-connection-string"
OPENAI_API_KEY="your-openai-api-key"
```
4️⃣ Run the Application
```
streamlit run app.py
```

🔍 How It Works
1️⃣ Upload a PDF 📄
2️⃣ The app extracts and indexes text using LlamaIndex
3️⃣ Store document embeddings in MongoDB Atlas Vector Search
4️⃣ Ask a question 💡 and get an AI-generated response with relevant sources


🏗️ Tech Stack
- Frontend: 🎨 Streamlit
- Vector Database: 🛢️ MongoDB Atlas Vector Search
- Embeddings & Indexing: 🤖 LlamaIndex + OpenAI
- Backend: ⚙️ Python

📝 License
This project is licensed under the MIT License. See LICENSE for more details.

