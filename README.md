# Boss Wallah Chatbot – Smart Support for Learners
 An intelligent virtual assistant chatbot designed to help users explore Boss Wallah's course offerings. The chatbot uses advanced AI capabilities to answer course-related questions, provide multilingual responses, suggest career opportunities, and perform web-based searches when necessary.

It delivers contextual, conversational answers in the user’s preferred language. This project combines **RAG (Retrieval-Augmented Generation)** for course content queries with **AI agents** for general knowledge and location-based queries, providing a seamless educational assistant experience.

## Features
- 📚 Course Information Retrieval: Answers questions about Boss Wallah courses from the database
- 📍 Location-Based Search: Finds nearby stores and shops using Google Serper API
- 🤖 General Queries (ReAct Agent) – Gemini + web tools for knowledge search.
- 🔄 Smart Query Routing – Classifies and directs queries to the right pipeline.
- 🌍 Multilingual Support: Detects user's language and responds in the same language (supports like Hindi, Kannada, Malayalam, Tamil, Telugu, and English)
- ⚡ Conversational Memory: Maintains context across conversations for more natural interactions


  ## 📊 System Architecture

```text
                        ┌──────────────────────────┐
                        │        User Query        │
                        └─────────────┬────────────┘
                                      │
                              Language Detection
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
 Course-related Query        Place/location Query           General Knowledge Query
       │                              │                              │
 ┌─────▼─────┐                 ┌──────▼─────┐                ┌───────▼─────┐
 │  RAG Flow │                 │ Serper API │                │ ReAct Agent │
 │ (VectorDB │                 │   (Places) │                │ (Gemini +   │
 │ + Gemini) │                 └────────────┘                │ Web Tools)  │
 └─────┬─────┘                                               └───────┬─────┘
       │                                                              │
   Response in English -----------------------------------------------┘
                                      │
                             Translation Layer
                         (auto-translate back to user language)
                                      │
                            ┌─────────▼─────────┐
                            │     Final Answer  │
                            └───────────────────┘
```



## Tech Stack
- **Framework:** Streamlit for web interface
- **LLM:** Google Gemini 2.0 Flash for natural language processing
- **Vector Database:** ChromaDB for course information retrieval
- **Embeddings:** HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Search APIs:** Google Serper for web and place search
- **Agent Framework:** LangGraph for tool-using AI agent
- **Backend:** Python



## 📁 Project Structure

| File/Folder           | Description                                      | 
|-----------------------|--------------------------------------------------|
| app.py                | Main Streamlit application                       | 
| courses.csv           |The dataset containing all course information.    |
| .env                  | Stores the GOOGLE_API_KEY and SERPER_API_KEY     | 
| requirements.txt      | Python dependencies                              | 
| chroma_langchain_db/  | Vector database                                  | 
| venv                  |  Python virtual environment                      | 
| README.md             |  Project documentation                           | 
| docs                  |  screenshots of the Project                      | 

---

## 🚀 Setup and Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Akashn12434/Boss-Wallah-AI-Engineer-Assessment.git
    cd Boss-Wallah-AI-Engineer-Assessment
    ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   
3. **Install dependencies**:
   - Install the required Python packages:   
    ```bash
    pip install -r requirements.txt
    ```
  
4. **Run the application:**:
    ```bash
    streamlit run app.py
    ```

4.**Access it On running**:
   - Open your browser and go to
   ```bash
   http://localhost:8501
   ```
   
