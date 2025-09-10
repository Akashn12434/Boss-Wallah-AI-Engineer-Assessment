# Codemonk---Backend-Intern-Assignment-
**Boss Wallah - AI Support Chatbot** is an intelligent virtual assistant designed to help users explore Boss Wallah's course offerings. The chatbot uses advanced AI capabilities to answer course-related questions, provide multilingual responses, suggest career opportunities, and perform web-based searches when necessary.

It delivers contextual, conversational answers in the user’s preferred language. This project combines **RAG (Retrieval-Augmented Generation)** for course content queries with **AI agents** for general knowledge and location-based queries, providing a seamless educational assistant experience.


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
 (keywords matched)            (keywords matched)            (fallback to agent)
       │                              │                              │
 ┌─────▼─────┐                 ┌──────▼─────┐                ┌───────▼─────┐
 │  RAG Flow │                 │ Serper API │                │ ReAct Agent  │
 │ (VectorDB │                 │   (Places) │                │ (Gemini +    │
 │ + Gemini) │                 └────────────┘                │ Web Tools)   │
 └─────┬─────┘                                              └───────┬─────┘
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

## Technical Implementation:

 - Custom User model with UUID primary keys
 - Paragraph and WordIndex models with relationships
 - RESTful API design with proper HTTP methods
 - Input validation and error handling
 - Swagger/OpenAPI documentation
 - Docker and Docker Compose setup
 - Comprehensive test coverage

## Tech Stack
- **Framework:** Streamlit for web interface
- **LLM:** Google Gemini 2.0 Flash for natural language processing
- **Vector Database:** ChromaDB for course information retrieval
- **Embeddings:** HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Search APIs:** Google Serper for web and place search
- **Agent Framework:** LangGraph for tool-using AI agent
- **Backend:** Python

## Architecture
The API follows these design patterns:

- **Models:** Custom User, Paragraph, WordIndex with proper relationships
- **Views:** Function-based and class-based views with proper permissions
- **Serializers:** Comprehensive validation and data transformation
- **Utils:** Reusable text processing functions
- **Tests:** Unit tests for all major functionality

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
| docs                  |  screenshots                                     | 

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
   
