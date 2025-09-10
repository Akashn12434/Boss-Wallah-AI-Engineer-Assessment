# Codemonk---Backend-Intern-Assignment-
**Boss Wallah - AI Support Chatbot** is an intelligent virtual assistant designed to help users explore Boss Wallah's course offerings. The chatbot uses advanced AI capabilities to answer course-related questions, provide multilingual responses, suggest career opportunities, and perform web-based searches when necessary.

It delivers contextual, conversational answers in the user’s preferred language. This project combines **RAG (Retrieval-Augmented Generation)** for course content queries with **AI agents** for general knowledge and location-based queries, providing a seamless educational assistant experience.


  ## Features
- **Course Recommendation:** Finds the most relevant courses from Boss Wallah’s database.
- **Multilingual Support:** Detects user language and translates responses automatically.
- **Conversational History:** Maintains chat context for better interaction.
- **Web Search Integration:** Answers general questions using Google search when outside course scope.
- **Places Search:** Provides detailed information about locations, stores, or markets.
- **Predefined Smart Responses:** Supports specialized responses like dairy farming queries in multiple languages.
- **Streamlit UI:** Interactive chat interface with sidebar shortcuts and clear chat functionality.


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

## 🚀 API Endpoints
File/Folder	Description
### 🔐 Authentication
| Method | Endpoint                     | Description                      |
|--------|------------------------------|----------------------------------|
| POST   | /api/v1/auth/register/       | Register a new user              |
| POST   | /api/v1/auth/login/          | Login and get authentication token |
| POST   | /api/v1/auth/logout/         | Logout (requires authentication) |

### 📝 Paragraphs
| Method | Endpoint                          | Description                          |
|--------|-----------------------------------|--------------------------------------|
| POST   | /api/v1/paragraphs/store/         | Store paragraphs (requires authentication) |
| GET    | /api/v1/paragraphs/               | List user's paragraphs (requires authentication) |
| GET    | /api/v1/paragraphs/{id}/          | Get specific paragraph (requires authentication) |
| DELETE | /api/v1/paragraphs/{id}/          | Delete paragraph (requires authentication) |

### 🔍 Search
| Method | Endpoint                          | Description                           |
|--------|-----------------------------------|---------------------------------------|
| GET    | /api/v1/search/?word={word}       | Search paragraphs by word (requires authentication) |

---

# 📁 Project Structure

- **app.py** - Main Streamlit application
- **courses.csv** - Course dataset (not included in repo)
- **.env** - Environment variables (not included in repo)
- **requirements.txt** - Python dependencies
- **chroma_langchain_db/** - Vector database (auto-created on first run)
- **docs/** - Documentation folder
  - **screenshots/** - Application screenshots
    - query1.png
    - query2.png
    - query3.png
    - query4.png
- **venv/** - Python virtual environment (auto-created)
- **README.md** - Project documentation
## 🚀 Quick Start with Docker

## Setup and Installation
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
   
