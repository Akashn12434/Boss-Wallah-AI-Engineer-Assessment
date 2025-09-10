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
- **Backend:** Django  with Django REST Framework
- **Database:** PostgreSQL 17
- **Authentication:** Token-based authentication
- **Documentation:** drf-spectacular (Swagger/OpenAPI)
- **Containerization:** Docker & Docker Compose

## Architecture
The API follows these design patterns:

- **Models:** Custom User, Paragraph, WordIndex with proper relationships
- **Views:** Function-based and class-based views with proper permissions
- **Serializers:** Comprehensive validation and data transformation
- **Utils:** Reusable text processing functions
- **Tests:** Unit tests for all major functionality

## 🚀 API Endpoints

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

## 🚀 Quick Start with Docker

## Setup and Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Akashn12434/Codemonk---Backend-Intern-Assignment-.git
    cd Codemonk---Backend-Intern-Assignment-
    ```

2. **Install dependencies**:
   - Install the required Python packages:   
    ```bash
    pip install -r requirements.txt
    ```
3. **Running the Application Using Docker Setup**:
   - Build, run with Docker Compose: 
    ```bash
    docker-compose up --build
    ```
  
1. **Run Migrations and create superuser**:
   - Open a new terminal window and run:
    ```bash
    docker-compose exec web python manage.py makemigrations
    docker-compose exec web python manage.py migrate
    docker-compose exec web python manage.py createsuperuser
    ```

4.**Access it On running**:
   - Swagger UI:
   ```bash
   http://localhost:8000/api/docs/
   ```
   - Admin Interface:
   ```bash
   http://localhost:8000/admin/
   ```
