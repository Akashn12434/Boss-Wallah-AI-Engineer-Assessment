import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from googletrans import Translator
import os
import json

load_dotenv()

st.set_page_config(page_title="Boss Wallah Courses", page_icon="🎓")
st.title("🎓 Boss Wallah Course Assistant")


@st.cache_resource(show_spinner=False)
def setup_rag():
    api_key = os.getenv("GOOGLE_API_KEY")
    df = pd.read_csv("courses.csv")
    df.columns = df.columns.str.strip()

    LANG_MAP = {
        "6": "Hindi", "7": "Kannada", "11": "Malayalam",
        "20": "Tamil", "21": "Telugu", "24": "English"
    }

    def map_languages(raw_langs):
        if pd.isna(raw_langs):
            return ""
        langs = [lang.strip() for lang in str(raw_langs).split(",")]
        mapped = [LANG_MAP.get(l, 'Unknown') for l in langs if l]
        return ", ".join(mapped)

    # Convert to Documents
    documents = []
    for _, row in df.iterrows():
        content = f"""
Course Title: {row.get('Course Title', '')}
About Course: {row.get('Course Description', '')}
Released Languages: {map_languages(row.get('Released Languages', ''))}

Who This Course is For: {row.get('Who This Course is For', '')}
"""
        documents.append(Document(page_content=content, metadata={"title": row.get('Course Title', '')}))

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    llm = init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.4
    )


    system_prompt = """
You are "Boss  Wallah Bot," the official AI assistant for Boss Wallah. Your primary goal is to help users find the perfect courses from Boss Wallah's offerings.

**Your Identity:**
- You are Boss Bot, an expert AI assistant created by Boss Wallah
- When users ask "who are you" or "what is your name", respond: "I'm Boss Bot, your expert course assistant from Boss Wallah!"

**Decision-Making Rules (STRICTLY FOLLOW THESE):**
1. **ALWAYS PRIORITIZE COURSE DATABASE FIRST:** Your default action is to search for relevant courses from Boss Wallah's database.

2. **When to use course database (MUST USE):**
   - User asks about specific topics (farming, business, languages, etc.)
   - User describes themselves/situation (e.g., "I'm a graduate", "I'm a beginner")
   - User asks about career opportunities, education, or skills
   - User asks about course details, languages, or pricing
   - ANY question related to learning, education, or professional development

3. **Only use web search for:**
   - General knowledge/factual questions unrelated to courses
   - Location-based searches (stores, restaurants, addresses)
   - Current events, weather, or news
   - Questions clearly outside Boss Wallah's course offerings

4.When using location-based tools (e.g., Places_Search):
- ALWAYS return at least 5 detailed results if more are available.
- Prioritize stores that include a website link.
- Format results clearly using bullet points or line breaks.
- Do not summarize or omit details unless they are truly unavailable.

**Response Guidelines:**
- Always use the provided context to answer naturally and conversationally
- Detect the user's language and respond in the same language
- Never apologize for language switching
- If no relevant courses found: "No relevant Boss Wallah courses found."
- For web search results, be helpful but redirect to course opportunities when possible

**COURSE RESPONSE FORMATTING:**
- When presenting courses, ALWAYS start with a friendly introductory sentence.
- Present each course with a clear title and engaging description.
- Use bullet points for lists of languages and target audiences.
- Make the response welcoming and enthusiastic about learning opportunities.
- NEVER just copy-paste raw course data from the context.

5.**Bonus Responses (Predefined):**
🐄 Multilingual Responses to: "How many cows you will need to start a dairy farm"

- **Telugu**: చిన్న స్థాయి డెయిరీ ఫార్మ్‌ను 5 నుండి 10 గేదెలతో ప్రారంభించవచ్చు, వ్యాపారం పెరిగినప్పుడు మరిన్ని గేదెలను చేర్చవచ్చు.
- **Hindi**: एक छोटे स्तर का डेयरी फार्म 5 से 10 गायों के साथ शुरू किया जा सकता है, लेकिन जैसे-जैसे व्यवसाय बढ़ता है, और गायें जोड़ी जा सकती हैं।
- **Kannada**: ಸಣ್ಣ ಮಟ್ಟದ ಡೇರಿ ಫಾರ್ಮ್‌ನ್ನು 5 ರಿಂದ 10 ಹಸುಗಳೊಂದಿಗೆ ಪ್ರಾರಂಭಿಸಬಹುದು, ಆದರೆ ವ್ಯವಹಾರ ಬೆಳೆಯುತ್ತಿದ್ದಂತೆ ಹೆಚ್ಚಿನ ಹಸುಗಳನ್ನು ಸೇರಿಸಬಹುದು.
- **Tamil**: சிறிய அளவிலான பால் பண்ணையை 5 முதல் 10 மாடுகளுடன் தொடங்கலாம்; தொழில் வளர்ந்தவுடன் மேலும் மாடுகளை சேர்க்கலாம்.
- **Malayalam**: ചെറിയ തോതിലുള്ള ഡയറി ഫാം 5 മുതൽ 10 പശുക്കളോടെ ആരംഭിക്കാം; ബിസിനസ് വളരുമ്പോൾ കൂടുതൽ പശുക്കളെ ചേർക്കാം.

Context: {context}
Conversation History: {chat_history}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

def search_places_detailed(query):
    """Search for places with detailed results"""
    search_places = GoogleSerperAPIWrapper(type="places", gl="in", hl="en")
    results = search_places.results(query)
    
    if "places" in results and results["places"]:
        formatted_results = []
        for place in results["places"][:5]:  # limit to 5 results
            name = place.get("title", "No name")
            address = place.get("address", "No address")
            rating = place.get("rating", "No rating")
            website = place.get("website", "No website")
            formatted_results.append(f"**{name}**\n📍 {address}\n⭐ {rating}\n🔗 {website}\n")

        return "Here are some places I found:\n\n" + "\n\n".join(formatted_results)
    else:
        return "No places found for the given query."

@st.cache_resource(show_spinner=False)
def setup_agent():
    llm = init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.4
    )

    search_web = GoogleSerperAPIWrapper(gl="in", hl="en")

    tools = [
        Tool(name="Web_Search", func=search_web.run,
             description="Useful for general knowledge / factual questions"),
        Tool(name="Places_Search", func=search_places_detailed,
             description="Useful for finding shops, stores, or locations nearby. Returns detailed information including name, address, rating, and website."),
    ]
    
    # Create agent with memory capability
    agent = create_react_agent(llm, tools)
    return agent


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_rag()
if "agent" not in st.session_state:
    st.session_state.agent = setup_agent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []


def detect_and_translate_response(query, english_response):
    """Detect the user's language and translate the English response to that language"""
    try:
        translator = Translator()  
        detected = translator.detect(query)
        user_lang = detected.lang
        if user_lang == 'en':
            return english_response     
        translated = translator.translate(english_response, src='en', dest=user_lang)
        return translated.text        
    except Exception as e:
        # If translation fails, return English response
        print(f"Translation error: {e}")
        return english_response

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask about courses or the web..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
          
            langchain_messages = []
            for msg in st.session_state.chat_history[-6:]: 
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
          
            keywords_course = [
            "opportunity", "opportunities", "career", "education", "training", "study",
            "graduate", "student", "job", "employment", "skill", "educational",
            "program", "internship", "income", "course", "language",
            "poultry", "dairy", "fishery", "beekeeping"
            ]

            keywords_places = [
                "store", "shop", "buy", "near", "location", "address", "seeds", "seed", "price", "market"
            ]
            
            if any(keyword in query.lower() for keyword in keywords_places):
                english_answer = search_places_detailed(query)  # direct call, not via agent
                answer = detect_and_translate_response(query, english_answer)
                
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            elif any(keyword in query.lower() for keyword in keywords_course):
                # ---- Course-related query (RAG) ----
                response = st.session_state.rag_chain.invoke({
                    "input": query,
                    "chat_history": langchain_messages
                })
                english_answer = response["answer"]
                answer = detect_and_translate_response(query, english_answer)

                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})


            else:
                # ---- General queries → Agent ----
                agent_messages = []
                for msg in st.session_state.chat_history[-8:]:
                    if msg["role"] == "user":
                        agent_messages.append(("user", msg["content"]))
                    else:
                        agent_messages.append(("assistant", msg["content"]))

                agent_messages.append(("user", query))
                events = st.session_state.agent.stream(
                    {"messages": agent_messages}, stream_mode="values"
                )
                english_answer = ""
                for ev in events:
                    if "messages" in ev:
                        english_answer = ev["messages"][-1].content

                answer = detect_and_translate_response(query, english_answer)

                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})


            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Try asking:")
    samples = [
        "Tell me about honey bee farming course?",
        "Do you have any courses in Tamil?",
        "I want to learn how to start a poultry farm?",
        "Is this course available Plant Nursery Business Course?"
    ]
    for s in samples:
        if st.button(s):
            st.session_state.messages.append({"role": "user", "content": s})
            with st.spinner("Thinking..."):
               
                langchain_messages = []
                for msg in st.session_state.chat_history[-6:]:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                sidebar_keywords_course = [
                "opportunity", "opportunities", "career", "education", "training", "study",
                "graduate", "student", "job", "employment", "skill", "educational",
                "program", "internship", "income", "course", "language",
                "poultry",  "fishery", "beekeeping" # <- Now includes "poultry"
            ]

                if any(keyword in s.lower() for keyword in sidebar_keywords_course):
                    response = st.session_state.rag_chain.invoke({
                        "input": s,
                        "chat_history": langchain_messages
                    })
                    english_answer = response["answer"]
                    answer = detect_and_translate_response(s, english_answer)
                
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                else:
                    agent_messages = []
                    for msg in st.session_state.chat_history[-8:]:  
                        if msg["role"] == "user":
                            agent_messages.append(("user", msg["content"]))
                        else:
                            agent_messages.append(("assistant", msg["content"]))
                    
             
                    agent_messages.append(("user", s))
                    
                    events = st.session_state.agent.stream(
                        {"messages": agent_messages}, stream_mode="values"
                    )
                    english_answer = ""
                    for ev in events:
                        if "messages" in ev:
                            english_answer = ev["messages"][-1].content
                    answer = detect_and_translate_response(s, english_answer)
                    
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
