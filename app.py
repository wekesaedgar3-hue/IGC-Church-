# app.py — Agentic Bible AI (Streamlit + RAG + Multi-turn Memory)
import streamlit as st
from dotenv import load_dotenv
import os, requests, shutil
from typing import Optional
from pydantic import BaseModel, Field

# LangChain / Groq / Vector Memory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("🚨 Missing GROQ_API_KEY in your .env file.")
    st.stop()

# --- Initialize LLM and Vector Memory ---
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.4)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="bible_memory", embedding_function=embeddings)

# --- Structured Output Schema ---
class BibleResponse(BaseModel):
    reference: str = Field(..., description="Bible verse or passage being discussed")
    reasoning_type: str = Field(..., description="Reasoning strategy used")
    summary: str = Field(..., description="Short summary or explanation of the passage")
    reflection: Optional[str] = Field(None, description="Reflective or devotional insight")

# --- Page setup ---
st.set_page_config(page_title="Agentic Bible AI", page_icon="📖", layout="centered")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #f0f7ff, #ffffff);
        color: #0d1b2a;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom right, #eaf4ff, #ffffff);
        border-radius: 12px;
        padding: 1rem 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#003366;'>📖 Agentic Bible AI Assistant</h1>", unsafe_allow_html=True)
st.write("Ask about any Bible verse or topic. The assistant recalls past reflections and continues reasoning across turns.")
st.markdown("---")

# --- Sidebar tone selection ---
st.sidebar.header("✨ Choose Summary Style")
style = st.sidebar.selectbox(
    "Select your preferred tone:",
    ["friendly", "scholarly", "devotional", "neutral"],
    index=0
)
tone_map = {
    "friendly": "Use warm, conversational, and simple language.",
    "scholarly": "Use an academic tone referencing theology.",
    "devotional": "Use reflective and encouraging language emphasizing faith and growth.",
    "neutral": "Use a clear, balanced tone suitable for all readers."
}
tone = tone_map.get(style, tone_map["neutral"])

# --- Helper: Search & Memory Management ---
def search_similar(query, k=3):
    """Retrieve top-k similar past reflections."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return "No similar reflections found yet."
        output = "### 🔎 Similar Past Reflections:\n"
        for doc, score in results:
            output += f"- **Relevance {score:.3f}:** {doc.page_content}\n\n"
        return output
    except Exception as e:
        return f"Error searching memory: {e}"

def get_all_reflections():
    """List all stored reflections."""
    try:
        docs = vectorstore.get()["documents"]
        return docs if docs else []
    except Exception:
        return []

def clear_memory():
    """Completely clear stored vector memory."""
    try:
        shutil.rmtree("bible_memory")
        st.success("🧹 Memory cleared successfully!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error clearing memory: {e}")

# --- Sidebar Memory Panel ---
st.sidebar.markdown("## 🧠 Memory Panel")
memories = get_all_reflections()
if memories:
    st.sidebar.write(f"**Stored Reflections:** {len(memories)}")
    with st.sidebar.expander("View All Reflections"):
        for idx, m in enumerate(memories, start=1):
            st.sidebar.markdown(f"**{idx}.** {m[:200]}{'...' if len(m) > 200 else ''}")
else:
    st.sidebar.info("No stored reflections yet.")

if st.sidebar.button("🧹 Clear All Memory"):
    clear_memory()

# --- Initialize conversation memory ---
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        SystemMessage(content=f"You are a compassionate AI Bible assistant. Maintain continuity and ground answers in Scripture. Tone: {tone}")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Reasoning selector ---
reasoning_selector_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
Given the user input: "{user_input}"
Choose the best reasoning strategy:
1. chain-of-thought → for summarizing or contextual explanation.
2. react → for interpretive or moral analysis.
3. self-ask → for conceptual or philosophical reflection.

Respond with only one: 'chain-of-thought', 'react', or 'self-ask'.
"""
)
reasoning_selector_chain = reasoning_selector_prompt | llm | StrOutputParser()

# --- Explanation generator ---
bible_explanation_prompt = PromptTemplate(
    input_variables=["tone", "reasoning_type", "verse_text", "retrieved", "conversation", "user_input"],
    template="""
You are an AI Bible reasoning assistant.
Tone: {tone}
Reasoning Strategy: {reasoning_type}

Conversation so far:
{conversation}

Relevant Past Reflections:
{retrieved}

Current Passage or Question:
{verse_text}

User Input:
{user_input}

Generate a unified explanation (under 250-500 words depending on the complexity of the question or verse) grounded in Scripture and conversation context.
Return structured output with:
- reference
- reasoning_type
- summary
- reflection
"""
)
bible_explanation_chain = bible_explanation_prompt | llm.with_structured_output(BibleResponse)

# --- Chat input ---
user_input = st.chat_input("Enter a Bible verse or question ...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle search queries
    if user_input.lower().startswith("search "):
        query = user_input.replace("search ", "")
        results = search_similar(query)
        with st.chat_message("assistant"):
            st.markdown(results)
        st.session_state.messages.append({"role": "assistant", "content": results})

    else:
        # Fetch verse text if possible
        verse_text = user_input
        try:
            res = requests.get(f"https://bible-api.com/{user_input.replace(' ', '%20')}")
            data = res.json()
            if "text" in data:
                verse_text = data["text"].strip()
                reference = data.get("reference", user_input)
                verse_display = f"**📜 {reference}**\n\n{verse_text}"
            else:
                verse_display = f"(Could not find '{user_input}' — continuing with AI interpretation.)"
        except Exception as e:
            verse_display = f"(Error fetching verse: {e})"

        # Select reasoning strategy
        reasoning_type = reasoning_selector_chain.invoke({"user_input": user_input}).strip().lower()

        # Retrieve memory + recent conversation
        retrieved = search_similar(user_input, k=3)
        conversation_text = "\n".join(
            [msg.content for msg in st.session_state.conversation[-4:] if hasattr(msg, "content")]
        )

        # Generate structured Bible response
        result = bible_explanation_chain.invoke({
            "tone": tone,
            "reasoning_type": reasoning_type,
            "verse_text": verse_text,
            "retrieved": retrieved,
            "conversation": conversation_text,
            "user_input": user_input
        })

        # Format output
        explanation = f"{verse_display}\n\n---\n\n**🧠 Explanation ({reasoning_type} reasoning):**\n{result.summary}"
        if result.reflection:
            explanation += f" {result.reflection}"

        with st.chat_message("assistant"):
            st.markdown(explanation)
        st.session_state.messages.append({"role": "assistant", "content": explanation})

        # Save reflection to vector memory
        combined_text = f"{result.reference}\n{result.summary}\n{result.reflection or ''}"
        vectorstore.add_documents([
            Document(page_content=combined_text, metadata={"reference": result.reference, "reasoning_type": result.reasoning_type})
        ])
        vectorstore.persist()

        # Update conversation memory
        st.session_state.conversation.append(HumanMessage(content=user_input))
        st.session_state.conversation.append(AIMessage(content=explanation))

st.markdown("---")
st.caption("💡 Built with LangChain + Groq + Chroma + Streamlit | Agentic Bible AI © 2025")
