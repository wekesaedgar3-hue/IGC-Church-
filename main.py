# bible_ai_with_memory.py
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import requests, os

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize LLM
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.4)

# ✅ Initialize Vector Memory (ChromaDB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="bible_memory",
    embedding_function=embeddings
)
print("🧩 Vector database loaded — ready to store and search Bible reflections.\n")

# ✅ Structured Output Schema
class BibleResponse(BaseModel):
    reference: str = Field(..., description="Bible verse or passage being discussed")
    reasoning_type: str = Field(..., description="Reasoning strategy used: chain-of-thought, react, or self-ask")
    summary: str = Field(..., description="Short summary or explanation of the passage")
    reflection: Optional[str] = Field(None, description="Reflective or devotional insight")

print("💬 Agentic Bible AI (RAG + Multi-Turn Memory) ready!")
print("Type 'search love' or 'search faith' to explore past reflections.")
print("Type 'exit' to quit.\n")

# 🔹 Choose tone
style = input("Choose tone (friendly / scholarly / devotional): ").strip().lower()
if style == "friendly":
    tone = "Use warm, conversational, and simple language."
elif style == "scholarly":
    tone = "Use an academic tone referencing theology."
elif style == "devotional":
    tone = "Use reflective and encouraging language emphasizing faith and growth."
else:
    tone = "Use a clear, balanced tone suitable for all readers."

# ✅ Helper: Semantic Search
def search_similar(query, k=3):
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        print("No similar reflections found yet.")
        return ""
    print("\n🔎 Similar Past Reflections:")
    retrieved_texts = []
    for doc, score in results:
        print(f"→ Relevance: {score:.3f}")
        print(doc.page_content)
        print("---")
        retrieved_texts.append(doc.page_content)
    return "\n".join(retrieved_texts)

# ✅ Multi-turn conversation memory
conversation_history = [
    SystemMessage(content=f"""
You are a compassionate and wise AI Bible assistant.
Always ground your answers in Scripture, using retrieved context and verse text.
Maintain continuity across turns in the conversation.
Tone: {tone}
""")
]

# ✅ Reasoning strategy selector
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

# ✅ Explanation generator with grounding
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

Generate a unified response (under 250 words):
- Ground your answer in Scripture and past reflections.
- Maintain conversational flow.
- Return structured output with reference, reasoning_type, summary, and reflection.
"""
)
bible_explanation_chain = bible_explanation_prompt | llm.with_structured_output(BibleResponse)

# ✅ Main loop (RAG + Multi-turn memory)
while True:
    user_input = input("\n📖 Enter a Bible verse, chapter, question, or 'search <term>': ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Session ended. God bless you!")
        break

    if user_input.lower().startswith("search "):
        query = user_input.replace("search ", "")
        search_similar(query)
        continue

    verse_text = user_input
    if any(char.isdigit() for char in user_input):
        try:
            res = requests.get(f"https://bible-api.com/{user_input.replace(' ', '%20')}")
            data = res.json()
            if "text" in data:
                verse_text = data["text"].strip()
                print(f"\n📜 {data['reference']}\n{verse_text}")
            else:
                verse_text = "(Verse not found)"
        except Exception as e:
            verse_text = f"(Error fetching verse: {e})"

    reasoning_type = reasoning_selector_chain.invoke({"user_input": user_input}).strip().lower()

    retrieved = search_similar(user_input, k=3)
    conversation_text = "\n".join([msg.content for msg in conversation_history[-4:]])  # Last few turns

    result = bible_explanation_chain.invoke({
        "tone": tone,
        "reasoning_type": reasoning_type,
        "verse_text": verse_text,
        "retrieved": retrieved,
        "conversation": conversation_text,
        "user_input": user_input
    })

    print(f"\n🧠 Explanation ({reasoning_type} reasoning):\n")
    explanation = result.summary
    if result.reflection:
        explanation += " " + result.reflection
    print(explanation)

    combined_text = f"{result.reference}\n{result.summary}\n{result.reflection or ''}"
    vectorstore.add_documents([
        Document(
            page_content=combined_text,
            metadata={"reference": result.reference, "reasoning_type": result.reasoning_type}
        )
    ])
    vectorstore.persist()

    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=explanation))
    print("💾 Explanation stored in vector memory & conversation context updated.")
