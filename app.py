import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. Check API Key - warn if not set but don't fail immediately
# In AWS App Runner, this will be injected by AWS Secrets Manager
if "OPENAI_API_KEY" not in os.environ:
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not set. Chat functionality will fail.")
else:
    print("✅ OPENAI_API_KEY is set")

# --- Lazy loading: Initialize RAG components on first use ---

rag_chain = None
fallback_chain = None

def get_rag_chain():
    global rag_chain, fallback_chain
    if rag_chain is None:
        print("Loading RAG model and vector store...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        retriever = vectorstore.as_retriever()
        
        # RAG Prompt template
        template = """Use the following pieces of context to answer the question.
If you don't know the answer, say I don't know.

Context: {context}

Question: {question}

Helpful Answer: """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # LLM model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # RAG Chain using LCEL
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Fallback chain for direct ChatGPT when RAG doesn't know
        fallback_prompt = ChatPromptTemplate.from_template("Answer the following question directly:\n\nQuestion: {question}\n\nAnswer:")
        fallback_chain = fallback_prompt | llm | StrOutputParser()

        print("✅ Mommy RAG Application is ready.")
    return rag_chain, fallback_chain

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Mommy Application is live!", "version": "v2"}

@app.post("/chat")
def chat(query: Query):
    try:
        # Lazy load RAG chain on first use
        rag_chain, fallback_chain = get_rag_chain()
        print(f"Question: {query.question}")
        
        # Try RAG first
        answer = rag_chain.invoke(query.question)
        print(f"Initial RAG answer: {answer}")
        
        # Check if answer indicates "I don't know" - fallback to ChatGPT
        answer_lower = answer.lower().strip()
        dont_know_phrases = [
            "i don't know",
            "i do not know",
            "don't know",
            "i don't have",
            "i do not have",
            "i'm sorry, i don't know",
            "i don't have information",
            "i cannot answer",
            "i can't answer",
            "i'm sorry",
            "sorry, i don't",
            "unable to answer",
            "i'm not able to"
            "i'm not sure"
        ]
        
        # Check if answer contains any "don't know" phrases
        should_fallback = any(phrase in answer_lower for phrase in dont_know_phrases)
        print(f"Should fallback: {should_fallback}")
        
        if should_fallback:
            print("RAG didn't know, falling back to ChatGPT...")
            answer = fallback_chain.invoke(query.question)
            print(f"ChatGPT fallback answer: {answer}")
        
        return {"answer": f"Helpful Answer: V1 {answer}"}
    except Exception as e:
        # Return error message
        return {"error": str(e)}, 500
