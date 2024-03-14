# az webapp up --name NarallelBackendTest --resource-group AskNarelle-Sandbox --runtime PYTHON:3.11
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
#load_dotenv(find_dotenv("./config/.env"))

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
template = """ You are AskNarelle, a FAQ (Frequently Asked Questions) chatbot that is designed to answer course-related queries by undergraduate students. You are to use the provided pieces of context to answer any questions. If you do not know the answer, just reply with "Sorry, I'm not sure.", do not try to make up your own answer. You are also to required to keep the answers as concise as possible. Always end with "Thanks for using AskNarelle!" at the end of your answer. {context} Question: {question} Helpful Answer: """
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

@app.get('/')
async def home():
    return "Hello World! Testing CICD with Azure."

@app.post('/getAns')
async def getAnswer(request: Request):
    try:
        data = await request.json()
        prompt = data.get("userInput", "")  # Default set the input to blank
        result = qa.run(prompt)
        return {"Answer": result}
    except Exception as e:
        error_message = f"Failure --- Error with OpenAI API: {str(e)}"
        return {"Status": error_message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")