from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv(find_dotenv("./config/.env"))

app = Flask(__name__)
CORS(app)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

template = """ 
You are AskNarelle, a FAQ (Frequently Asked Questions) chatbot that is designed to answer course-related queries by undergraduate students.
You are to use the provided pieces of context to answer any questions. 
If you do not know the answer, just reply with "Sorry, I'm not sure.", do not try to make up your own answer.
You are also to required to keep the answers as concise as possible.
Always end with "Thanks for using AskNarelle!" at the end of your answer.

{context}
Question: {question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(llm,
                                chain_type='stuff',
                                retriever=vectorstore.as_retriever(),
                                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


@app.route('/getAns', methods=['POST', 'OPTIONS'])
def getAnswer():
    if request.method == 'OPTIONS':
        # Respond to preflight request
        response = jsonify({'status': 'success'})
        response.headers['Access-Control-Allow-Origin'] = '*'  # Adjust as needed
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        return response
    
    try: 
        data = request.get_json()
        prompt = data.get("userInput","") # Default set the input to blank
        result = qa.run(prompt)
        return{"Answer":result}
    except:
        return jsonify({"Status":"Failure --- Error with OpenAI API"})

if __name__ == "__main__":
    app.run(debug=True)