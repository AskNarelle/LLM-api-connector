import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv("./config/.env"))

app = Flask(__name__)
CORS(app)

CONNECTION_STRING = os.getenv("MONGO_URI")
NAMESPACE = "testdb.testcollection"
vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
    CONNECTION_STRING, NAMESPACE, OpenAIEmbeddings(), index_name="AN-testindex"
)

#vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

template = """ 
You are AskNarelle, a FAQ (Frequently Asked Questions) chatbot that is designed to answer course-related queries by undergraduate students.
You are to use the provided piece(s) of context to answer any question. 
If you do not know the answer or there is no context provided, just reply with "Sorry, I'm not sure.", do not try to make up your own answer.
You are also to required to keep the answers as concise as possible, but do not leave out any important details.

Context: {context}

Chat History (May or may not be empty): {history}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

chat = ChatOpenAI(temperature=0)

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
        history = data.get("chatHistory", "") # Default set to blank

        if len(history) >= 4:
            history[:] = history[-4:]

        docs = vectorstore.similarity_search(prompt, k=3)

        if docs == []:
            vector_result = "Not sure of the answer"
        else:
            temp = []
            for i in range(len(docs)):
                temp.append([docs[i].page_content])
            vector_result = str(temp)
            # vector_result = docs[0].page_content
            print(vector_result)

        reply = chat(
            chat_prompt.format_prompt(
                context=vector_result, history=history, text=prompt
            ).to_messages()
        )

        return {"Answer":reply.content}
    except:
        return jsonify({"Status":"Failure --- Error with OpenAI API"})

if __name__ == "__main__":
    app.run(debug=True)