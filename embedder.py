import os
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
)

load_dotenv(find_dotenv("./config/.env"))

CONNECTION_STRING = os.getenv("MONGO_URI")
NAMESPACE = "testdb.testcollection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

def CosmosEmbedder():
    global CONNECTION_STRING, NAMESPACE, DB_NAME, COLLECTION_NAME

    all_docs = []

    for filename in os.listdir("./data"):
        file_path = os.path.join("./data", filename)

        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(file_path)

            if file_extension.lower() == '.pdf':
                pdf_parser = PyPDFLoader(file_path)
                pdf_docs = pdf_parser.load()
                all_docs.extend(pdf_docs)

            elif file_extension.lower() == '.docx':
                word_loader = Docx2txtLoader(file_path)
                word_docs = word_loader.load()
                all_docs.extend(word_docs)

    # loader = TextLoader("./data/clean-sc1015.txt")
    # data = loader.load()

    # Document Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(all_docs)

    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]

    vectorstore = AzureCosmosDBVectorSearch.from_documents(
        all_splits,
        OpenAIEmbeddings(),
        collection=collection,
        index_name="AN-testindex",
    )

    num_lists = 100
    dimensions = 1536
    similarity_algorithm = CosmosDBSimilarityType.COS

    vectorstore.create_index(num_lists, dimensions, similarity_algorithm)

    print("Added to CosmosDB")

def checkCosmos():
    global CONNECTION_STRING, NAMESPACE, DB_NAME, COLLECTION_NAME

    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]

    for document in collection.find():
        print(document)

def deleteCosmos():
    global CONNECTION_STRING, NAMESPACE, DB_NAME, COLLECTION_NAME

    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]

        # Delete all documents in the collection
    result = collection.delete_many({})

    # Print the number of deleted documents
    print(f"Deleted {result.deleted_count} documents.")

def fetchCosmos():
    CONNECTION_STRING = os.getenv("MONGO_URI")
    NAMESPACE = "testdb.testcollection"
    openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", chunk_size=1
    )

    vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
        CONNECTION_STRING, NAMESPACE, openai_embeddings, index_name="AN-testindex"
    )

    query = "who are the course instructors?"

    docs = vectorstore.similarity_search(query)
    print(docs[0].page_content)

if __name__ == "__main__":
    #textEmbedder()
    CosmosEmbedder()
    #checkCosmos()
    #deleteCosmos()
    #fetchCosmos()




