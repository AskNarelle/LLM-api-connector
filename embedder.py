from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv(find_dotenv("./config/.env"))

def textEmbedder():
    # Document Loading 

    loader = TextLoader("./data/clean-sc1015.txt")
    data = loader.load()

    # Document Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)

    # Storing into VectorDB (ChromaDB)
    persist_directory = "chroma_db"
    vectorstore = Chroma.from_documents(documents=all_splits,embedding=OpenAIEmbeddings(), persist_directory=persist_directory)

    vectorstore.persist()
    
if __name__ == "__main__":
    textEmbedder()




