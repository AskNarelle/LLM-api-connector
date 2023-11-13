# AskNarelle-Backend

This repository contains the code for a backend Flask server designed as a RESTful API, connecting to the GPT API using the LangChain framework.

The server also uses a locally hosted [ChromaDB](https://www.trychroma.com/) vector database, which requires you to generate your own embeddings prior to running the server.

## Configuration

Ensure that you create a `config` folder containing a `.env` file with your own OpenAI API key. 

## Running the server

Create embeddings using `embedder.py`

Run the server using `server.py`

## API Endpoints

`/getAns`: Endpoint for generating reply using GPT API  
**Method**: POST  
**Request Body Format**: `{"userInput":"Your prompt here"}`