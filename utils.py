import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()


def load_document(file_path):
    documents =  []
    if os.path.exists(file_path):
        loader =  PyPDFLoader(file_path)
        documents =  loader.load()
        #Removing pages which does not haveany content 
        documnets =  [doc for doc in documents if doc.page_content != '']
    return documents

def chunk_documents(documents):
    splitter =  RecursiveCharacterTextSplitter(chunk_size = 512,chunk_overlap = 100)
    chunked_documents =  splitter.split_documents(documents)
    return chunked_documents

def store_in_vectorstore(file_name,chunked_documents):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",batch_size =100)
    collection_name = f"{os.path.basename(file_name).split('.')[0]}"
    batch_size = 100
    final_embeddings =  []
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]
        final_embeddings.extend(embedding_model.embed_documents([doc.page_content for doc in batch]))
    final_embeddings =  [list(i) for i in final_embeddings]
    metadatas = []
    for doc in chunked_documents:
        metadatas.append(doc.metadata)
    final_docs = []
    for doc in chunked_documents:
        final_docs.append(doc.page_content)
    ids =  [str(uuid.uuid4()) for i in range(0,len(chunked_documents))]
    batch_size = 100
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_or_create_collection(collection_name)
    for i in range(0, len(chunked_documents), batch_size):
        documents = final_docs[i:i + batch_size]
        embeddings = final_embeddings[i:i + batch_size]
        metadata = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        # Debugging: Print embeddings to check their structure
        print(f"Batch {i // batch_size + 1} embeddings: {embeddings}")
        
        # Check if all embeddings are lists
        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise ValueError(f"Expected each embedding to be a list, got {type(embedding)}")
        
        collection.add(ids=batch_ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadata)
    return True


def search_collection(query,collection_name):
    logging.info(f"Connecting to Collection {collection_name}")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",batch_size =100)
    embedding = list(embedding_model.embed_query(query))
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_or_create_collection(collection_name)
    results =  collection.query(embedding)
    logger.info(f"Results: {results}")
    retrieved_docs = results['documents']
    content = ''' '''
    for doc in retrieved_docs:
        content += '/n'.join(doc)
    logger.info(content)
    return content


def process_pdf(pdf_file_path):
    pdf_file_path =  os.path.join('./data',pdf_file_path)
    documents =  load_document(pdf_file_path)
    chunked_documents =  chunk_documents(documents)
    status  =  store_in_vectorstore(pdf_file_path,chunked_documents)
    return status
    

def condense_question(question,history):
    prompt =  """Based on the user conversation history ,if it a followup question phrase a new question,
    else send the question as it is
    Conversation History:
    {history}
    New Question: {question}
    """
    llm =  ChatGoogleGenerativeAI(model="gemini-pro")
    response =  llm.invoke(prompt.format(history = history,question = question))
    return response.content


def chat_with_document(question,collection_name):
    prompt = """You are an expert in answering machine,solely based on the provided question answer user's question
    Question:{question}
    -----
    context:{context}"""
    context =  search_collection(question,collection_name)
    llm =  ChatGoogleGenerativeAI(model="gemini-pro")
    response =  llm.invoke(prompt.format(question =  question, context  = context))
    return response.content




