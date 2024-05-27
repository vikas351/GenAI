import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
import chromadb


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


def search_collection(query,collection_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",batch_size =100)
    embedding = list(embedding_model.embed_query(query))
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_or_create_collection(collection_name)
    results =  collection.query(embedding)
    retrieved_docs = results['documents']
    content = ''' '''
    for doc in retrieved_docs['documents']:
        content += '/n'.join(doc)
    return content








def load_vector_store(file_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = f"./data/{os.path.basename(file_name).split('.')[0]}"
    vector_db = Chroma(persist_directory = persist_directory,embedding_function = embedding_model)
    return vector_db.as_retriever()


def process_pdf(pdf_file_path):
    documents =  load_document(pdf_file_path)
    chunked_documents =  chunk_documents(documents)
    retriever =  store_in_vectorstore(pdf_file_path,chunked_documents)
    return retriever
    




