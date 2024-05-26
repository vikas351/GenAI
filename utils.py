from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

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
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = f{./data/{os.path.basename(file_name).split('.')[0]}}
    vector_db = Chroma(persist_directory = persist_directory,embedding = embedding_model)
    vector_db.from_documents(chunk_documents)
    return vector_db.as_retriever()

def load_vector_store(file_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = f{./data/{os.path.basename(file_name).split('.')[0]}}
    vector_db = Chroma(persist_directory = persist_directory,embedding = embedding_model)
    return vector_db.as_retriever()


    




