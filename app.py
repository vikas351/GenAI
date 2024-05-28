from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_pdf,condense_question,chat_with_document
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Global dictionary to store session chat history
session_history: Dict[str, List[str]] = {}


def get_history(session_id):
    session_details =  session_history.get(session_id,{})
    if 'history' in session_details:
        history =  session_details['history']
        history_string =  ".\n".join(history)
        return history_string
    return ''

def get_collection_name(session_id):
    session_details =  session_history.get(session_id,{})
    if 'file_name' in session_details:
        collection_name =  session_details['file_name'].split('.')[0]
        return collection_name
    



class ConversationPayload(BaseModel):
    session_id: str
    user_message: str

class UploadPdf(BaseModel):
    session_id: str
    file_name: str

@app.post("/chat")
async def chat(conversation: ConversationPayload):
    session_id = conversation.session_id
    user_message = conversation.user_message
    history = get_history(session_id)
    rephrased_question =  condense_question(user_message,history)
    collection_name =  get_collection_name(session_id)
    logger.info(f"collection name {collection_name}")
    assistant_message =  chat_with_document(rephrased_question,collection_name)
    # Append the user message to the session history
    if session_id not in session_history:
        session_history[session_id] = {'history':[f"user_message: {user_message},\n system_message :{assistant_message}"]}
    else:
        session_history[session_id]['history'].append(f"user_message: {user_message}, \n system_message :{assistant_message}")
    
    response = {
        'response': assistant_message,
        'session_id' :session_id
    }
    return response

@app.post("/upload")
async def upload(upload: UploadPdf):
    session_id = upload.session_id
    file_name = upload.file_name
    status =  process_pdf(file_name)
    # Add file upload event to the session history
    if session_id not in session_history:
        session_history[session_id] = {'file_name':file_name,'history' :[]}
    
    response = {
        'status': status,
        'session_id': session_id
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
