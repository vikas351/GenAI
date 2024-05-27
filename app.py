from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_pdf
from typing import Dict, List

app = FastAPI()

# Global dictionary to store session chat history
session_history: Dict[str, List[str]] = {}

class ConversationPayload(BaseModel):
    sesson_id: str
    user_message: str

class UploadPdf(BaseModel):
    sesson_id: str
    file_name: str

@app.post("/chat")
async def chat(conversation: ConversationPayload):
    session_id = conversation.sesson_id
    user_message = conversation.user_message

    # Append the user message to the session history
    if session_id not in session_history:
        session_history[session_id] = {'history':[f"user_message: {user_message}"]}
    else:
        session_history[session_id]['history'].append(f"user_message: {user_message}")
    response = {
        'response': "hello",
        'session_history': session_history[session_id]
    }
    return response

@app.post("/upload")
async def upload(upload: UploadPdf):
    session_id = upload.sesson_id
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
