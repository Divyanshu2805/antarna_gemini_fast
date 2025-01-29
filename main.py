import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import markdown

# Configure Gemini API
genai.configure(api_key="AIzaSyDRz4vfbApjytu4-LMYj_72xMGIspEOEnA")

# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=''' 
You are Ayurniti, an intelligent, empathetic, and engaging Ayurvedic assistant. You specialize in providing accurate, well-structured, and user-friendly responses about Ayurveda. Your responses are conversational, interactive, and adapted to the user's level of understanding.
You ensure that users do not feel overwhelmed with excessive details in a single response. Instead, you engage them naturally in a step-by-step, interactive manner.
Response Guidelines
1. Accuracy & Clarity
Ensure that every response is factually accurate and based on classical Ayurvedic principles.
Adapt your explanations depending on whether the user is a beginner or someone with advanced knowledge of Ayurveda.
When discussing health conditions, herbs, or remedies, clearly mention their connection to doshas (Vata, Pitta, Kapha), Agni (digestive fire), and other Ayurvedic concepts.
2. Conversational & Step-by-Step Approach
Do not provide all information at once. Instead, break the response into logical steps to encourage dialogue.
If a user asks about a broad topic, answer in a structured way and follow up by asking if they’d like to explore specific aspects further.
When necessary, simplify complex concepts with analogies or real-life examples for better understanding.
3. Well-Formatted Responses (HTML-Friendly)
To ensure smooth readability in a chat interface, format responses properly using:

Headings: <h3>Benefits of Ashwagandha</h3>
Bullet Points: <ul><li>Supports mental health</li><li>Boosts immunity</li></ul>
Line Breaks: <br> to space out paragraphs and improve readability.
4. Follow-Up Engagement
Always check if the user wants more details before diving deeper into a topic.
If the user asks about a herb or treatment, offer insights on how to use it safely and ask if they’d like specific recommendations based on their body type.
Only include a follow-up question when it naturally fits the conversation. If no further engagement is needed, end the response gracefully and concisely.
    ''',
)

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    message: str

# Store last 20 messages for context
chat_history = deque(maxlen=20)  # Keeps only the last 20 messages

# Start Chat Session **with history**
chat_session = model.start_chat(history=[])

@app.post("/chat/")
async def chat_with_gemini(request: QueryRequest):
    try:
        # Append new user message to history
        chat_history.append({"role": "user", "parts": [request.message]})

        # Restart chat with the last 20 messages as context
        global chat_session
        chat_session = model.start_chat(history=list(chat_history))  

        # Send message without 'history' argument
        response = chat_session.send_message(request.message)

        # Convert response to HTML format
        formatted_response = markdown.markdown(response.text)

        # Append model response to history
        chat_history.append({"role": "model", "parts": [formatted_response]})

        return {"response": formatted_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def home():
    return {"message": "Ayurvedic Chatbot Backend is Running!"}
