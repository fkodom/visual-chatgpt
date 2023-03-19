import base64
import io
import uuid
from typing import Optional

from fastapi import Cookie, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string."""
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_image(string: str) -> Image:
    """Convert a base64 string to a PIL image."""
    image_buffer = io.BytesIO(base64.b64decode(string))
    return Image.open(image_buffer)


class StorageConnector:
    def __init__(self):
        image = Image.new("RGB", (100, 100))
        self.history = [
            {"text": "Hello!", "role": "assistant"},
            {"text": "How are you?", "role": "assistant"},
            {"image": image_to_base64(image), "role": "system"},
        ]

    def get_chat_history(self, chat_id: str):
        # Placeholder function for storage connector logic
        return self.history

    def add_message(self, chat_id: str, message: dict):
        # Placeholder function for storage connector logic
        self.history.append(message)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
storage = StorageConnector()


class Chatbot:
    @staticmethod
    def predict(text: str, image: Optional[str] = None):
        # Placeholder function for chatbot prediction logic
        messages = [{"text": f"You said: {text}"}]
        if image:
            messages.append({"image": image})  # Return the same image for demo purposes
        return messages


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request, chat_id: Optional[str] = Cookie(None)):
    if chat_id is None:
        chat_id = str(uuid.uuid4())
    chat_history = storage.get_chat_history(chat_id)
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "chat_history": chat_history, "chat_id": chat_id},
    )


class PredictRequest(BaseModel):
    text: str
    image: Optional[str] = None


@app.post("/predict/")
async def get_prediction(request: PredictRequest):
    messages = Chatbot.predict(request.text, request.image)
    return {"messages": messages}
