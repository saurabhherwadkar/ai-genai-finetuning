"""Main application entry point."""
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from finetuning.api.router import router
from finetuning.config.settings import get_settings

load_dotenv()

def create_app() -> FastAPI:
    app = FastAPI(title="LLM Fine-Tuning API", description="LoRA, QLoRA, DPO fine-tuning framework", version="1.0.0")
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("finetuning.main:app", host=settings.api.host, port=settings.api.port, reload=settings.api.reload)
