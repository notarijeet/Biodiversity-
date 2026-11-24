import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    # Paths
    data_dir: Path = REPO_DIR / "data"
    models_dir: Path = BACKEND_DIR / "models" / "checkpoints"

    # Text model
    text_model_name: str = "xlm-roberta-base"
    text_model_checkpoint: str = ""

    # Image model
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    # Database
    database_url: str = "sqlite:///./wildlife_cases.db"

    # Hardware / Device
    device: str = ""

    # Optional demo key (not used in core production pipeline)
    gemini_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


def get_device() -> str:
    """Return device string: cuda if available, otherwise cpu. Respects DEVICE env var."""
    forced = os.getenv("DEVICE") or settings.device
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
