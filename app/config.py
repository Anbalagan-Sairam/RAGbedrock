# app/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator

VECTORSTORE_PATH = "vectorstore"

class AppConfig(BaseSettings):
    aws_region: str = Field(..., env="AWS_REGION")
    bedrock_embed_model: str = Field(..., env="BEDROCK_EMBED_MODEL")
    bedrock_llm_model: str = Field(..., env="BEDROCK_LLM_MODEL")
    vectorstore_path: str = Field(VECTORSTORE_PATH)
    
    model_k: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # -----------------------------
    # Validators
    # -----------------------------
    @validator("vectorstore_path")
    def vectorstore_must_exist(cls, v):
        path = Path(v)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Vectorstore path '{v}' does not exist or is not a directory.")
        if not os.access(v, os.R_OK):
            raise ValueError(f"Vectorstore path '{v}' is not readable.")
        return v

    @validator("aws_region")
    def aws_region_must_be_set(cls, v):
        if not v:
            raise ValueError("AWS_REGION is not set or empty in the environment.")
        return v

    @validator("bedrock_embed_model")
    def embed_model_must_be_set(cls, v):
        if not v:
            raise ValueError("BEDROCK_EMBED_MODEL is not set or empty in the environment.")
        return v

    @validator("bedrock_llm_model")
    def llm_model_must_be_set(cls, v):
        if not v:
            raise ValueError("BEDROCK_LLM_MODEL is not set or empty in the environment.")
        return v