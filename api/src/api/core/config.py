from  pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
    openai_api_key: str
    google_api_key: str
    groq_api_key: str

config = Config()