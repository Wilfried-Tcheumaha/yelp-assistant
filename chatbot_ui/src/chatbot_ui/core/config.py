from  pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    API_URL: str = "http://api:8000"
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
    
config = Config()