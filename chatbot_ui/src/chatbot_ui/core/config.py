from  pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    # In-Docker URL the Streamlit process uses to talk to the FastAPI service.
    API_URL: str = "http://api:8000"

    # Browser-facing URL used when we render `<img src="...">` tags. The user's
    # browser cannot resolve the docker network hostname `api`, so for static
    # assets like photos we need a publicly reachable URL.
    PUBLIC_API_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
    
config = Config()