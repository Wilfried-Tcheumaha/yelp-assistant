from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Only the keys this service actually needs at runtime.

    The full Yelp-assistant repo also wires Google + Groq, but this MCP
    server only calls OpenAI (via Superlinked NLQ + the embeddings API).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    openai_api_key: str


config = Config()
