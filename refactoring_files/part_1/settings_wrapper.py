from pydantic_settings import BaseSettings

class SettingsWrapper(BaseSettings):
    # parameters from the .env variables with missing default values
    PROXY: dict = {"http": "", "https": ""}
    OPENAI_API_KEY : str = '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'
    ANTHROPIC_API_KEY: str = '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'
    REPLICATE_API_TOKEN: str = ''
    OPENROUTER_API_KEY: str = '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'
    GOOGLE_API_KEY: str = ''
    DEEPSEEK_API_KEY: str = ''
    XAI_API_KEY: str = ''

    class Config:
        env_file = '.env' # default location, can be overridden
        env_file_encoding = "utf-8"
