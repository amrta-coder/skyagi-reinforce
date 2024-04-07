from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel, BaseSettings, Extra


def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    from skyagi.util import load_json

    # Load settings from JSON config file
    config_dir = Path(Path.home(), ".skyagi")
    config_file = Path(config_dir, "config.json")
    if not config_dir.exists() or not config_file.exists():
        print("[Error] Please config skyagi first by running `skyagi config --help`")
        import sys

        sys.exit(-1)
    return load_json(config_file)


class LLMSettings(BaseModel):
    """
    LLM/ChatModel related settings
    """

    type: str = "chatopenai"

    class Config:
        extra = Extra.allow


class EmbeddingSettings(BaseModel):
    """
    Embedding related settings
    """

    type: str = "openaiembeddings"

    class Config:
        extra = Extra.allow


class ModelSettings(BaseModel):
    """
    Model related settings
    """

    type: str = ""
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    class Config:
        extra = Extra.allow


class Settings(BaseSettings):
    """
    Root settings
    """

    name: str = "default"
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "skyagi_"
        env_file_encoding = "utf-8"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )


# ---------------------------------------------------------------------------- #
#                             Preset configurations                            #
# ---------------------------------------------------------------------------- #
class OpenAIGPT4TurboSettings(ModelSettings):
    type = "gpt-4-0125-preview"
    llm = LLMSettings(type="chatopenai", model="gpt-4-0125-preview", max_tokens=1500, request_timeout=300)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT4Settings(ModelSettings):
    type = "openai-gpt-4"
    llm = LLMSettings(type="chatopenai", model="gpt-4", max_tokens=1500)
    # llm = LLMSettings(type="chatopenai", model="gpt-4-0613", max_tokens=1500, temperature=0.5, request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT3_5TurboSettings(ModelSettings):
    type = "gpt-3.5-turbo-0125"
    llm = LLMSettings(type="chatopenai", model="gpt-3.5-turbo-0125", max_tokens=16385)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAITextEmbedding3LargeSettings(ModelSettings):
    type = "openai-gpt-3.5-text-davinci-003"
    llm = LLMSettings(type="openai", model_name="text-embedding-3-large", max_tokens=1500)
    embedding = EmbeddingSettings(type="openaiembeddings")


# ------------------------- Model settings registry ------------------------ #
model_setting_type_to_cls_dict: Dict[str, Type[ModelSettings]] = {
    "openai-gpt-4-turbo": OpenAIGPT4TurboSettings,
    "openai-gpt-4": OpenAIGPT4Settings,
    "openai-gpt-3.5-turbo": OpenAIGPT3_5TurboSettings,
    "openai-text-embedding-3-large": OpenAITextEmbedding3LargeSettings,
}


def load_model_setting(type: str) -> ModelSettings:
    if type not in model_setting_type_to_cls_dict:
        raise ValueError(f"Loading {type} setting not supported")

    cls = model_setting_type_to_cls_dict[type]
    return cls()


def get_all_model_settings() -> List[str]:
    """Get all supported Embeddings"""
    return list(model_setting_type_to_cls_dict.keys())
