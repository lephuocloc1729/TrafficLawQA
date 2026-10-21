import enum
from abc import ABC
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class ProjectBaseSettings(BaseSettings, ABC):
    """Base settings for the project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ProjectSettings(ProjectBaseSettings):
    log_level: LogLevel = LogLevel.INFO

    huggingface_token: str | None = None

    # Vector Store
    qdrant_uri: str = "http://localhost:6333"
    collection_name: str | None = None

    # Embedding Model
    embedding_name: str = "CLIP-GmP-ViT-L-14"
    limit: int = 5

    # LLM Models
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_keys: str | None = None
    llm_model: str = "gemma-3-12b-it"
    llm_temperature: float = 0.0
    llm_max_new_tokens: int = 5000

    # Detection Model
    image_lawdb_detection_model: str = "ImageProcessing"
    image_query_detection_model: str = "Yolo26s"
    sign_model_path: str = str(
        Path(__file__).resolve().parents[1] / "models" / "yolo26s_vtsdb100_best.pt"
    )
    sign_detection_confidence_threshold: float = 0.25
    # Legacy detectors may opt into image retrieval; YOLO26s QA bypasses it.
    enable_sign_retrieval_fallback: bool = False
    enable_symbolic_reasoning: bool = True
    sign_recognition_candidate_limit: int = 10

    # OCR for explicit text/numeric values printed on sign crops.
    traffic_sign_ocr_enabled: bool = True
    traffic_sign_ocr_device: str = "auto"
    traffic_sign_ocr_language: str = "vi"

    # LawDB directory
    lawdb_json_path: str = "data/law_db/vlsp2025_law_new.json"
    lawdb_image_dir: str = "data/law_db/images.fld"
    # Processed LawDB directory
    lawdb_json_preprocessing_path: str | None = None
    lawdb_json_sign_parsing_path: str | None = None
    lawdb_extracted_sign_dir: str | None = None

    # Train directory
    train_json_path: str = "data/train_data/vlsp_2025_train.json"
    train_image_dir: str = "data/train_data/train_images"
    # Processed train directory
    train_extracted_sign_dir: str | None = None
    train_extracted_sign_json_path: str | None = None

    # Public test
    public_test_json_path: str = "data/public_test/vlsp_2025_public_test_task1.json"
    public_test_image_dir: str = "data/public_test/public_test_images"
    public_test_extracted_sign_dir: str | None = None
    public_test_extracted_sign_json_path: str | None = None

    # Private test
    private_test_json_path: str = "data/private_test/vlsp2025_submission_task1.json"
    private_test_image_dir: str = "data/private_test/private_test_images"
    private_test_extracted_sign_dir: str | None = None
    private_test_extracted_sign_json_path: str | None = None


settings = ProjectSettings()
settings.lawdb_extracted_sign_dir = (
    f"data/law_db/signs_{settings.image_lawdb_detection_model.lower()}"
)
settings.lawdb_json_preprocessing_path = Path(settings.lawdb_json_path).with_name(
    f"{Path(settings.lawdb_json_path).stem}_preprocessed.json"
)
settings.lawdb_json_sign_parsing_path = Path(settings.lawdb_json_path).with_name(
    f"{Path(settings.lawdb_json_path).stem}_preprocessed_with_{settings.image_lawdb_detection_model.lower()}.json"
)
settings.collection_name = f"lawdb_processed_by_{settings.image_lawdb_detection_model.lower()}_and_{settings.embedding_name.lower()}"

# Train data
settings.train_extracted_sign_dir = (
    f"data/train_data/train_signs_{settings.image_query_detection_model.lower()}"
)
settings.train_extracted_sign_json_path = Path(settings.train_json_path).with_name(
    f"{Path(settings.train_json_path).stem}_preprocessed_with_{settings.image_query_detection_model.lower()}_and_{settings.llm_model.replace('/', '_')}_and_{settings.embedding_name.lower()}.json"
)

# Public test
settings.public_test_extracted_sign_dir = (
    f"data/public_test/public_test_signs_{settings.image_query_detection_model.lower()}"
)
settings.public_test_extracted_sign_json_path = Path(
    settings.public_test_json_path
).with_name(
    f"{Path(settings.public_test_json_path).stem}_preprocessed_with_{settings.image_query_detection_model.lower()}_and_{settings.llm_model.replace('/', '_')}_and_{settings.embedding_name.lower()}.json"
)

# Private test
settings.private_test_extracted_sign_dir = f"data/private_test/private_test_signs_{settings.image_query_detection_model.lower()}"
settings.private_test_extracted_sign_json_path = Path(
    settings.private_test_json_path
).with_name(
    f"{Path(settings.private_test_json_path).stem}_preprocessed_with_{settings.image_query_detection_model.lower()}_and_{settings.llm_model.replace('/', '_')}_and_{settings.embedding_name.lower()}.json"
)
