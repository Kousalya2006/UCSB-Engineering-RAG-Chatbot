
import os
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Load environment variables from the project .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class Config:
    """Configuration for UCSB RAG Assistant."""

    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Model Configuration
    GEMINI_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "models/gemini-embedding-001"

    # Path Configuration
    CHROMADB_PATH = os.path.join(PROJECT_ROOT, "embeddings")
    DATA_PATH = os.path.join(PROJECT_ROOT, "src", "data")
    SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")
    STYLES_PATH = os.path.join(PROJECT_ROOT, "styles")

    # ChromaDB Configuration
    COLLECTION_NAME = "ucsb_engineering_docs"

    # RAG Configuration
    MAX_CONTEXT_DOCS = 5
    SIMILARITY_THRESHOLD = 0.7
    MAX_TOKENS = 1000
    DEFAULT_RETRIEVAL_COUNT = 5

    # UI Configuration
    PAGE_TITLE = "UCSB College of Engineering Assistant"
    PAGE_ICON = "🛠️"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"

    # CSS Configuration
    CSS_FILE = "app.css"

    # Response Generation Configuration
    TEMPERATURE = 0.1
    TOP_P = 0.8
    TOP_K = 40

    # Error Messages
    INITIALIZATION_ERROR_MSG = "Failed to initialize the system"
    API_KEY_ERROR_MSG = (
        "Google API Key not found. Please check your .env file."
    )

    @classmethod
    def validate_config(cls):
        """Validate the required configuration."""
        if not cls.GOOGLE_API_KEY:
            return False, cls.API_KEY_ERROR_MSG

        if not os.path.isdir(cls.CHROMADB_PATH):
            return False, (
                f"ChromaDB path not found: {cls.CHROMADB_PATH}"
            )

        if not os.path.isdir(cls.DATA_PATH):
            return False, f"Data path not found: {cls.DATA_PATH}"

        return True, "Configuration valid"

    @classmethod
    def get_css_path(cls):
        """Return the full path to the CSS file."""
        return os.path.join(cls.STYLES_PATH, cls.CSS_FILE)

    @classmethod
    def get_script_path(cls, script_name):
        """Return the full path to a script."""
        return os.path.join(cls.SCRIPTS_PATH, script_name)


def load_config():
    """Load and validate configuration."""
    config = Config()
    is_valid, error_msg = config.validate_config()

    if not is_valid:
        raise ValueError(f"Configuration error: {error_msg}")

    return config
