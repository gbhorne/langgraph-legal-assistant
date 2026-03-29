import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PROJECT_ID:          str = os.getenv("GCP_PROJECT_ID",             "legal-adk")
    REGION:              str = os.getenv("GCP_REGION",                 "us-central1")
    SEARCH_LOCATION:     str = os.getenv("VERTEX_SEARCH_LOCATION",     "us")
    SEARCH_DATASTORE_ID: str = os.getenv("VERTEX_SEARCH_DATASTORE_ID", "legal-search-shared_1774804330209_gcs_store")
    SEARCH_ENGINE_ID:    str = os.getenv("SEARCH_ENGINE_ID",           "legal-search-app_1774806682231")
    SEARCH_SERVING_CONFIG: str = (
        "projects/1073947050575/locations/us"
        "/collections/default_collection"
        "/engines/legal-search-app_1774806682231"
        "/servingConfigs/default_serving_config"
    )
    GEMINI_MODEL:        str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

config = Config()
