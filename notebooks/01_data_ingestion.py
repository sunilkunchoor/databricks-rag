# MAGIC %pip install uv
# MAGIC !uv pip install --system -e ..

import os
from db_rag_app.ingestion import download_data, process_and_save_to_delta

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
# Set this to False if running on Community Edition (Free Tier)
IS_PREMIUM_TIER = True

CATALOG = "main"
SCHEMA = "default"
VOLUME = "raw_data"
TABLE_NAME = "rag_documents"

if IS_PREMIUM_TIER:
    FILE_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
    FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
else:
    FILE_DIR = f"/dbfs/FileStore/{SCHEMA}/{VOLUME}"
    FULL_TABLE_NAME = f"hive_metastore.default.{TABLE_NAME}"

if not IS_PREMIUM_TIER:
    os.makedirs(FILE_DIR, exist_ok=True)

FILE_NAME = "sample_data.json"
FILE_PATH = f"{FILE_DIR}/{FILE_NAME}"
DATA_URL = "https://jsonplaceholder.typicode.com/posts"

if __name__ == "__main__":
    download_data(data_url=DATA_URL, file_path=FILE_PATH)
    process_and_save_to_delta(file_path=FILE_PATH, full_table_name=FULL_TABLE_NAME, is_premium_tier=IS_PREMIUM_TIER)
