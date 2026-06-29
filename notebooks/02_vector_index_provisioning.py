# MAGIC %pip install uv
# MAGIC !uv pip install --system -e ..

from db_rag_app.vector_index import provision_vector_search

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
# Set this to False if running on Community Edition (Free Tier)
IS_PREMIUM_TIER = True

CATALOG = "main"
SCHEMA = "default"
SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.rag_documents"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_documents_index"

VECTOR_SEARCH_ENDPOINT_NAME = "vs_endpoint"
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"
PRIMARY_KEY = "chunk_id"
TEXT_COLUMN = "chunk_text"

if __name__ == "__main__":
    provision_vector_search(
        is_premium_tier=IS_PREMIUM_TIER,
        vector_search_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE_NAME,
        primary_key=PRIMARY_KEY,
        text_column=TEXT_COLUMN,
        embedding_model_endpoint=EMBEDDING_MODEL_ENDPOINT
    )
