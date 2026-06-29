# MAGIC %pip install uv
# MAGIC !uv pip install --system -e ..

import mlflow
from db_rag_app.rag_chain import create_rag_chain, test_and_log_model

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
# Set this to False if running on Community Edition (Free Tier)
IS_PREMIUM_TIER = True

CATALOG = "main"
SCHEMA = "default"

if IS_PREMIUM_TIER:
    INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_documents_index"
    VECTOR_SEARCH_ENDPOINT_NAME = "vs_endpoint"
    EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"
    LLM_ENDPOINT = "databricks-meta-llama-3-70b-instruct"
    MODEL_NAME = f"{CATALOG}.{SCHEMA}.rag_chatbot_model"
    FULL_TABLE_NAME = None
    mlflow.set_registry_uri("databricks-uc")
else:
    INDEX_NAME = None
    VECTOR_SEARCH_ENDPOINT_NAME = None
    EMBEDDING_MODEL_ENDPOINT = None
    LLM_ENDPOINT = None
    FULL_TABLE_NAME = f"hive_metastore.default.rag_documents"
    MODEL_NAME = "rag_chatbot_model_free"
    mlflow.set_registry_uri("databricks")

if __name__ == "__main__":
    rag_chain = create_rag_chain(
        is_premium_tier=IS_PREMIUM_TIER,
        index_name=INDEX_NAME,
        vector_search_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        embedding_model_endpoint=EMBEDDING_MODEL_ENDPOINT,
        llm_endpoint=LLM_ENDPOINT,
        full_table_name=FULL_TABLE_NAME
    )
    test_and_log_model(
        chain=rag_chain,
        model_name=MODEL_NAME,
        is_premium_tier=IS_PREMIUM_TIER
    )
