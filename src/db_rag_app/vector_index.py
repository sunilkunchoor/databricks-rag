import time

def provision_vector_search(
    is_premium_tier: bool,
    vector_search_endpoint_name: str,
    index_name: str,
    source_table_name: str,
    primary_key: str,
    text_column: str,
    embedding_model_endpoint: str
):
    """Provisions a Vector Search endpoint and Delta Sync index."""
    if not is_premium_tier:
        print("Free Tier detected. Skipping Databricks Vector Search provisioning.")
        print("We will use a local in-memory Vector Store (FAISS) in the next step.")
        return

    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.vectorsearch import EndpointType, VectorIndexType
    except ImportError:
        print("Please install databricks-sdk: pip install databricks-sdk")
        return

    w = WorkspaceClient()

    try:
        print(f"Checking for Vector Search endpoint: {vector_search_endpoint_name}")
        endpoints = w.vector_search_endpoints.list_endpoints()
        endpoint_names = [e.name for e in endpoints if e.name == vector_search_endpoint_name]
        
        if not endpoint_names:
            print(f"Creating endpoint {vector_search_endpoint_name}...")
            w.vector_search_endpoints.create_endpoint_and_wait(
                name=vector_search_endpoint_name,
                endpoint_type=EndpointType.STANDARD,
                timeout=time.timedelta(minutes=15)
            )
            print("Endpoint created successfully.")
        else:
            print("Endpoint already exists.")
            
    except Exception as e:
        print(f"Error managing Vector Search endpoint: {e}")
        raise

    try:
        print(f"Checking for index: {index_name}")
        indexes = w.vector_search_indexes.list_indexes(vector_search_endpoint_name)
        index_names = [i.name for i in indexes if i.name == index_name]
        
        if not index_names:
            print(f"Creating Delta Sync Vector Index {index_name}...")
            w.vector_search_indexes.create_index_and_wait(
                name=index_name,
                endpoint_name=vector_search_endpoint_name,
                primary_key=primary_key,
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec={
                    "source_table": source_table_name,
                    "pipeline_type": "TRIGGERED",
                    "embedding_source_columns": [
                        {
                            "name": text_column,
                            "embedding_model_endpoint_name": embedding_model_endpoint
                        }
                    ]
                },
                timeout=time.timedelta(minutes=15)
            )
            print("Vector Index created and synced successfully.")
        else:
            print("Index already exists. Triggering manual sync...")
            w.vector_search_indexes.sync_index(index_name)
            print("Sync triggered.")
            
    except Exception as e:
        print(f"Error managing Vector Search index: {e}")
        raise
