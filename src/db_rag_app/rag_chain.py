import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(
    is_premium_tier: bool,
    index_name: str = None,
    vector_search_endpoint_name: str = None,
    embedding_model_endpoint: str = None,
    llm_endpoint: str = None,
    full_table_name: str = None
):
    """Defines and returns the LangChain RAG pipeline based on the tier."""
    
    if is_premium_tier:
        from langchain_community.vectorstores import DatabricksVectorSearch
        from langchain_community.embeddings import DatabricksEmbeddings
        from langchain_community.chat_models.databricks import ChatDatabricks
        
        print("Initializing Databricks LangChain components (Premium Tier)...")
        embeddings = DatabricksEmbeddings(endpoint=embedding_model_endpoint)
        vector_store = DatabricksVectorSearch(
            endpoint=vector_search_endpoint_name,
            index_name=index_name,
            text_column="chunk_text",
            embedding=embeddings
        )
        llm = ChatDatabricks(endpoint=llm_endpoint, max_tokens=500)
        
    else:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        
        print("Initializing Local/Open-Source LangChain components (Free Tier)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        spark = SparkSession.builder.getOrCreate()
        print(f"Loading data from {full_table_name} for local FAISS index...")
        pdf = spark.table(full_table_name).select("chunk_text").toPandas()
        texts = pdf["chunk_text"].tolist()
        
        vector_store = FAISS.from_texts(texts, embeddings)
        
        print("Loading local HuggingFace model...")
        pipe = pipeline("text-generation", model="gpt2", max_new_tokens=50)
        llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Keep the answer concise.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def test_and_log_model(chain, model_name: str, is_premium_tier: bool):
    """Tests the chain and logs it to MLflow."""
    
    question = "What is the main topic of the documents?"
    
    print(f"Testing chain with question: '{question}'")
    try:
        answer = chain.invoke(question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error invoking chain: {e}")
        
    print("Logging model to MLflow...")
    
    if is_premium_tier:
        mlflow.langchain.autolog()
    
    input_example = "Tell me about the recent posts."
    signature = infer_signature(input_example, "This is a placeholder answer.")
    
    try:
        with mlflow.start_run(run_name="rag_chain_run") as run:
            model_info = mlflow.langchain.log_model(
                lc_model=chain,
                artifact_path="langchain_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name
            )
            print(f"Model logged successfully: {model_info.model_uri}")
            print(f"Model registered as: {model_name}")
    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        raise
