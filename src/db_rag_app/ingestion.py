import os
import json
import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, udf, explode
from langchain.text_splitter import RecursiveCharacterTextSplitter

def download_data(data_url: str, file_path: str):
    """Downloads a JSON feed and writes it to the appropriate storage."""
    try:
        print(f"Downloading data from {data_url}...")
        response = requests.get(data_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"Writing data to {file_path}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print("Data ingestion complete.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise
    except IOError as e:
        print(f"Error writing to storage: {e}")
        raise

def process_and_save_to_delta(file_path: str, full_table_name: str, is_premium_tier: bool):
    """Reads JSON, chunks text using LangChain, and saves to Delta."""
    spark = SparkSession.builder.getOrCreate()
    
    print(f"Reading data from {file_path}...")
    try:
        # In Spark, we read from dbfs:/ if it's Free Tier, or /Volumes/ if Premium
        spark_read_path = file_path.replace("/dbfs", "dbfs:") if not is_premium_tier else file_path
        df = spark.read.json(spark_read_path, multiLine=True)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        raise

    if "body" not in df.columns or "id" not in df.columns or "title" not in df.columns:
        raise ValueError("Source JSON must contain 'id', 'title', and 'body' fields.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    @udf(returnType="array<string>")
    def split_text(text: str):
        if not text:
            return []
        chunks = text_splitter.split_text(text)
        return chunks

    print("Chunking text data...")
    chunked_df = df.withColumn("chunks", split_text(col("body"))) \
                   .withColumn("chunk_text", explode(col("chunks"))) \
                   .drop("chunks", "body")
    
    chunked_df = chunked_df.withColumn("chunk_id", udf(lambda i, ct: f"{i}_{hash(ct)}", StringType())(col("id"), col("chunk_text")))

    print(f"Writing to Delta table {full_table_name}...")
    try:
        writer = chunked_df.write.format("delta").mode("overwrite")
        if is_premium_tier:
            writer = writer.option("delta.enableChangeDataFeed", "true")
        writer.saveAsTable(full_table_name)
        print("Data processing and Delta table creation complete.")
    except Exception as e:
        print(f"Error writing to Delta table: {e}")
        raise
