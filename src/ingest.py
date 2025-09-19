from __future__ import annotations

import argparse
import logging
import os
import shutil

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("[INGEST]")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set in .env")

EMBEDDING_MODEL = "text-embedding-3-small"
DATASET_CSV = "data/products.csv"
CHROMA_DIR = ".chroma"
CHROMA_COLLECTION = "products"


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV into Chroma (SQLite).")
    parser.add_argument(
        "--reset", action="store_true", help="Delete chroma dir before indexing"
    )
    args = parser.parse_args()

    if args.reset and os.path.isdir(CHROMA_DIR):
        logger.info("Reset requested. Removing directory: %s", CHROMA_DIR)
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    logger.info("Loading dataset: %s", DATASET_CSV)
    df = pd.read_csv(DATASET_CSV)
    if df.empty:
        raise SystemExit("CSV is empty. Aborting.")

    logger.info(f"Building documents: {len(df)} rows")

    # Convert CSV rows to documents
    documents = []
    for _, row in df.iterrows():
        # Create a comprehensive text representation of the product
        text_content = f"""
            Product: {row["title"]}
            Category: {row["category"]}
            Brand: {row["brand"]}
            Condition: {row["condition"]}
            Price: {row["price"]} {row["currency"]}
            Description: {row["description"]}
            Attributes: {row["attributes"]}
            Availability: {row["availability"]}
            URL: {row["url"]}
            """.strip()

        # Create metadata for filtering and retrieval
        metadata = {
            "id": row["id"],
            "title": row["title"],
            "category": row["category"],
            "brand": row["brand"],
            "condition": row["condition"],
            "price": float(row["price"]),
            "currency": row["currency"],
            "availability": row["availability"],
            "url": row["url"],
        }

        # Create Document object
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    logger.info(f"Created {len(documents)} documents")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    # Initialize or load Chroma vector store
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Add documents to vector store
    logger.info("Adding documents to Chroma database...")
    vectorstore.add_documents(documents)

    logger.info(f"Successfully ingested {len(documents)} products into Chroma database")
    logger.info(f"Database saved to: {CHROMA_DIR}")
    logger.info(f"Collection name: {CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()
