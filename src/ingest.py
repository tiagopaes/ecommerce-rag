#!/usr/bin/env python3
"""
Improved Ingest for RAG
Enhanced document processing with better text representation and metadata
"""

import argparse
import json
import logging
import os
import shutil
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("[IMPROVED_INGEST]")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set in .env")

EMBEDDING_MODEL = "text-embedding-3-small"
DATASET_CSV = "data/products.csv"
CHROMA_DIR = ".chroma"
CHROMA_COLLECTION = "products"

class ImprovedIngest:
    """Enhanced ingest with better document processing"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Brand normalization mapping
        self.brand_mappings = {
            'samsung': 'Samsung',
            'google': 'Google',
            'apple': 'Apple',
            'dell': 'Dell'
        }

        # Category normalization mapping
        self.category_mappings = {
            'phone': 'smartphone',
            'phones': 'smartphone',
            'smartphone': 'smartphone',
            'laptop': 'laptop',
            'laptops': 'laptop',
            'notebook': 'laptop',
            'tablet': 'tablet',
            'tablets': 'tablet',
            'ipad': 'tablet',
            'watch': 'wearable',
            'smartwatch': 'wearable',
            'airpods': 'accessory',
            'headphones': 'accessory',
            'earbuds': 'accessory'
        }

    def normalize_brand(self, brand: str) -> str:
        """Normalize brand names"""
        brand_lower = brand.lower().strip()
        return self.brand_mappings.get(brand_lower, brand)

    def normalize_category(self, category: str) -> str:
        """Normalize category names"""
        category_lower = category.lower().strip()
        return self.category_mappings.get(category_lower, category)

    def extract_specifications(self, attributes_str: str) -> Dict[str, Any]:
        """Extract and normalize specifications from attributes JSON"""
        try:
            attributes = json.loads(attributes_str)
        except (json.JSONDecodeError, TypeError):
            return {}

        # Normalize specifications
        normalized = {}
        for key, value in attributes.items():
            if isinstance(value, str):
                # Normalize storage
                if key == 'storage':
                    normalized[key] = value.upper()
                # Normalize screen size
                elif key == 'screen_size':
                    normalized[key] = str(value)
                # Normalize camera
                elif key == 'camera':
                    normalized[key] = value.upper()
                # Normalize RAM
                elif key == 'ram':
                    normalized[key] = value.upper()
                # Normalize processor
                elif key == 'processor':
                    normalized[key] = value.title()
                else:
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized

    def create_enhanced_text_content(self, row: pd.Series) -> str:
        """Create enhanced text content for better semantic search"""

        # Parse attributes
        attributes = self.extract_specifications(row["attributes"])

        # Create comprehensive text representation
        text_parts = []

        # Basic product info
        text_parts.append(f"Product: {row['title']}")
        text_parts.append(f"Brand: {row['brand']}")
        text_parts.append(f"Category: {row['category']}")

        # Price and availability
        text_parts.append(f"Price: ${row['price']} {row['currency']}")
        text_parts.append(f"Availability: {row['availability']}")

        # Description
        if pd.notna(row['description']) and row['description'].strip():
            text_parts.append(f"Description: {row['description']}")

        # Specifications
        if attributes:
            spec_parts = []
            for key, value in attributes.items():
                if key == 'storage':
                    spec_parts.append(f"Storage: {value}")
                elif key == 'screen_size':
                    spec_parts.append(f"Screen Size: {value} inches")
                elif key == 'camera':
                    spec_parts.append(f"Camera: {value}")
                elif key == 'ram':
                    spec_parts.append(f"RAM: {value}")
                elif key == 'processor':
                    spec_parts.append(f"Processor: {value}")
                elif key == 'color':
                    spec_parts.append(f"Color: {value}")
                elif key == 'chip':
                    spec_parts.append(f"Chip: {value}")
                elif key == 'size':
                    spec_parts.append(f"Size: {value}")
                elif key == 'connectivity':
                    spec_parts.append(f"Connectivity: {value}")
                elif key == 'material':
                    spec_parts.append(f"Material: {value}")
                elif key == 'type':
                    spec_parts.append(f"Type: {value}")
                elif key == 'generation':
                    spec_parts.append(f"Generation: {value}")
                elif key == 'features':
                    spec_parts.append(f"Features: {value}")

            if spec_parts:
                text_parts.append("Specifications:")
                text_parts.extend(spec_parts)

        # Add searchable keywords
        keywords = []

        # Brand keywords
        brand = row['brand'].lower()
        if 'samsung' in brand:
            keywords.extend(['galaxy', 'android', 'samsung'])
        elif 'google' in brand:
            keywords.extend(['pixel', 'android', 'google'])
        elif 'apple' in brand:
            keywords.extend(['iphone', 'ipad', 'macbook', 'airpods', 'watch', 'apple'])
        elif 'dell' in brand:
            keywords.extend(['xps', 'laptop', 'dell'])

        # Category keywords
        category = row['category'].lower()
        if 'smartphone' in category:
            keywords.extend(['phone', 'mobile', 'cell phone'])
        elif 'laptop' in category:
            keywords.extend(['notebook', 'computer', 'pc'])
        elif 'tablet' in category:
            keywords.extend(['ipad', 'tablet'])
        elif 'wearable' in category:
            keywords.extend(['watch', 'smartwatch', 'fitness'])
        elif 'accessory' in category:
            keywords.extend(['headphones', 'earbuds', 'airpods'])

        # Price range keywords
        price = float(row['price'])
        if price < 500:
            keywords.extend(['budget', 'affordable', 'cheap'])
        elif price < 1000:
            keywords.extend(['mid-range', 'moderate'])
        else:
            keywords.extend(['premium', 'high-end', 'expensive'])

        # Add keywords to text
        if keywords:
            text_parts.append(f"Keywords: {', '.join(set(keywords))}")

        return "\n".join(text_parts)

    def create_enhanced_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Create enhanced metadata with normalized values"""

        # Parse attributes
        attributes = self.extract_specifications(row["attributes"])

        # Base metadata
        metadata = {
            "id": row["id"],
            "title": row["title"],
            "category": self.normalize_category(row["category"]),
            "brand": self.normalize_brand(row["brand"]),
            "condition": row["condition"],
            "price": float(row["price"]),
            "currency": row["currency"],
            "availability": row["availability"],
            "url": row["url"],
        }

        # Add normalized attributes to metadata
        for key, value in attributes.items():
            metadata[key] = value

        # Add computed fields
        metadata["price_range"] = self._get_price_range(float(row["price"]))
        metadata["is_budget"] = float(row["price"]) < 500
        metadata["is_premium"] = float(row["price"]) > 1000

        return metadata

    def _get_price_range(self, price: float) -> str:
        """Get price range category"""
        if price < 300:
            return "budget"
        elif price < 800:
            return "mid-range"
        elif price < 1500:
            return "premium"
        else:
            return "luxury"

    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create enhanced documents from DataFrame"""
        documents = []

        for _, row in df.iterrows():
            # Create enhanced text content
            text_content = self.create_enhanced_text_content(row)

            # Create enhanced metadata
            metadata = self.create_enhanced_metadata(row)

            # Create Document object
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)

        return documents

    def ingest(self, csv_path: str = DATASET_CSV, reset: bool = False):
        """Main ingest process"""

        if reset and os.path.isdir(CHROMA_DIR):
            logger.info("Reset requested. Removing directory: %s", CHROMA_DIR)
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        logger.info("Loading dataset: %s", csv_path)
        df = pd.read_csv(csv_path)
        if df.empty:
            raise SystemExit("CSV is empty. Aborting.")

        logger.info(f"Building documents: {len(df)} rows")

        # Create enhanced documents
        documents = self.create_documents(df)

        logger.info(f"Created {len(documents)} documents")

        # Initialize Chroma vector store
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        # Add documents to vector store
        logger.info("Adding documents to Chroma database...")
        vectorstore.add_documents(documents)

        logger.info(f"Successfully ingested {len(documents)} products into Chroma database")
        logger.info(f"Database saved to: {CHROMA_DIR}")
        logger.info(f"Collection name: {CHROMA_COLLECTION}")

        # Log some statistics
        self._log_statistics(documents)

    def _log_statistics(self, documents: List[Document]):
        """Log ingestion statistics"""
        categories = {}
        brands = {}
        price_ranges = {}

        for doc in documents:
            metadata = doc.metadata

            # Category stats
            category = metadata.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1

            # Brand stats
            brand = metadata.get('brand', 'unknown')
            brands[brand] = brands.get(brand, 0) + 1

            # Price range stats
            price_range = metadata.get('price_range', 'unknown')
            price_ranges[price_range] = price_ranges.get(price_range, 0) + 1

        logger.info("Ingestion Statistics:")
        logger.info(f"Categories: {categories}")
        logger.info(f"Brands: {brands}")
        logger.info(f"Price Ranges: {price_ranges}")

def main():
    parser = argparse.ArgumentParser(description="Improved Ingest CSV into Chroma")
    parser.add_argument(
        "--reset", action="store_true", help="Delete chroma dir before indexing"
    )
    parser.add_argument(
        "--csv", default=DATASET_CSV, help="CSV file to ingest"
    )
    args = parser.parse_args()

    # Initialize improved ingest
    ingest = ImprovedIngest()

    # Run ingest
    ingest.ingest(args.csv, args.reset)

if __name__ == "__main__":
    main()
