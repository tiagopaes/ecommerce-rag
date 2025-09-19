# E-commerce RAG System

Simple RAG (Retrieval-Augmented Generation) system for e-commerce product search using ChromaDB and OpenAI embeddings.

## Quick Start

### 1. Data Ingestion
```bash
# Ingest products from CSV to vector database
python src/ingest.py
```

### 2. Quality Evaluation
```bash
# Evaluate RAG quality with essential metrics
python src/evaluate.py
```

## Evaluation Metrics

The system evaluates RAG quality with 4 essential metrics:

### **Precision**
- **Formula**: Relevant results / Total retrieved results
- **Interpretation**: How many of the returned results are actually relevant?
- **Target**: > 0.7 (70%)

### **Recall**
- **Formula**: Relevant results / Total expected results
- **Interpretation**: How many of the expected results were found?
- **Target**: > 0.7 (70%)

### **F1 Score**
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean between precision and recall
- **Target**: > 0.7 (70%)

### **Similarity Score Validation**
- **Metric**: Rate of results above similarity threshold
- **Interpretation**: How many results have adequate similarity?
- **Target**: > 0.8 (80%)

## File Structure

```
ecommerce-rag/
├── data/
│   ├── products.csv              # Product data
│   └── expected_answers.csv      # Expected answers for testing
├── src/
│   ├── ingest.py                 # Ingestion script
│   └── evaluate.py               # Evaluation script
├── .chroma/                      # ChromaDB vector database
└── requirements.txt              # Python dependencies
```

## How to Run

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### Dependencies Installation
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Usage

### Ingestion
```bash
# Basic ingestion
python src/ingest.py

# Reset database and re-ingest
python src/ingest.py --reset
```

### Evaluation
```bash
# Basic evaluation
python src/evaluate.py

# Evaluation with custom parameters
python src/evaluate.py --k 10 --output results.json

# Use custom CSV
python src/evaluate.py --csv data/my_tests.csv
```

## Test CSV

The `data/expected_answers.csv` file contains tests with the following format:

```csv
question,expected_product_ids,expected_categories,expected_brands,min_similarity_score
"Find Samsung smartphones","E001,E004,E007","smartphone","Samsung",0.8
```

### Columns:
- **question**: Test question
- **expected_product_ids**: Expected product IDs (comma-separated)
- **expected_categories**: Expected categories (comma-separated)
- **expected_brands**: Expected brands (comma-separated)
- **min_similarity_score**: Minimum expected similarity score

## Interpreting Results

### Result Status:
- **✅ Good**: Metric within target
- **⚠️ Needs Improvement**: Metric below target

### Example Output:
```
RAG Evaluation Results
┌─────────────────────┬───────┬─────────────────────┐
│ Metric              │ Value │ Status              │
├─────────────────────┼───────┼─────────────────────┤
│ Average Precision   │ 0.850 │ ✅ Good             │
│ Average Recall      │ 0.720 │ ✅ Good             │
│ Average F1 Score    │ 0.780 │ ✅ Good             │
│ Average Similarity  │ 0.820 │ ✅ Good             │
│ Threshold Rate      │ 0.900 │ ✅ Good             │
│ Avg Response Time   │ 0.450s│ ✅ Fast             │
└─────────────────────┴───────┴─────────────────────┘
```

