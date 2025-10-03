# E-commerce RAG System

RAG system for e-commerce product search using agents, ChromaDB, and OpenAI embeddings with LLM-based evaluation.

## Features
- **LangChain Agent**: Conversational product search with natural language understanding
- **Vector Store Ingestion**: Enhanced document processing with metadata enrichment
- **LLM Judge Evaluation**: Evaluation system using GPT-4.1-nano as judge

## Technologies
- **Python**: Core language
- **LangChain**: Agent framework and RAG implementation
- **ChromaDB**: Vector store for embeddings
- **OpenAI**: LLM for embedding, agent and judge evaluation
- **Rich**: Beautiful terminal output formatting
- **Pandas**: Data processing and manipulation

## Project Structure
```
ecommerce-rag/
├── data/
│   ├── products.csv              # Products data
│   └── test_questions.csv        # Test questions for evaluation
├── src/
│   ├── ingest.py                 # Ingestion script
│   ├── agent.py                  # LangChain Agent with conversational i
│   └── evaluate.py               # LLM Judge evaluation system
├── .chroma/                      # ChromaDB vector store
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Data Ingestion
```bash
# Ingest products into vector database
python src/ingest.py
```

### 4. Test the Agent
```bash
# Run the conversational agent
python src/agent.py --query "Find Samsung smartphones under $1000"
```

### 5. Evaluate Performance
```bash
# Evaluate agent with LLM judge
python src/evaluate.py --max-tests 5 --output results.json
```

## RAG Agent Features

### Conversational Interface
The agent provides natural language product search with:
- **Smart Query Understanding**: Interprets complex product requests
- **Contextual Responses**: Provides detailed product information
- **Professional Formatting**: Well-structured, customer-friendly responses
- **Multi-criteria Search**: Handles price, brand, category, and feature filters

### Example Interactions
```
User: "Find Samsung smartphones under $1000"
Agent: "Here are some Samsung smartphones under $1000:
        1. Samsung Galaxy S23 Ultra 256GB - $999
        2. Samsung Galaxy S23 - $799
        ..."
```

## LLM Judge Evaluation System

### Evaluation Approach
Instead of traditional metrics, we use **GPT-4.1-nano as a judge** to evaluate response quality:

### Evaluation Criteria (0-1 scale)
1. **Product Relevance**: Does the response mention expected products?
2. **Information Completeness**: Are product details comprehensive?
3. **Response Structure**: Is the response well-organized and clear?
4. **Customer Service Quality**: Is the tone helpful and professional?
5. **Accuracy**: Are product details accurate and consistent?

### Scoring System
- **0.0-0.5**: Poor/Incorrect
- **0.5-0.7**: Average/Partially correct
- **0.7**: Good (baseline threshold) ⭐
- **0.8-0.9**: Very good
- **1.0**: Excellent/Perfect

### Baseline Success Rate
- **Target**: ≥80% of responses score ≥0.7
- **Production Ready**: When baseline success >80% and response time <10s

## Test Data Format

The `data/test_questions.csv` file contains test cases:

```csv
question,expected_product_ids,expected_categories,expected_brands,min_similarity_score
"Find Samsung smartphones","E001,E004,E007","smartphone","Samsung",0.8
"Show me laptops under $1000","E003,E006","laptop","Dell",0.7
```

### Columns:
- **question**: Test question
- **expected_product_ids**: Expected product IDs (comma-separated)
- **expected_categories**: Expected categories (comma-separated)
- **expected_brands**: Expected brands (comma-separated)
- **min_similarity_score**: Minimum expected similarity score

## Understanding Results

### Agent Performance Metrics
```
Agent Performance Summary
┌─────────────────────┬───────┬─────────────────────┐
│ Metric              │ Score │ Status              │
├─────────────────────┼───────┼─────────────────────┤
│ Overall Score       │ 0.820 │ ✅ Excellent        │
│ Product Relevance   │ 0.720 │ ✅ Good             │
│ Info Completeness   │ 0.900 │ ✅ Excellent        │
│ Response Structure  │ 0.950 │ ✅ Excellent        │
│ Customer Service    │ 1.000 │ ✅ Perfect          │
│ Accuracy            │ 0.900 │ ✅ Excellent        │
│ Baseline Success    │ 100%  │ ✅ Excellent        │
│ Response Time       │ 9.8s  │ ⚠️ Needs Optimization │
└─────────────────────┴───────┴─────────────────────┘
```

### Status Indicators
- **✅ Excellent**: Score > 0.8
- **✅ Good**: Score > 0.7 (meets baseline)
- **⚠️ Needs Improvement**: Score < 0.7

### Production Readiness
The system provides clear recommendations:
- **🚀 Ready for Production**: When all metrics meet standards
- **⚠️ Needs Optimization**: When improvements are required


