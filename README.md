# Agentic RAG for E-commerce

An **Agentic Retrieval-Augmented Generation (RAG)** system for e-commerce product search, combining intelligent agents, vector retrieval, and LLM-based evaluation (LLM as Judge).

## Overview

This project implements a **RAG pipeline** that:
* Stores and retrieves product data using **ChromaDB** and **OpenAI embeddings**
* Uses a **LangChain Agent** to reason, reformulate queries, and generate natural answers to user queries
* Evaluates its own performance via an **LLM Judge** powered by GPT-4.1-nano (LLM as Judge)

---

## Tech Stack

* **Python** – Core language
* **LangChain** – Agent framework
* **ChromaDB** – Vector store
* **OpenAI API** – Embeddings + LLMs (Agent + Judge)
* **Rich / Pandas** – UI + data handling

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
echo "OPENAI_API_KEY=sk-xxx" > .env

# 3. Ingest product data
python src/ingest.py

# 4. Run the agent
python src/agent.py --query "Find Samsung smartphones under $1000"

# 5. Evaluate results
python src/evaluate.py --max-tests 5
```

---

## Example Evaluation Output

```
Agentic Evaluation Summary
─────────────────────────────────────────────
Overall Score       0.82   ✅ Excellent
Relevance           0.72   ✅ Good
Completeness        0.90   ✅ Excellent
Structure           0.95   ✅ Excellent
Tone                1.00   ✅ Perfect
Accuracy            0.90   ✅ Excellent
Response Time       9.8s   ⚠️ Needs Optimization
─────────────────────────────────────────────
Status: 🚀 Production-Ready
```

---

## Project Structure

```
ecommerce-agentic-rag/
├── data/
│   ├── products.csv
│   └── test_questions.csv
├── src/
│   ├── ingest.py
│   ├── agent.py
│   └── evaluate.py
└── requirements.txt
```
