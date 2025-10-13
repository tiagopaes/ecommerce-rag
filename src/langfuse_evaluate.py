#!/usr/bin/env python3
"""
RAG Agent Evaluation Script
Evaluates agent performance using Langfuse LLM as judge
"""

import json
import logging
import os

from dotenv import load_dotenv
from langfuse import Langfuse
from rich.console import Console

from agent import RAGAgent

# Config
load_dotenv()
logging.basicConfig(level=logging.WARNING)
console = Console()
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://us.cloud.langfuse.com"
)

def my_task(*, item, **kwargs):
    question = item.input
    agent = RAGAgent()
    response = agent.search_products(question)
    return json.dumps(response, indent=2)


def main():
    dataset = langfuse.get_dataset("Test Questions")

    result = dataset.run_experiment(
        name="Agentic RAG Test",
        description="Agentic RAG Evaluation",
        task=my_task
    )

    print(result.format())

if __name__ == "__main__":
    main()
