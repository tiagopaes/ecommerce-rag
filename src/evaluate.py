#!/usr/bin/env python3
"""
Simple RAG Evaluation Script
Tests vector database quality with essential metrics:
- Precision, Recall, F1 Score
- Similarity Score Validation
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.table import Table

# Config
load_dotenv()
logging.basicConfig(level=logging.WARNING)
console = Console()

@dataclass
class TestCase:
    """Single test case with expected results"""
    question: str
    expected_product_ids: List[str]
    expected_categories: List[str]
    expected_brands: List[str]
    min_similarity_score: float

@dataclass
class QueryResult:
    """Results from a single query"""
    question: str
    retrieved_products: List[Dict[str, Any]]
    similarity_scores: List[float]
    response_time: float
    expected_product_ids: List[str]
    expected_categories: List[str]
    expected_brands: List[str]

class RAGEvaluator:
    """Simple RAG evaluation with essential metrics"""

    def __init__(self, chroma_dir: str = ".chroma", collection_name: str = "products"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir
        )

    def load_test_cases(self, csv_path: str = "data/expected_answers.csv") -> List[TestCase]:
        """Load test cases from CSV file"""
        test_cases = []

        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse expected product IDs
                expected_ids = [id.strip() for id in row['expected_product_ids'].split(',')] if row['expected_product_ids'] else []

                # Parse expected categories
                expected_categories = [cat.strip() for cat in row['expected_categories'].split(',')] if row['expected_categories'] else []

                # Parse expected brands
                expected_brands = [brand.strip() for brand in row['expected_brands'].split(',')] if row['expected_brands'] else []

                test_case = TestCase(
                    question=row['question'],
                    expected_product_ids=expected_ids,
                    expected_categories=expected_categories,
                    expected_brands=expected_brands,
                    min_similarity_score=float(row['min_similarity_score'])
                )
                test_cases.append(test_case)

        return test_cases

    def run_query(self, test_case: TestCase, k: int = 5) -> QueryResult:
        """Run a single query and return results"""
        start_time = time.time()

        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(
            test_case.question,
            k=k
        )

        response_time = time.time() - start_time

        # Extract documents and scores
        retrieved_products = []
        similarity_scores = []

        for doc, score in results:
            retrieved_products.append({
                "id": doc.metadata["id"],
                "title": doc.metadata["title"],
                "category": doc.metadata["category"],
                "brand": doc.metadata["brand"],
                "price": doc.metadata["price"],
                "content": doc.page_content
            })
            similarity_scores.append(score)

        return QueryResult(
            question=test_case.question,
            retrieved_products=retrieved_products,
            similarity_scores=similarity_scores,
            response_time=response_time,
            expected_product_ids=test_case.expected_product_ids,
            expected_categories=test_case.expected_categories,
            expected_brands=test_case.expected_brands
        )

    def calculate_precision(self, result: QueryResult) -> float:
        """Calculate Precision: Relevant results / Total retrieved results"""
        if not result.retrieved_products:
            return 0.0

        retrieved_ids = [product["id"] for product in result.retrieved_products]
        relevant_retrieved = len(set(retrieved_ids) & set(result.expected_product_ids))

        return relevant_retrieved / len(retrieved_ids)

    def calculate_recall(self, result: QueryResult) -> float:
        """Calculate Recall: Relevant results / Total expected results"""
        if not result.expected_product_ids:
            return 1.0 if not result.retrieved_products else 0.0

        retrieved_ids = [product["id"] for product in result.retrieved_products]
        relevant_retrieved = len(set(retrieved_ids) & set(result.expected_product_ids))

        return relevant_retrieved / len(result.expected_product_ids)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 Score: Harmonic mean of precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_similarity_validation(self, result: QueryResult, min_threshold: float = 0.7) -> Dict[str, Any]:
        """Calculate similarity score validation metrics"""
        if not result.similarity_scores:
            return {
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "above_threshold": 0,
                "below_threshold": 0,
                "threshold_rate": 0.0
            }

        avg_similarity = sum(result.similarity_scores) / len(result.similarity_scores)
        min_similarity = min(result.similarity_scores)
        max_similarity = max(result.similarity_scores)

        above_threshold = sum(1 for score in result.similarity_scores if score >= min_threshold)
        below_threshold = len(result.similarity_scores) - above_threshold
        threshold_rate = above_threshold / len(result.similarity_scores)

        return {
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "above_threshold": above_threshold,
            "below_threshold": below_threshold,
            "threshold_rate": threshold_rate
        }

    def evaluate_all(self, csv_path: str = "data/expected_answers.csv", k: int = 5) -> Dict[str, Any]:
        """Run evaluation on all test cases"""
        console.print("[bold blue]Starting RAG Evaluation[/bold blue]")

        # Load test cases
        test_cases = self.load_test_cases(csv_path)
        console.print(f"[green]Loaded {len(test_cases)} test cases[/green]")

        # Run all queries
        results = []
        for i, test_case in enumerate(test_cases, 1):
            console.print(f"[yellow]Running test {i}/{len(test_cases)}: {test_case.question[:50]}...[/yellow]")
            result = self.run_query(test_case, k)
            results.append(result)

        # Calculate metrics for each result
        individual_metrics = []
        for result in results:
            precision = self.calculate_precision(result)
            recall = self.calculate_recall(result)
            f1_score = self.calculate_f1_score(precision, recall)
            similarity_metrics = self.calculate_similarity_validation(result)

            individual_metrics.append({
                "question": result.question,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "response_time": result.response_time,
                "similarity_metrics": similarity_metrics,
                "retrieved_count": len(result.retrieved_products),
                "expected_count": len(result.expected_product_ids)
            })

        # Calculate overall metrics
        overall_metrics = {
            "avg_precision": sum(m["precision"] for m in individual_metrics) / len(individual_metrics),
            "avg_recall": sum(m["recall"] for m in individual_metrics) / len(individual_metrics),
            "avg_f1_score": sum(m["f1_score"] for m in individual_metrics) / len(individual_metrics),
            "avg_response_time": sum(m["response_time"] for m in individual_metrics) / len(individual_metrics),
            "avg_similarity": sum(m["similarity_metrics"]["avg_similarity"] for m in individual_metrics) / len(individual_metrics),
            "avg_threshold_rate": sum(m["similarity_metrics"]["threshold_rate"] for m in individual_metrics) / len(individual_metrics),
            "total_tests": len(test_cases)
        }

        return {
            "overall_metrics": overall_metrics,
            "individual_results": individual_metrics,
            "raw_results": [
                {
                    "question": r.question,
                    "retrieved_products": r.retrieved_products,
                    "similarity_scores": r.similarity_scores,
                    "expected_product_ids": r.expected_product_ids
                }
                for r in results
            ]
        }

    def display_results(self, results: Dict[str, Any]):
        """Display evaluation results in a formatted table"""
        overall = results["overall_metrics"]

        # Overall metrics table
        console.print("\n[bold green]RAG Evaluation Results[/bold green]")

        summary_table = Table(title="Overall Metrics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_column("Status", style="yellow")

        # Precision
        precision_status = "✅ Good" if overall["avg_precision"] > 0.7 else "⚠️ Needs Improvement"
        summary_table.add_row("Average Precision", f"{overall['avg_precision']:.3f}", precision_status)

        # Recall
        recall_status = "✅ Good" if overall["avg_recall"] > 0.7 else "⚠️ Needs Improvement"
        summary_table.add_row("Average Recall", f"{overall['avg_recall']:.3f}", recall_status)

        # F1 Score
        f1_status = "✅ Good" if overall["avg_f1_score"] > 0.7 else "⚠️ Needs Improvement"
        summary_table.add_row("Average F1 Score", f"{overall['avg_f1_score']:.3f}", f1_status)

        # Similarity
        similarity_status = "✅ Good" if overall["avg_similarity"] > 0.7 else "⚠️ Needs Improvement"
        summary_table.add_row("Average Similarity", f"{overall['avg_similarity']:.3f}", similarity_status)

        # Threshold Rate
        threshold_status = "✅ Good" if overall["avg_threshold_rate"] > 0.8 else "⚠️ Needs Improvement"
        summary_table.add_row("Threshold Rate", f"{overall['avg_threshold_rate']:.3f}", threshold_status)

        # Response Time
        time_status = "✅ Fast" if overall["avg_response_time"] < 1.0 else "⚠️ Slow"
        summary_table.add_row("Avg Response Time", f"{overall['avg_response_time']:.3f}s", time_status)

        # Total Tests
        summary_table.add_row("Total Tests", str(overall["total_tests"]), "📊")

        console.print(summary_table)

        # Individual results table
        individual_table = Table(title="Individual Test Results")
        individual_table.add_column("Question", style="cyan", max_width=40)
        individual_table.add_column("Precision", style="green")
        individual_table.add_column("Recall", style="green")
        individual_table.add_column("F1", style="green")
        individual_table.add_column("Similarity", style="yellow")
        individual_table.add_column("Time (s)", style="blue")

        for result in results["individual_results"]:
            question_short = result["question"][:37] + "..." if len(result["question"]) > 40 else result["question"]
            individual_table.add_row(
                question_short,
                f"{result['precision']:.3f}",
                f"{result['recall']:.3f}",
                f"{result['f1_score']:.3f}",
                f"{result['similarity_metrics']['avg_similarity']:.3f}",
                f"{result['response_time']:.3f}"
            )

        console.print(individual_table)

        # Recommendations
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        if overall["avg_f1_score"] < 0.7:
            console.print("• Improve embedding model or query preprocessing")
        if overall["avg_similarity"] < 0.7:
            console.print("• Consider fine-tuning similarity thresholds")
        if overall["avg_response_time"] > 1.0:
            console.print("• Optimize vector store configuration")
        if overall["avg_threshold_rate"] < 0.8:
            console.print("• Review similarity score thresholds")

        if all([
            overall["avg_f1_score"] > 0.7,
            overall["avg_similarity"] > 0.7,
            overall["avg_response_time"] < 1.0,
            overall["avg_threshold_rate"] > 0.8
        ]):
            console.print("• [green]All metrics look good! 🎉[/green]")

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Evaluation")
    parser.add_argument("--chroma-dir", default=".chroma", help="Chroma database directory")
    parser.add_argument("--collection", default="products", help="Chroma collection name")
    parser.add_argument("--csv", default="data/expected_answers.csv", help="CSV file with expected answers")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve per query")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RAGEvaluator(args.chroma_dir, args.collection)

    # Run evaluation
    results = evaluator.evaluate_all(args.csv, args.k)

    # Display results
    evaluator.display_results(results)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {args.output}[/green]")

if __name__ == "__main__":
    main()
