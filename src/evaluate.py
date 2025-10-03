#!/usr/bin/env python3
"""
RAG Agent Evaluation Script
Evaluates agent performance using LLM as judge
Focuses on agent quality assessment
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from agent import RAGAgent

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

class AgentEvaluator:
    """Evaluator for RAG Agent using LLM as judge"""

    def __init__(self, chroma_dir: str = ".chroma", collection_name: str = "products"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

        # Initialize agent
        self.agent = RAGAgent(chroma_dir, collection_name)

        # Initialize LLM judge
        self.judge_client = OpenAI()
        self.judge_model = "gpt-4o-mini"

    def load_test_cases(self, csv_path: str = "data/test_questions.csv") -> List[TestCase]:
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

    def test_agent_approach(self, question: str) -> Dict[str, Any]:
        """Test agent approach"""
        start_time = time.time()

        try:
            result = self.agent.search_products(question)
            response_time = time.time() - start_time

            if result["success"]:
                products = self.agent.get_products_from_response(result)
                return {
                    "success": True,
                    "products": products,
                    "response_time": response_time,
                    "raw_response": result["response"]
                }
            else:
                return {
                    "success": False,
                    "products": [],
                    "response_time": response_time,
                    "error": result.get("error", "Unknown error")
                }
        except Exception as e:
            return {
                "success": False,
                "products": [],
                "response_time": time.time() - start_time,
                "error": str(e)
            }

    def llm_judge_evaluation(self, question: str, agent_response: str, expected_products: List[str], expected_categories: List[str], expected_brands: List[str]) -> Dict[str, Any]:
        """Use LLM as judge to evaluate agent response quality"""

        # Create expected response context
        expected_context = ""
        if expected_products:
            expected_context += f"Expected Product IDs: {', '.join(expected_products)}\n"
        if expected_categories:
            expected_context += f"Expected Categories: {', '.join(expected_categories)}\n"
        if expected_brands:
            expected_context += f"Expected Brands: {', '.join(expected_brands)}\n"

        judge_prompt = f"""
You are an expert e-commerce assistant evaluator. Your task is to evaluate how well an AI agent answered a customer's product search question.

CUSTOMER QUESTION: {question}

AGENT RESPONSE: {agent_response}

EXPECTED CONTEXT:
{expected_context}

EVALUATION CRITERIA:
1. **Product Relevance (0-1)**: Does the response mention the expected products or similar relevant products?
2. **Information Completeness (0-1)**: Does the response provide useful product details (price, features, specifications)?
3. **Response Structure (0-1)**: Is the response well-organized, clear, and easy to read?
4. **Customer Service Quality (0-1)**: Does the response sound helpful, professional, and customer-focused?
5. **Accuracy (0-1)**: Are the product details accurate and consistent?

SCORING INSTRUCTIONS:
- Give each criterion a score from 0.0 to 1.0
- 0.0 = Poor/Incorrect
- 0.5 = Average/Partially correct
- 0.7 = Good (baseline threshold)
- 0.8-0.9 = Very good
- 1.0 = Excellent/Perfect

RESPOND ONLY WITH A JSON OBJECT in this exact format:
{{
    "product_relevance": 0.0-1.0,
    "information_completeness": 0.0-1.0,
    "response_structure": 0.0-1.0,
    "customer_service_quality": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "explanation": "Brief explanation of the scores"
}}
"""

        try:
            response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert e-commerce assistant evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            # Parse JSON response
            import json
            judge_result = json.loads(response.choices[0].message.content)

            return {
                "success": True,
                "scores": judge_result,
                "overall_score": judge_result.get("overall_score", 0.0),
                "explanation": judge_result.get("explanation", ""),
                "meets_baseline": judge_result.get("overall_score", 0.0) >= 0.7
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "scores": {},
                "overall_score": 0.0,
                "explanation": f"Judge evaluation failed: {str(e)}",
                "meets_baseline": False
            }

    def evaluate_agent(self, csv_path: str = "data/test_questions.csv", max_tests: int = None) -> Dict[str, Any]:
        """Evaluate agent performance using LLM judge"""
        console.print("[bold blue]Starting Agent Evaluation with LLM Judge[/bold blue]")

        # Load test cases
        test_cases = self.load_test_cases(csv_path)
        if max_tests:
            test_cases = test_cases[:max_tests]

        console.print(f"[green]Loaded {len(test_cases)} test cases[/green]")

        # Run evaluations
        results = []
        for i, test_case in enumerate(test_cases, 1):
            question = test_case.question
            expected_ids = test_case.expected_product_ids

            console.print(f"[yellow]Running test {i}/{len(test_cases)}: {question[:50]}...[/yellow]")

            # Test agent approach
            agent_result = self.test_agent_approach(question)

            # Use LLM judge to evaluate agent response
            agent_response_text = agent_result.get("raw_response", "")
            judge_evaluation = self.llm_judge_evaluation(
                question,
                agent_response_text,
                expected_ids,
                test_case.expected_categories,
                test_case.expected_brands
            )

            results.append({
                "question": question,
                "expected_ids": expected_ids,
                "agent": {
                    "success": agent_result["success"],
                    "response_time": agent_result["response_time"],
                    "response_text": agent_response_text,
                    "judge_scores": judge_evaluation.get("scores", {}),
                    "overall_score": judge_evaluation.get("overall_score", 0.0),
                    "meets_baseline": judge_evaluation.get("meets_baseline", False),
                    "judge_explanation": judge_evaluation.get("explanation", ""),
                    "judge_success": judge_evaluation.get("success", False)
                }
            })

        # Calculate overall metrics
        agent_overall_scores = [r["agent"]["overall_score"] for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_meets_baseline = [r["agent"]["meets_baseline"] for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_times = [r["agent"]["response_time"] for r in results if r["agent"]["success"]]

        # Extract individual judge scores
        agent_product_relevance = [r["agent"]["judge_scores"].get("product_relevance", 0.0) for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_info_completeness = [r["agent"]["judge_scores"].get("information_completeness", 0.0) for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_response_structure = [r["agent"]["judge_scores"].get("response_structure", 0.0) for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_customer_service = [r["agent"]["judge_scores"].get("customer_service_quality", 0.0) for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]
        agent_accuracy = [r["agent"]["judge_scores"].get("accuracy", 0.0) for r in results if r["agent"]["success"] and r["agent"]["judge_success"]]

        overall_metrics = {
            "agent": {
                "avg_overall_score": sum(agent_overall_scores) / len(agent_overall_scores) if agent_overall_scores else 0.0,
                "avg_product_relevance": sum(agent_product_relevance) / len(agent_product_relevance) if agent_product_relevance else 0.0,
                "avg_info_completeness": sum(agent_info_completeness) / len(agent_info_completeness) if agent_info_completeness else 0.0,
                "avg_response_structure": sum(agent_response_structure) / len(agent_response_structure) if agent_response_structure else 0.0,
                "avg_customer_service": sum(agent_customer_service) / len(agent_customer_service) if agent_customer_service else 0.0,
                "avg_accuracy": sum(agent_accuracy) / len(agent_accuracy) if agent_accuracy else 0.0,
                "baseline_success_rate": sum(agent_meets_baseline) / len(agent_meets_baseline) if agent_meets_baseline else 0.0,
                "avg_response_time": sum(agent_times) / len(agent_times) if agent_times else 0.0,
                "success_rate": len([r for r in results if r["agent"]["success"]]) / len(results),
                "judge_success_rate": len([r for r in results if r["agent"]["judge_success"]]) / len(results),
                "total_tests": len(results)
            }
        }

        return {
            "overall_metrics": overall_metrics,
            "individual_results": results
        }

    def display_results(self, results: Dict[str, Any]):
        """Display evaluation results"""
        overall = results["overall_metrics"]["agent"]

        console.print("\n[bold green]Agent Evaluation Results (LLM Judge)[/bold green]")

        # Overall metrics table
        summary_table = Table(title="Agent Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Score", style="green")
        summary_table.add_column("Status", style="yellow")

        # Overall Score
        overall_status = "✅ Excellent" if overall["avg_overall_score"] > 0.8 else "✅ Good" if overall["avg_overall_score"] > 0.7 else "⚠️ Needs Improvement"
        summary_table.add_row("Overall Score", f"{overall['avg_overall_score']:.3f}", overall_status)

        # Individual metrics
        summary_table.add_row("Product Relevance", f"{overall['avg_product_relevance']:.3f}", "✅" if overall['avg_product_relevance'] > 0.7 else "⚠️")
        summary_table.add_row("Info Completeness", f"{overall['avg_info_completeness']:.3f}", "✅" if overall['avg_info_completeness'] > 0.7 else "⚠️")
        summary_table.add_row("Response Structure", f"{overall['avg_response_structure']:.3f}", "✅" if overall['avg_response_structure'] > 0.7 else "⚠️")
        summary_table.add_row("Customer Service", f"{overall['avg_customer_service']:.3f}", "✅" if overall['avg_customer_service'] > 0.7 else "⚠️")
        summary_table.add_row("Accuracy", f"{overall['avg_accuracy']:.3f}", "✅" if overall['avg_accuracy'] > 0.7 else "⚠️")

        # Performance metrics
        summary_table.add_row("Baseline Success Rate", f"{overall['baseline_success_rate']:.1%}", "✅" if overall['baseline_success_rate'] > 0.8 else "⚠️")
        summary_table.add_row("Response Time", f"{overall['avg_response_time']:.2f}s", "✅" if overall['avg_response_time'] < 5.0 else "⚠️")
        summary_table.add_row("Success Rate", f"{overall['success_rate']:.1%}", "✅" if overall['success_rate'] > 0.9 else "⚠️")
        summary_table.add_row("Judge Success Rate", f"{overall['judge_success_rate']:.1%}", "✅" if overall['judge_success_rate'] > 0.9 else "⚠️")

        console.print(summary_table)

        # Individual results table
        individual_table = Table(title="Individual Test Results")
        individual_table.add_column("Question", style="cyan", max_width=40)
        individual_table.add_column("Overall Score", style="green")
        individual_table.add_column("Relevance", style="green")
        individual_table.add_column("Structure", style="green")
        individual_table.add_column("Baseline", style="green")
        individual_table.add_column("Time (s)", style="blue")

        for result in results["individual_results"]:
            question_short = result["question"][:37] + "..." if len(result["question"]) > 40 else result["question"]
            meets_baseline = "✅" if result['agent']['meets_baseline'] else "❌"
            individual_table.add_row(
                question_short,
                f"{result['agent']['overall_score']:.3f}",
                f"{result['agent']['judge_scores'].get('product_relevance', 0.0):.3f}",
                f"{result['agent']['judge_scores'].get('response_structure', 0.0):.3f}",
                meets_baseline,
                f"{result['agent']['response_time']:.2f}"
            )

        console.print(individual_table)

        # Analysis and recommendations
        console.print("\n[bold yellow]Analysis:[/bold yellow]")

        # Overall performance
        if overall["avg_overall_score"] > 0.8:
            console.print("• [green]Agent shows excellent overall performance! 🎉[/green]")
        elif overall["avg_overall_score"] > 0.7:
            console.print("• [green]Agent meets baseline performance standards[/green]")
        else:
            console.print("• [yellow]Agent performance needs improvement[/yellow]")

        # Baseline success
        if overall["baseline_success_rate"] > 0.8:
            console.print("• [green]Agent consistently meets quality standards (≥0.7)[/green]")
        else:
            console.print("• [yellow]Agent needs to improve consistency[/yellow]")

        # Response time
        if overall["avg_response_time"] < 5.0:
            console.print("• [green]Agent response times are acceptable[/green]")
        else:
            console.print("• [yellow]Agent response times need optimization[/yellow]")

        # Specific recommendations
        if overall["avg_product_relevance"] < 0.7:
            console.print("• [yellow]Improve product relevance in responses[/yellow]")
        if overall["avg_info_completeness"] < 0.7:
            console.print("• [yellow]Provide more complete product information[/yellow]")
        if overall["avg_response_structure"] < 0.7:
            console.print("• [yellow]Improve response structure and formatting[/yellow]")
        if overall["avg_customer_service"] < 0.7:
            console.print("• [yellow]Enhance customer service quality[/yellow]")
        if overall["avg_accuracy"] < 0.7:
            console.print("• [yellow]Improve accuracy of product details[/yellow]")

        # Final recommendation
        if all([
            overall["avg_overall_score"] > 0.7,
            overall["baseline_success_rate"] > 0.8,
            overall["avg_response_time"] < 10.0
        ]):
            console.print("\n• [green]Agent is ready for production! 🚀[/green]")
        else:
            console.print("\n• [yellow]Agent needs optimization before production[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="RAG Agent Evaluation with LLM Judge")
    parser.add_argument("--chroma-dir", default=".chroma", help="Chroma database directory")
    parser.add_argument("--collection", default="products", help="Chroma collection name")
    parser.add_argument("--csv", default="data/test_questions.csv", help="CSV file with test questions")
    parser.add_argument("--max-tests", type=int, help="Maximum number of tests to run")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AgentEvaluator(args.chroma_dir, args.collection)

    # Run evaluation
    results = evaluator.evaluate_agent(args.csv, args.max_tests)

    # Display results
    evaluator.display_results(results)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {args.output}[/green]")

if __name__ == "__main__":
    main()
