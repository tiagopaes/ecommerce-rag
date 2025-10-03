#!/usr/bin/env python3
"""
RAG Agent with LangChain
Uses RAG as a tool with metadata filtering for better results
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

# Config
load_dotenv()
logging.basicConfig(level=logging.WARNING)

class ProductSearchInput(BaseModel):
    """Input for product search tool"""
    query: str = Field(description="The search query for products")
    category: Optional[str] = Field(default=None, description="Filter by product category (smartphone, laptop, tablet, wearable, accessory)")
    brand: Optional[str] = Field(default=None, description="Filter by brand (Samsung, Google, Apple, Dell)")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter")
    min_price: Optional[float] = Field(default=None, description="Minimum price filter")
    storage: Optional[str] = Field(default=None, description="Storage capacity filter (e.g., '256GB', '512GB', '1TB')")
    screen_size: Optional[str] = Field(default=None, description="Screen size filter (e.g., '6.8', '13.4', '10.9')")
    camera: Optional[str] = Field(default=None, description="Camera specification filter (e.g., '200MP', '64MP', '48MP')")
    processor: Optional[str] = Field(default=None, description="Processor filter (e.g., 'Snapdragon 8 Gen 2', 'Intel i7', 'M1')")
    k: int = Field(default=5, description="Number of results to return (default: 5)")

class ProductSearchTool(BaseTool):
    """Tool for searching products with metadata filtering"""

    name: str = "product_search"
    description: str = """
    Search for products in the e-commerce database with advanced filtering capabilities.
    Use this tool to find products based on query and various metadata filters.

    Available filters:
    - category: smartphone, laptop, tablet, wearable, accessory
    - brand: Samsung, Google, Apple, Dell
    - max_price/min_price: price range filtering
    - storage: storage capacity (256GB, 512GB, 1TB, etc.)
    - screen_size: screen size in inches
    - camera: camera specifications
    - processor: processor type
    - k: number of results (default: 5)

    Always use appropriate filters to narrow down results and improve relevance.
    """
    args_schema: Type[BaseModel] = ProductSearchInput
    vectorstore: Chroma = Field(default=None, exclude=True)

    def __init__(self, vectorstore: Chroma, **kwargs):
        super().__init__(vectorstore=vectorstore, **kwargs)

    def _run(self, **kwargs) -> str:
        """Execute the product search with filtering"""
        try:
            # Extract parameters
            query = kwargs.get("query", "")
            category = kwargs.get("category")
            brand = kwargs.get("brand")
            max_price = kwargs.get("max_price")
            min_price = kwargs.get("min_price")
            storage = kwargs.get("storage")
            screen_size = kwargs.get("screen_size")
            camera = kwargs.get("camera")
            processor = kwargs.get("processor")
            k = kwargs.get("k", 5)

            # Perform similarity search first
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k * 2  # Get more results to filter from
            )

            # Apply all filters
            filtered_results = []
            for doc, score in results:
                # Apply price filters
                product_price = float(doc.metadata.get("price", 0))
                if max_price is not None and product_price > max_price:
                    continue
                if min_price is not None and product_price < min_price:
                    continue

                # Apply metadata filters
                if category and doc.metadata.get("category") != category:
                    continue
                if brand and doc.metadata.get("brand") != brand:
                    continue
                if storage and doc.metadata.get("storage") != storage:
                    continue
                if screen_size and doc.metadata.get("screen_size") != screen_size:
                    continue
                if camera and doc.metadata.get("camera") != camera:
                    continue
                if processor and doc.metadata.get("processor") != processor:
                    continue

                filtered_results.append((doc, score))

                # Stop when we have enough results
                if len(filtered_results) >= k:
                    break

            # Format results
            if not filtered_results:
                return "No products found matching the criteria."

            formatted_results = []
            for doc, score in filtered_results:
                product_info = {
                    "id": doc.metadata["id"],
                    "title": doc.metadata["title"],
                    "category": doc.metadata["category"],
                    "brand": doc.metadata["brand"],
                    "price": doc.metadata["price"],
                    "similarity_score": score,
                    "description": doc.page_content
                }
                formatted_results.append(product_info)

            return json.dumps({
                "products": formatted_results,
                "total_found": len(formatted_results),
                "query": query,
                "filters_applied": {
                    "category": category,
                    "brand": brand,
                    "max_price": max_price,
                    "min_price": min_price,
                    "storage": storage,
                    "screen_size": screen_size,
                    "camera": camera,
                    "processor": processor
                }
            }, indent=2)

        except Exception as e:
            return f"Error searching products: {str(e)}"

class RAGAgent:
    """RAG Agent that uses product search as a tool"""

    def __init__(self, chroma_dir: str = ".chroma", collection_name: str = "products"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.1,
            max_tokens=2000
        )

        # Create product search tool
        self.search_tool = ProductSearchTool(self.vectorstore)

        # Create agent
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with tools and prompt"""

        # System prompt for the agent
        system_prompt = """You are an intelligent e-commerce product search assistant.
        Your job is to help users find the best products based on their queries.

        You have access to a product search tool that can filter products by:
        - Category (smartphone, laptop, tablet, wearable, accessory)
        - Brand (Samsung, Google, Apple, Dell)
        - Price range (min_price, max_price)
        - Storage capacity
        - Screen size
        - Camera specifications
        - Processor type

        When a user asks for products:
        1. Analyze the query to understand what they're looking for
        2. Extract relevant filters from the query (brand, category, price, specifications)
        3. Use the product_search tool with appropriate filters
        4. Present the results in a clear, organized way

        Always try to apply relevant filters to get more precise results.
        If the user mentions specific brands, categories, or specifications, use them as filters.
        If they mention price ranges, apply price filters.

        Be helpful and provide detailed information about the products you find.
        """

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=[self.search_tool],
            prompt=prompt
        )

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.search_tool],
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )

        return agent_executor

    def search_products(self, query: str) -> Dict[str, Any]:
        """Search for products using the agent"""
        try:
            # Run the agent
            result = self.agent.invoke({"input": query})

            # Parse the result
            response = result.get("output", "")

            # Try to extract structured data from the response
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    structured_data = json.loads(json_match.group())
                    return {
                        "success": True,
                        "response": response,
                        "structured_data": structured_data
                    }
                except json.JSONDecodeError:
                    pass

            return {
                "success": True,
                "response": response,
                "structured_data": None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }

    def get_products_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract product list from agent response"""
        if not response.get("success"):
            return []

        structured_data = response.get("structured_data")
        if structured_data and "products" in structured_data:
            return structured_data["products"]

        return []

def main():
    """Run the RAG agent as a CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    parser.add_argument("--chroma-dir", default=".chroma", help="Chroma database directory")
    parser.add_argument("--collection", default="products", help="Chroma collection name")
    parser.add_argument("--query", required=True, help="Query to search for")

    args = parser.parse_args()

    # Initialize agent
    agent = RAGAgent(args.chroma_dir, args.collection)

    # Test query
    print(f"Query: {args.query}")
    print("-" * 50)

    result = agent.search_products(args.query)

    if result["success"]:
        print("Response:")
        print(result["response"])

        products = agent.get_products_from_response(result)
        if products:
            print(f"\nFound {len(products)} products:")
            for i, product in enumerate(products, 1):
                print(f"{i}. {product['title']} - ${product['price']} ({product['brand']})")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
