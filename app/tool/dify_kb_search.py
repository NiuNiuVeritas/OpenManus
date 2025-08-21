# -*- coding: utf-8 -*-
# Dify Knowledge Base Search Tool
import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DifySearchItem:
    """Represents a single search result from Dify Knowledge Base."""
    id: str
    position: int
    document_id: str
    content: str
    answer: Optional[str]
    word_count: int
    tokens: int
    keywords: List[str]
    score: float
    document_name: str
    document_type: str

@dataclass
class DifySearchMetadata:
    """Metadata about the search operation."""
    total_results: int
    query: str
    dataset_id: str

@dataclass
class DifySearchResponse:
    """Structured response from Dify Knowledge Base search."""
    status: str
    query: str
    results: List[DifySearchItem]
    metadata: DifySearchMetadata
    error: Optional[str] = None

    def to_tool_result(self) -> Dict[str, Any]:
        """Convert to tool result format."""
        if self.error:
            return {
                "status": "error",
                "error": self.error,
                "query": self.query
            }
        
        return {
            "status": self.status,
            "query": self.query,
            "results": [
                {
                    "id": item.id,
                    "position": item.position,
                    "content": item.content,
                    "answer": item.answer,
                    "score": item.score,
                    "document_name": item.document_name,
                    "document_type": item.document_type,
                    "keywords": item.keywords,
                    "word_count": item.word_count,
                    "tokens": item.tokens
                }
                for item in self.results
            ],
            "metadata": {
                "total_results": self.metadata.total_results,
                "query": self.metadata.query,
                "dataset_id": self.metadata.dataset_id
            }
        }


class DifyKnowledgeBaseSearch:
    """Search engine for Dify Knowledge Base."""
    
    def __init__(self, api_key: str, dataset_id: str, base_url: str = "https://api.dify.ai/v1"):
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.base_url = base_url
        self.endpoint = f"{base_url}/datasets/{dataset_id}/retrieve"
        
    async def perform_search(
        self, 
        query: str, 
        top_k: int = 5,
        search_method: str = "keyword_search",
        reranking_enable: bool = False,
        score_threshold_enabled: bool = False,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[DifySearchItem]:
        """Perform search in Dify Knowledge Base."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": search_method,
                "reranking_enable": reranking_enable,
                "reranking_mode": None,
                "reranking_model": {
                    "reranking_provider_name": "",
                    "reranking_model_name": ""
                },
                "weights": None,
                "top_k": top_k,
                "score_threshold_enabled": score_threshold_enabled,
                "score_threshold": score_threshold,
                "metadata_filtering_conditions": metadata_filter or {
                    "logical_operator": "and",
                    "conditions": []
                }
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            return self._parse_search_results(data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON response: {response_text}")
                            return []
                    else:
                        logger.error(f"Dify API error: {response.status} - {response_text}")
                        return []
        except Exception as e:
            logger.error(f"Error performing Dify search: {e}")
            return []
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[DifySearchItem]:
        """Parse the response from Dify API into structured results."""
        results = []
        
        if "records" not in data:
            return results
            
        for record in data["records"]:
            if "segment" in record:
                segment = record["segment"]
                document = segment.get("document", {})
                
                item = DifySearchItem(
                    id=segment.get("id", ""),
                    position=segment.get("position", 0),
                    document_id=segment.get("document_id", ""),
                    content=segment.get("content", ""),
                    answer=segment.get("answer"),
                    word_count=segment.get("word_count", 0),
                    tokens=segment.get("tokens", 0),
                    keywords=segment.get("keywords", []),
                    score=record.get("score", 0.0),
                    document_name=document.get("name", ""),
                    document_type=document.get("data_source_type", "")
                )
                results.append(item)
        
        return results


class DifyKBSearch:
    """Search Dify Knowledge Base for information."""

    name: str = "dify_kb_search"
    description: str = """Search the Dify Knowledge Base for relevant information.
    This tool searches through the configured knowledge base and returns relevant documents,
    segments, and metadata based on the query."""
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the knowledge base.",
            },
            "top_k": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 5.",
                "default": 5,
            },
            "search_method": {
                "type": "string",
                "description": "(optional) Search method: 'keyword_search' or 'semantic_search'. Default is 'keyword_search'.",
                "default": "keyword_search",
                "enum": ["keyword_search", "semantic_search"]
            },
            "reranking_enable": {
                "type": "boolean",
                "description": "(optional) Whether to enable reranking. Default is false.",
                "default": False,
            },
            "score_threshold_enabled": {
                "type": "boolean",
                "description": "(optional) Whether to enable score threshold filtering. Default is false.",
                "default": False,
            },
            "score_threshold": {
                "type": "number",
                "description": "(optional) Score threshold for filtering results (0.0 to 1.0).",
            },
            "metadata_filter": {
                "type": "object",
                "description": "(optional) Metadata filtering conditions for the search.",
            }
        },
        "required": ["query"],
    }
    
    def __init__(self, api_key: str, dataset_id: str, base_url: str = "https://api.dify.ai/v1"):
        self.search_engine = DifyKnowledgeBaseSearch(api_key, dataset_id, base_url)
        self.dataset_id = dataset_id

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        search_method: str = "keyword_search",
        reranking_enable: bool = False,
        score_threshold_enabled: bool = False,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> DifySearchResponse:
        """
        Execute a Dify Knowledge Base search and return detailed search results.

        Args:
            query: The search query to submit to the knowledge base
            top_k: The number of search results to return (default: 5)
            search_method: Search method to use (default: keyword_search)
            reranking_enable: Whether to enable reranking (default: False)
            score_threshold_enabled: Whether to enable score threshold filtering (default: False)
            score_threshold: Score threshold for filtering results (optional)
            metadata_filter: Metadata filtering conditions (optional)

        Returns:
            A structured response containing search results and metadata
        """
        
        try:
            logger.info(f"🔍 Searching Dify Knowledge Base for: {query}")
            
            results = await self._perform_search_with_retry(
                query, top_k, search_method, reranking_enable, 
                score_threshold_enabled, score_threshold, metadata_filter
            )
            
            if results:
                return DifySearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    metadata=DifySearchMetadata(
                        total_results=len(results),
                        query=query,
                        dataset_id=self.dataset_id
                    )
                )
            else:
                return DifySearchResponse(
                    status="no_results",
                    query=query,
                    results=[],
                    metadata=DifySearchMetadata(
                        total_results=0,
                        query=query,
                        dataset_id=self.dataset_id
                    ),
                    error="No results found for the query."
                )
                
        except Exception as e:
            logger.error(f"Error in Dify KB search: {e}")
            return DifySearchResponse(
                status="error",
                query=query,
                results=[],
                metadata=DifySearchMetadata(
                    total_results=0,
                    query=query,
                    dataset_id=self.dataset_id
                ),
                error=f"Search failed: {str(e)}"
            )

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _perform_search_with_retry(
        self,
        query: str,
        top_k: int,
        search_method: str,
        reranking_enable: bool,
        score_threshold_enabled: bool,
        score_threshold: Optional[float],
        metadata_filter: Optional[Dict[str, Any]]
    ) -> List[DifySearchItem]:
        """Execute search with retry logic."""
        return await self.search_engine.perform_search(
            query=query,
            top_k=top_k,
            search_method=search_method,
            reranking_enable=reranking_enable,
            score_threshold_enabled=score_threshold_enabled,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    API_KEY = "dataset-1yBGLBKl2ud8zU2tvenpQo4B"
    DATASET_ID = "your_dataset_id_here"  # Replace with actual dataset ID
    
    async def test_dify_search():
        """Test the Dify Knowledge Base search functionality."""
        dify_search = DifyKBSearch(API_KEY, DATASET_ID)
        
        # Test search
        search_response = await dify_search.execute(
            query="test",
            top_k=3,
            search_method="keyword_search",
            reranking_enable=False
        )
        
        print("Search Response:")
        print(search_response.to_tool_result())
        
        # Test with different parameters
        semantic_response = await dify_search.execute(
            query="operation guide",
            top_k=2,
            search_method="semantic_search",
            reranking_enable=True
        )
        
        print("\nSemantic Search Response:")
        print(semantic_response.to_tool_result())
    
    # Run the test
    asyncio.run(test_dify_search())
