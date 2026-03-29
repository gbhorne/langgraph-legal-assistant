import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions
from config import config

log = logging.getLogger("agents.rag")


def query_corpus(query, max_results=5):
    """
    Query the Vertex AI Search corpus for relevant case law.

    Returns a list of dicts with keys:
        case_name, court_id, date_filed, citation, source_url, text

    Empty list if search fails or returns nothing.
    """
    try:
        client = discoveryengine.SearchServiceClient(
            client_options=ClientOptions(
                api_endpoint="us-discoveryengine.googleapis.com"
            )
        )
        response = client.search(
            discoveryengine.SearchRequest(
                serving_config=config.SEARCH_SERVING_CONFIG,
                query=query,
                page_size=max_results,
            )
        )
        results = []
        for r in response.results:
            s = dict(r.document.struct_data)
            results.append({
                "case_name":  s.get("case_name",  "Unknown"),
                "court_id":   s.get("court_id",   ""),
                "date_filed": s.get("date_filed",  ""),
                "citation":   s.get("citation",    ""),
                "source_url": s.get("source_url",  ""),
                "text":       s.get("text",        ""),
            })
        return results
    except Exception as e:
        log.error("RAG query failed: %s", e)
        return []
