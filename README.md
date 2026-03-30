# langgraph-legal-assistant

LangGraph companion to [adk-legal-assistant](https://github.com/gbhorne/legal-adk-gcp). The same contract ReviewAgent implemented as a LangGraph StateGraph for framework comparison.

**Primary repo:** [adk-legal-assistant](https://github.com/gbhorne/legal-adk-gcp) (Google ADK implementation)

---

## What this repo demonstrates

The ReviewAgent from adk-legal-assistant rebuilt as an explicit LangGraph StateGraph with four nodes:
```
tokenize --> extract_clauses --> rate_clauses --> compile_report --> END
```

Same inputs, same outputs, same RAG corpus (Vertex AI Search, 1,010+ Georgia court opinions), same Cloud DLP tokenization. Different framework.

---

## ADK vs LangGraph comparison

| Dimension | Google ADK | LangGraph |
|-----------|-----------|-----------|
| Agent definition | Declarative (Agent + FunctionTool) | Explicit StateGraph nodes |
| Multi-agent routing | Built-in sub_agents | Manual conditional edges |
| State management | Implicit session state | Explicit TypedDict state |
| Debugging | ADK web UI with trace panel | LangGraph Studio / LangSmith |
| Gemini integration | Native google-genai | Via langchain-google-genai |
| Boilerplate | Low | Higher but more explicit |
| Best for | GCP-native production deployments | Complex branching, auditability |

**Key finding from this implementation:** LangGraph requires more explicit state management but makes every state transition visible and testable. For regulated environments where an auditor needs to see exactly what happened at each step, the explicit graph is a stronger compliance story. ADK ships faster for straightforward tool-calling pipelines.

---

## Graph structure
```
ReviewState (TypedDict)
  contract_text   str        # raw input
  jurisdiction    str        # e.g. "Georgia"
  contract_name   str
  clean_text      str        # after PII tokenization
  dlp_context     dict       # token-to-original mapping
  raw_clauses     list       # extracted by LLM
  analyzed        list       # ClauseAnalysis objects (reducer: operator.add)
  report          dict       # final ContractRiskReport
```

Nodes:
- **tokenize:** Cloud DLP inspect API tokenizes PII across seven infoTypes; local regex fallback if API unavailable
- **extract_clauses:** Gemini extracts clause_type + clause_text as JSON array
- **rate_clauses:** per-clause RAG lookup + Gemini risk rating
- **compile_report:** assembles ContractRiskReport Pydantic object

---

## Quickstart
```powershell
git clone https://github.com/gbhorne/langgraph-legal-assistant
cd langgraph-legal-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# add your GOOGLE_API_KEY to .env
python tests\test_review_graph.py
```

---

## Project structure
```
langgraph-legal-assistant/
+-- agents/
|   +-- review_graph.py    # LangGraph StateGraph ReviewAgent
|   +-- rag.py             # Vertex AI Search query helper (shared)
|   +-- schemas.py         # Pydantic output types (shared)
+-- dlp/
|   +-- tokenizer.py       # Cloud DLP PII tokenization with local regex fallback (shared)
+-- tests/
|   +-- test_review_graph.py
+-- config.py
+-- requirements.txt
```

---

## Experimental disclaimer

This project is experimental software intended for portfolio demonstration and research purposes only. It has not been validated for production legal use. All outputs require review by a licensed attorney before use in any legal matter. Outputs from this system do not constitute legal advice and do not create an attorney-client relationship.

---

*Companion to [adk-legal-assistant](https://github.com/gbhorne/legal-adk-gcp)*
