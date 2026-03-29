import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from typing import TypedDict, Annotated
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents.rag import query_corpus
from agents.schemas import (
    ClauseAnalysis, ClauseType, ContractRiskReport,
    LegalAuthority, RiskLevel,
)
from dlp.tokenizer import new_context, tokenize, detokenize
from config import config

log = logging.getLogger("agents.review_graph")

llm = ChatGoogleGenerativeAI(
    model=config.GEMINI_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    temperature=0,
)


# ── State ─────────────────────────────────────────────────────────────────────

class ReviewState(TypedDict):
    contract_text:  str
    jurisdiction:   str
    contract_name:  str
    clean_text:     str
    dlp_context:    dict
    raw_clauses:    list
    analyzed:       Annotated[list, operator.add]
    report:         dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

def tokenize_node(state: ReviewState) -> dict:
    log.info("Node: tokenize")
    ctx = new_context()
    clean = tokenize(state["contract_text"], ctx)
    return {"clean_text": clean, "dlp_context": ctx}


def extract_clauses_node(state: ReviewState) -> dict:
    log.info("Node: extract_clauses")
    clause_types = ", ".join(ct.value for ct in ClauseType)
    prompt = (
        "Extract all significant clauses from this contract as a JSON array.\n"
        "Each element must have clause_type (one of: " + clause_types + ") and clause_text.\n"
        "Return ONLY valid JSON.\n\nContract:\n" + state["clean_text"][:12000]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        try:
            clauses = json.loads(raw[start:end])
            log.info("Extracted %d clauses", len(clauses))
            return {"raw_clauses": clauses}
        except json.JSONDecodeError:
            pass
    log.warning("Clause extraction failed")
    return {"raw_clauses": []}


def rate_clause_node(state: ReviewState) -> dict:
    log.info("Node: rate_clauses (%d clauses)", len(state["raw_clauses"]))
    analyzed = []
    risk_order = {"high": 3, "medium": 2, "low": 1, "info": 0}

    for raw in state["raw_clauses"]:
        ct_str = raw.get("clause_type", "miscellaneous")
        ct_text = raw.get("clause_text", "").strip()
        if not ct_text:
            continue

        try:
            clause_type = ClauseType(ct_str)
        except ValueError:
            clause_type = ClauseType.MISCELLANEOUS

        rag = query_corpus(ct_str + " " + state["jurisdiction"] + " enforceability", max_results=3)
        case_law = "\n\n".join(
            "[" + r["case_name"] + " | " + r["court_id"] + " | " + r["date_filed"] + "]\n" + r["text"]
            for r in rag
        ) or "No relevant case law retrieved."

        prompt = (
            "Rate the legal risk of this clause for the signing party.\n\n"
            "Clause type: " + ct_str + "\n"
            "Jurisdiction: " + state["jurisdiction"] + "\n"
            "Clause text: " + ct_text[:2000] + "\n\n"
            "Case law:\n" + case_law[:3000] + "\n\n"
            "Return JSON: {risk_level, risk_summary, risk_basis, fallback_language}\n"
            "risk_level must be: high, medium, low, or info\n"
            "Return ONLY valid JSON."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_resp = response.content.strip()
        start = raw_resp.find("{")
        end = raw_resp.rfind("}") + 1
        rating = {}
        if start != -1 and end > start:
            try:
                rating = json.loads(raw_resp[start:end])
            except json.JSONDecodeError:
                pass

        try:
            risk_level = RiskLevel(rating.get("risk_level", "medium"))
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        risk_basis = rating.get("risk_basis", "")
        if isinstance(risk_basis, list):
            risk_basis = " ".join(risk_basis)

        risk_summary = rating.get("risk_summary", "")
        if isinstance(risk_summary, list):
            risk_summary = " ".join(risk_summary)

        fallback = rating.get("fallback_language")
        if isinstance(fallback, list):
            fallback = " ".join(fallback)

        citations = [
            LegalAuthority(
                case_name=r["case_name"],
                citation=r["citation"] or "",
                court=r["court_id"],
                year=r["date_filed"][:4] if r.get("date_filed") else None,
                source_url=r["source_url"],
                relevance_note="Relevant to " + ct_str + " in " + state["jurisdiction"],
            )
            for r in rag if r.get("case_name") and r["case_name"] != "Unknown"
        ]

        analyzed.append(ClauseAnalysis(
            clause_type=clause_type,
            clause_text=detokenize(ct_text, state["dlp_context"]),
            risk_level=risk_level,
            risk_summary=risk_summary,
            risk_basis=risk_basis,
            fallback_language=fallback,
            citations=citations,
        ))

    return {"analyzed": analyzed}


def compile_report_node(state: ReviewState) -> dict:
    log.info("Node: compile_report")
    analyzed = state["analyzed"]
    risk_order = {"high": 3, "medium": 2, "low": 1, "info": 0}

    overall = max(analyzed, key=lambda c: risk_order.get(c.risk_level.value, 0)).risk_level if analyzed else RiskLevel.INFO
    high = sum(1 for c in analyzed if c.risk_level == RiskLevel.HIGH)
    med  = sum(1 for c in analyzed if c.risk_level == RiskLevel.MEDIUM)

    report = ContractRiskReport(
        contract_name=state.get("contract_name", "Contract"),
        jurisdiction=state["jurisdiction"],
        overall_risk_level=overall,
        overall_summary=str(len(analyzed)) + " clauses analyzed. " + str(high) + " high-risk, " + str(med) + " medium-risk.",
        clauses=analyzed,
    )
    return {"report": json.loads(report.model_dump_json())}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_review_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("tokenize",        tokenize_node)
    graph.add_node("extract_clauses", extract_clauses_node)
    graph.add_node("rate_clauses",    rate_clause_node)
    graph.add_node("compile_report",  compile_report_node)

    graph.set_entry_point("tokenize")
    graph.add_edge("tokenize",        "extract_clauses")
    graph.add_edge("extract_clauses", "rate_clauses")
    graph.add_edge("rate_clauses",    "compile_report")
    graph.add_edge("compile_report",  END)

    return graph.compile()


review_graph = build_review_graph()


def analyze_contract(contract_text: str, jurisdiction: str, contract_name: str = "Contract") -> ContractRiskReport:
    result = review_graph.invoke({
        "contract_text": contract_text,
        "jurisdiction":  jurisdiction,
        "contract_name": contract_name,
        "clean_text":    "",
        "dlp_context":   {},
        "raw_clauses":   [],
        "analyzed":      [],
        "report":        {},
    })
    return ContractRiskReport(**result["report"])
