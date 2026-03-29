from enum import Enum
from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    INDEMNIFICATION      = "indemnification"
    IP_ASSIGNMENT        = "ip_assignment"
    NON_COMPETE          = "non_compete"
    NON_SOLICITATION     = "non_solicitation"
    CONFIDENTIALITY      = "confidentiality"
    LIMITATION_LIABILITY = "limitation_of_liability"
    TERMINATION          = "termination"
    GOVERNING_LAW        = "governing_law"
    DISPUTE_RESOLUTION   = "dispute_resolution"
    PAYMENT_TERMS        = "payment_terms"
    WARRANTY             = "warranty"
    FORCE_MAJEURE        = "force_majeure"
    ASSIGNMENT           = "assignment"
    MISCELLANEOUS        = "miscellaneous"


class RiskLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    INFO   = "info"


class LegalAuthority(BaseModel):
    case_name:      str
    citation:       str
    court:          str
    year:           str | None = None
    source_url:     str | None = None
    relevance_note: str


class ClauseAnalysis(BaseModel):
    clause_type:       ClauseType
    clause_text:       str
    risk_level:        RiskLevel
    risk_summary:      str
    risk_basis:        str
    fallback_language: str | None = None
    citations:         list[LegalAuthority] = Field(default_factory=list)


class ContractRiskReport(BaseModel):
    contract_name:            str
    jurisdiction:             str
    contract_type:            str | None = None
    overall_risk_level:       RiskLevel
    overall_summary:          str
    clauses:                  list[ClauseAnalysis]
    attorney_review_required: bool = True
    attorney_review_note:     str  = (
        "AI-generated analysis. Must be reviewed by a licensed attorney before use. "
        "Does not constitute legal advice. ABA Model Rule 1.6 applies."
    )


class LegalResearchMemo(BaseModel):
    question:                 str
    jurisdiction:             str | None = None
    answer:                   str
    supporting_analysis:      str
    authorities:              list[LegalAuthority]
    jurisdiction_warnings:    list[str] = Field(default_factory=list)
    related_questions:        list[str] = Field(default_factory=list)
    attorney_review_required: bool = True


class DraftedDocument(BaseModel):
    contract_type:              str
    jurisdiction:               str
    parties:                    dict[str, str]
    markdown_text:              str
    drafting_notes:             list[str]
    attorney_authored_required: bool = True
    attorney_authored_note:     str  = (
        "First-pass AI draft. Attorney must review and take full responsibility "
        "before use with any client."
    )
