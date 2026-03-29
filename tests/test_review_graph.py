import sys, os
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from agents.review_graph import analyze_contract

contract = """
NON-COMPETE AGREEMENT

1. NON-COMPETE
Employee agrees not to compete with Employer for five years anywhere in the United States.

2. CONFIDENTIALITY
All company information shall remain confidential indefinitely.

3. LIMITATION OF LIABILITY
Total liability shall not exceed five hundred dollars.

4. GOVERNING LAW
This Agreement is governed by the laws of Georgia.
"""

report = analyze_contract(contract, "Georgia", "Test NDA")
print()
print("=" * 60)
print("CONTRACT:", report.contract_name)
print("OVERALL RISK:", report.overall_risk_level.value.upper())
print("SUMMARY:", report.overall_summary)
print()
for clause in report.clauses:
    print("-" * 40)
    print("CLAUSE:", clause.clause_type.value)
    print("RISK:  ", clause.risk_level.value.upper())
    print("WHY:   ", clause.risk_summary[:100])
    print("CITES: ", len(clause.citations), "cases")
print()
print("ATTORNEY REVIEW REQUIRED:", report.attorney_review_required)
