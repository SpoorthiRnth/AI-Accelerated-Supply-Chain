#!/usr/bin/env python3
# run_demo.py
# Runs the full pipeline with a mock LLM that returns realistic pre-defined responses.
# Use this to demo outputs without an Azure OpenAI API key.
# Replace mock_llm_response() with real call_llm() for production.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)

from synthetic_data.generator import load_all_data
from config import OUTPUT_DIR, DQ_SCORE_THRESHOLD, ANOMALY_VOLUME_THRESHOLD, PROMOTION_THRESHOLD


# ── Mock LLM responses keyed by agent name ──────────────────────────────────

MOCK_RESPONSES = {
    "Document & Unstructured Data Agent": {
        "normalized_record": {},
        "extraction_quality": 97,
        "fields_extracted": 12,
        "issues_found": [],
        "document_category": "retailer_scorecard",
    },
    "Data Quality Gate Agent": {
        "overall_health": "Fair",
        "overall_health_score": 78,
        "critical_issues": ["PO-10009 missing destination warehouse — quarantined"],
        "recommended_actions": ["Review ERP data entry process for warehouse assignment", "Add mandatory field validation at source"],
        "narrative": "Overall data quality is fair with a 90% pass rate. One purchase order record was quarantined due to a missing destination warehouse. Corrective action at the ERP entry level is recommended to prevent recurrence.",
    },
    "Source Anomaly Detection Agent": {
        "overall_feed_health": "Degraded",
        "high_priority_actions": ["Escalate Port of Singapore disruption to logistics team immediately", "Monitor Asia-US West Coast shipments for delay propagation"],
        "risk_exposure_summary": "Active port disruption at Singapore and geopolitical risk in Shenzhen affect Asia-US West Coast routes. Two shipments currently flagged with high risk. Estimated combined exposure exceeds $300,000.",
        "affected_shipment_count": 2,
        "estimated_financial_exposure_usd": 312000,
    },
    "Reconciliation Agent": {
        "reconciled_status": "In Transit",
        "resolution_rationale": "TMS is the authoritative source for physical shipment status. ERP status 'Pending Dispatch' is stale and requires correction. Reconciled status set to TMS value 'In Transit'.",
        "erp_correction_needed": True,
        "confidence": 95,
    },
    "Semantic Enrichment Agent": {
        "business_summary": "Shipment of Winter Jackets (Premium) from Chicago Midwest DC to NorthStar Retail Group via FastFreight Logistics. High-priority retail account with seasonal demand peak active.",
        "risk_level": "High",
        "business_priority": "Critical",
        "key_context_flags": ["High-priority retail account", "Seasonal demand peak active", "Carrier performance declining on Northeast corridor"],
        "sla_at_risk": True,
        "enrichment_tags": ["seasonal", "high-priority-retail", "sla-watch", "northeast-corridor"],
    },
    "Data Classification & Promotion Agent": {
        "gold_layer_health": "Good",
        "promotion_rate_pct": 87.5,
        "summary": "87.5% of enriched records met all promotion criteria and were elevated to the Gold layer. Held records primarily lack complete carrier enrichment. The Gold layer is in good health with high data completeness.",
        "data_gaps_identified": ["Some records missing carrier enrichment", "Two records with incomplete retailer linkage"],
    },
    "Demand & Supply Signal Agent": {
        "overall_supply_health": "At Risk",
        "stockout_risks": [
            {
                "sku": "PRD-1001",
                "warehouse": "WH-NYC-001",
                "days_to_stockout": 3,
                "impact": "NorthStar Retail Group (High Priority) will face shelf shortage on Winter Jacket Premium during peak season",
                "recommended_action": "Initiate expedited transfer from WH-DAL-001 which holds 900+ units of PRD-1001",
                "action_deadline": "Today by 2:00 PM to guarantee Friday delivery",
            },
            {
                "sku": "PRD-2001",
                "warehouse": "WH-ATL-001",
                "days_to_stockout": 5,
                "impact": "Circuit Board X200 below reorder at Atlanta DC — production line risk",
                "recommended_action": "Expedite PO-10005 or source from WH-LAX-001",
                "action_deadline": "Within 48 hours",
            }
        ],
        "contingency_recommendations": [
            "Transfer 600 units PRD-1001 from WH-DAL-001 → WH-NYC-001 via MidWest Truckers",
            "Expedite PO-10002 clearing at Port of Singapore despite disruption risk",
        ],
        "executive_summary": "Two critical stockout risks identified — PRD-1001 at NYC and PRD-2001 at Atlanta. Dallas DC has sufficient contingency inventory for NYC. Immediate action required before 2pm today to preserve Friday delivery commitments to NorthStar Retail.",
    },
    "Supplier & Carrier Performance Agent": {
        "carrier_alerts": [
            {
                "carrier_id": "CAR-001",
                "carrier_name": "FastFreight Logistics",
                "alert_type": "Declining Trend — SLA At Risk",
                "severity": "High",
                "root_cause_hypothesis": "Performance has declined from 97% to 91% over 90 days. 70% of delays cluster on Chicago-Northeast corridor, partially explained by winter weather. 30% are unexplained operational underperformance.",
                "recommended_action": "Initiate SLA review meeting with FastFreight. Request root cause analysis on non-weather delays. Consider partial lane reallocation to backup carrier.",
                "timeline": "Schedule meeting within 5 business days",
            },
            {
                "carrier_id": "CAR-004",
                "carrier_name": "MidWest Truckers Inc.",
                "alert_type": "Below SLA Threshold",
                "severity": "Medium",
                "root_cause_hypothesis": "Current score 89% against 92% SLA. Three consecutive delayed shipments on Dallas-Southeast lane this week — statistically abnormal.",
                "recommended_action": "Investigate Dallas-Southeast lane capacity. Issue formal performance notice.",
                "timeline": "Issue notice this week",
            }
        ],
        "supplier_alerts": [
            {
                "supplier_id": "SUP-003",
                "supplier_name": "SunRise Packaging Co.",
                "alert_type": "Below SLA Fulfillment",
                "severity": "Medium",
                "root_cause_hypothesis": "SLA fulfillment at 88% against target. Multiple short deliveries noted in GRN data.",
                "recommended_action": "Request fulfillment improvement plan from supplier",
                "timeline": "Within 2 weeks",
            }
        ],
        "performance_narrative": "FastFreight Logistics is the most critical performance concern — a consistent downward trend over 3 months now threatens SLA compliance on the Chicago-Northeast corridor. NorthStar Retail's latest scorecard rating of 3.2/5 (down from 4.1) confirms the customer impact. MidWest Truckers requires a formal performance notice on the Dallas-Southeast lane.",
    },
    "Anomaly & Exception Agent": {
        "total_exceptions": 6,
        "critical_count": 2,
        "total_financial_exposure_usd": 87400,
        "top_priority_exceptions": [
            {
                "exception": "Invoice INV-3002: Overbilled 60 units by SUP-002 on PO-10008. $7,200 overcharge.",
                "action": "Raise credit note request with NordicParts GmbH. Do not process payment pending resolution.",
                "deadline": "Before month-end payment run",
            },
            {
                "exception": "Critical inventory depletion — PRD-1001 at WH-NYC-001 at 24% of reorder point.",
                "action": "Trigger emergency transfer from WH-DAL-001. See Demand-Supply Agent recommendation.",
                "deadline": "Today",
            },
            {
                "exception": "Lane delay cluster on WH-DAL-001 → RET-002 and WH-ATL-001 → RET-005. 3 consecutive delays.",
                "action": "Investigate carrier CAR-004 capacity on this lane. Consider temporary rerouting.",
                "deadline": "This week",
            }
        ],
        "exception_narrative": "Six exceptions detected with a combined financial exposure of $87,400. Two critical exceptions require immediate action — a stockout risk at NYC and an invoice overbilling by NordicParts GmbH. Lane delay clustering on the Dallas-South and Atlanta-South routes suggests a systematic carrier capacity issue requiring investigation.",
    },
    "KPI Report Generation Agent": {
        "report_title": "Weekly Supply Chain Performance Report",
        "report_date": datetime.now().strftime("%B %d, %Y"),
        "executive_summary": "This week's supply chain performance is mixed with two KPIs in amber status and one in red. On-time delivery has declined to 62.5% driven by FastFreight and MidWest Truckers underperformance. Inventory health is at risk with two critical stockout scenarios identified at NYC and Atlanta DCs. Purchase order fulfillment sits at 10% given the early stage of the cycle, with 9 orders still open. Immediate attention is required on the NYC contingency transfer and the FastFreight SLA review.",
        "kpi_commentary": {
            "on_time_delivery_rate": "At 62.5%, OTD is significantly below the 95% target. Two carriers — FastFreight and MidWest Truckers — account for all delayed shipments this week.",
            "carrier_sla_compliance": "Average carrier SLA compliance is 93.0%, below the 95% green threshold. FastFreight at 91% and MidWest Truckers at 89% are the primary detractors.",
            "inventory_health": "Inventory health at 74.3% indicates multiple SKU-warehouse combinations below reorder point. NYC and Atlanta are the critical nodes requiring immediate action.",
            "open_exceptions": "Six open exceptions totalling $87,400 in financial exposure. Two are critical severity requiring same-day resolution.",
        },
        "top_concerns": [
            "Stockout risk at WH-NYC-001 for PRD-1001 — contingency transfer window closes today at 2pm",
            "FastFreight Logistics OTD declining for third consecutive month — SLA breach risk in 6 weeks",
            "Invoice discrepancy with NordicParts GmbH — $7,200 overcharge on PO-10008 pending resolution",
        ],
        "positive_highlights": [
            "BlueOcean Shipping and SkyExpress Air Cargo maintaining strong performance above SLA thresholds",
            "Document extraction agent successfully processed all 3 retailer scorecards with 97%+ confidence",
        ],
        "recommended_actions": [
            "Initiate PRD-1001 transfer from WH-DAL-001 to WH-NYC-001 before 2pm today",
            "Schedule FastFreight SLA review meeting for this week",
            "Raise credit note request with NordicParts GmbH for INV-3002",
            "Issue formal performance notice to MidWest Truckers on Dallas-Southeast lane",
        ],
    },
    "Role-Based Copilot Agent_logistics_planner": {
        "direct_answer": "Three shipments are at risk this week: SHP-5002 (Electronics, NYC) routed through Port of Singapore with active port disruption risk; SHP-5003 (Packaging, Dallas→Atlanta) already showing Delayed status; and SHP-5007 (Apparel, Dallas→Miami) delayed by MidWest Truckers on the Dallas-Southeast lane.",
        "key_facts": [
            "SHP-5002 carries port disruption risk — estimated 6-day delay from Singapore disruption",
            "SHP-5003 is already in Delayed status — carrier CAR-004 has 100% delay rate on this lane this week",
            "SHP-5007 delayed — same carrier CAR-004 pattern on Dallas-Southeast lane",
            "Combined at-risk shipment value: approximately $182,000",
            "Dallas contingency for NYC stockout closes at 2pm today",
        ],
        "recommended_actions": [
            "Escalate SHP-5002 with SkyExpress to explore alternative routing around Singapore",
            "Contact MidWest Truckers on SHP-5003 and SHP-5007 for revised ETAs today",
            "Authorize PRD-1001 contingency transfer from Dallas before 2pm deadline",
        ],
        "urgency": "Immediate",
        "data_scope_note": "Answer based on active shipments, carrier performance data, and risk feed tags — logistics scope only",
    },
    "Role-Based Copilot Agent_finance_manager": {
        "direct_answer": "Total invoice discrepancy exposure this month is $7,200 from one confirmed overbilling. Additionally, a damaged goods claim of $1,200 from PO-10008 is pending GRN verification. Combined financial risk is $8,400.",
        "key_facts": [
            "INV-3002 (NordicParts GmbH, PO-10008): billed 400 units, GRN confirms 340 units received — overbilling of 60 units at $120 = $7,200",
            "SHP-5008 GRN records 10 damaged units — estimated claims value $1,200",
            "INV-3004 (FastComp, PO-10005): billed 620 units but shipped qty was 600 — $1,900 discrepancy under review",
            "Total identified discrepancies: $8,400 confirmed + $1,900 under review",
        ],
        "recommended_actions": [
            "Raise formal credit note request with NordicParts GmbH for INV-3002 before payment run",
            "Hold INV-3004 payment pending quantity confirmation with FastComp Electronics",
            "File damaged goods claim for SHP-5008 — $1,200",
        ],
        "urgency": "This Week",
        "data_scope_note": "Answer based on invoice data, GRN records, and financial exceptions — finance scope only",
    },
    "Role-Based Copilot Agent_executive": {
        "direct_answer": "The single biggest risk is a combination of a stockout at your highest-priority retail account (NorthStar Retail Group in NYC) and a declining carrier performance trend on your most critical corridor. A contingency exists but the action window closes at 2pm today.",
        "key_facts": [
            "PRD-1001 (Winter Jacket Premium) at WH-NYC-001 is at 24% of reorder point — 3 days to stockout",
            "NorthStar Retail Group is your highest-priority retail account ($12M annual volume) during peak season",
            "FastFreight Logistics (primary carrier for NYC) has declined from 97% to 91% OTD in 90 days",
            "Dallas DC contingency available — 900 units PRD-1001 — but transfer must be authorized by 2pm today",
            "Compounding risk: Port of Singapore disruption affects 2 inbound shipments worth $312,000",
        ],
        "recommended_actions": [
            "Authorize PRD-1001 emergency transfer from WH-DAL-001 to WH-NYC-001 immediately",
            "Direct supply chain VP to schedule FastFreight SLA review this week",
            "Review Q4 carrier diversification plan given FastFreight trajectory",
        ],
        "urgency": "Immediate",
        "data_scope_note": "Answer synthesized from KPI data, risk flags, supply signals, and carrier performance — executive scope",
    },
    "Role-Based Copilot Agent_procurement_manager": {
        "direct_answer": "Two suppliers have active performance issues: NordicParts GmbH has an open invoice overbilling dispute requiring a credit note, and SunRise Packaging Co. is operating below its 88% SLA fulfillment target with short deliveries noted on recent GRNs.",
        "key_facts": [
            "NordicParts GmbH (SUP-002): INV-3002 overbilled by 60 units — $7,200 discrepancy open",
            "SunRise Packaging Co. (SUP-003): SLA fulfillment at 88% — below contract target, short deliveries on PO-10003",
            "FastComp Electronics (SUP-005): INV-3004 quantity discrepancy of 20 units ($1,900) under review",
            "GlobalTex Industries and Apex Metals performing within SLA — no action needed",
        ],
        "recommended_actions": [
            "Contact NordicParts GmbH today — raise credit note for INV-3002, request explanation",
            "Issue SunRise Packaging a formal SLA performance notice — request improvement plan within 10 days",
            "Resolve FastComp quantity discrepancy before next payment cycle",
        ],
        "urgency": "This Week",
        "data_scope_note": "Answer based on supplier performance data, GRNs, and invoice discrepancies — procurement scope",
    },
    "Proactive Insight Agent": {
        "insights": [
            {
                "target_role": "logistics_planner",
                "subject": "ACTION REQUIRED by 2pm — NYC Stockout Contingency Window Closing",
                "message": "PRD-1001 (Winter Jacket Premium) at WH-NYC-001 will fall below critical threshold within 3 days. WH-DAL-001 has 900+ units available for transfer. To guarantee Friday delivery to NorthStar Retail Group, the transfer order must be placed before 2pm today.",
                "urgency": "Immediate",
                "action_required": "Authorize transfer of 600 units PRD-1001 from WH-DAL-001 to WH-NYC-001 before 2:00 PM today",
                "financial_or_operational_impact": "NorthStar Retail Group ($12M annual account) at risk of shelf shortage during peak season. Estimated lost sales impact: $45,000–$80,000 if not resolved today.",
            },
            {
                "target_role": "procurement_manager",
                "subject": "Invoice Discrepancy Pattern — NordicParts GmbH — $7,200 Exposure",
                "message": "NordicParts GmbH has overbilled on INV-3002 by 60 units ($7,200). This is the second billing discrepancy from this supplier in 60 days, suggesting a systematic issue in their billing process. Month-end payment run is approaching.",
                "urgency": "This Week",
                "action_required": "Raise credit note request with NordicParts GmbH. Place INV-3002 on payment hold. Schedule a billing process review call before month-end.",
                "financial_or_operational_impact": "$7,200 direct overbilling. Pattern risk across other open invoices from this supplier.",
            },
            {
                "target_role": "logistics_planner",
                "subject": "FastFreight Logistics — SLA Breach Trajectory Confirmed",
                "message": "FastFreight Logistics OTD performance has declined from 97% to 91% over the last 90 days — a third consecutive month of decline. At current trajectory, the 98% SLA threshold will be breached in approximately 6 weeks. NorthStar Retail Group has rated them 3.2/5 this month, down from 4.1.",
                "urgency": "This Week",
                "action_required": "Schedule SLA review meeting with FastFreight this week. Prepare lane reallocation options for Northeast corridor as contingency.",
                "financial_or_operational_impact": "Potential SLA penalty clauses + NorthStar relationship risk on $12M annual account.",
            },
            {
                "target_role": "executive",
                "subject": "Supply Chain Risk Briefing — Two Critical Issues Requiring Decision Today",
                "message": "Two critical issues require executive awareness today: (1) NYC inventory stockout risk for peak-season product at your highest-value retail account — contingency available but window closes at 2pm. (2) Port of Singapore disruption affects 2 inbound shipments valued at $312,000 with 6-day estimated delay.",
                "urgency": "Immediate",
                "action_required": "Authorize NYC contingency transfer. Direct logistics VP to activate Singapore disruption contingency routing.",
                "financial_or_operational_impact": "Combined financial risk: $357,000+ if neither issue is addressed today. NorthStar Retail relationship impact if NYC stockout is not prevented.",
            },
        ],
        "total_proactive_alerts": 4,
        "highest_urgency_count": 2,
    },
}


# ── Minimal pipeline runner using mock LLM ──────────────────────────────────

def print_zone_header(zone_num, zone_name, desc):
    print(f"\n\n{Fore.BLUE}{'█'*70}")
    print(f"  ZONE {zone_num} — {zone_name}")
    print(f"  {desc}")
    print(f"{'█'*70}{Style.RESET_ALL}\n")


def print_agent_header(agent_name, zone):
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"  AGENT  : {agent_name}")
    print(f"  ZONE   : {zone}")
    print(f"{'='*70}{Style.RESET_ALL}")


def print_output(label, value):
    if isinstance(value, (dict, list)):
        print(f"{Fore.GREEN}  {label}:{Style.RESET_ALL}")
        print(f"  {json.dumps(value, indent=4, default=str)[:1800]}")
    else:
        print(f"{Fore.GREEN}  {label}: {Style.RESET_ALL}{str(value)[:400]}")


def save_output(data, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    def serialize(obj):
        if isinstance(obj, pd.DataFrame): return obj.to_dict(orient="records")
        if hasattr(obj, "isoformat"): return obj.isoformat()
        return str(obj)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=serialize)
    print(f"  {Fore.CYAN}Output saved → {filepath}{Style.RESET_ALL}")


def run_demo():
    print(f"\n{Fore.CYAN}{'='*70}")
    print("  AI-ACCELERATED SUPPLY CHAIN — DEMO PIPELINE (Mock LLM)")
    print("  Azure OpenAI GPT-5 | 12 Agents | 4 Zones")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    # Load data
    print(f"{Fore.WHITE}Loading synthetic supply chain data...{Style.RESET_ALL}")
    data = load_all_data()
    for name, ds in data.items():
        count = len(ds) if isinstance(ds, (pd.DataFrame, list)) else "N/A"
        print(f"  ✓ {name}: {count} records")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline = {}


    # ZONE 0 — PRE-INGESTION

    print_zone_header(0, "PRE-INGESTION", "Document Extraction | DQ Gate | Source Anomaly")

    # Agent 1: Document Agent 
    print_agent_header("Document & Unstructured Data Agent", "Zone 0 — Pre-Ingestion")
    print_output("Input", f"{len(data['retailer_scorecards'])} raw PDF/document objects (retailer scorecards)")

    structured_records = []
    extraction_log = []
    for doc in data["retailer_scorecards"]:
        llm_out = MOCK_RESPONSES["Document & Unstructured Data Agent"].copy()
        normalized = doc.copy()
        normalized["_extraction_quality"] = llm_out["extraction_quality"]
        normalized["_extracted_at"]       = ts
        normalized["_document_category"]  = llm_out["document_category"]
        structured_records.append(normalized)
        extraction_log.append({
            "source": doc.get("retailer_name"),
            "fields_extracted": llm_out["fields_extracted"],
            "extraction_quality": llm_out["extraction_quality"],
            "issues_found": llm_out["issues_found"],
        })

    z0_a1 = {"structured_records": structured_records, "extraction_log": extraction_log,
              "total_extracted": len(structured_records), "processed_at": ts}
    print_output("Extraction Log", extraction_log)
    print(f"\n{Fore.GREEN}  ✓ Agent 1 Complete: {len(structured_records)} documents structured{Style.RESET_ALL}")
    save_output(z0_a1, "zone0_agent1_document_extraction.json")

    # Agent 2: DQ Gate
    print_agent_header("Data Quality Gate Agent", "Zone 0 — Pre-Ingestion")
    datasets_to_check = {
        "purchase_orders": data["purchase_orders"],
        "shipments":       data["shipments"],
        "supplier_invoices": data["supplier_invoices"],
        "inventory":       data["inventory"],
    }
    print_output("Input", {k: f"{len(v)} records" for k, v in datasets_to_check.items()})

    passed, quarantine, dq_summary = {}, {}, []
    for ds_name, df in datasets_to_check.items():
        p_recs, q_recs = [], []
        for _, row in df.iterrows():
            rec = row.to_dict()
            score = 100
            issues = []
            nulls = [k for k, v in rec.items() if v is None or (isinstance(v, float) and pd.isna(v))]
            if nulls:
                score -= len(nulls) * 15
                issues.append(f"Null fields: {nulls}")
            if ds_name == "purchase_orders" and not rec.get("destination_wh"):
                score -= 25; issues.append("Missing destination warehouse")
            rec["_dq_score"] = max(0, score)
            rec["_dq_issues"] = issues
            if max(0, score) >= DQ_SCORE_THRESHOLD:
                p_recs.append(rec)
            else:
                rec["_quarantine_reason"] = "; ".join(issues)
                q_recs.append(rec)
        passed[ds_name]     = pd.DataFrame(p_recs)
        quarantine[ds_name] = pd.DataFrame(q_recs) if q_recs else pd.DataFrame()
        dq_summary.append({"dataset": ds_name, "total": len(df), "passed": len(p_recs),
                            "quarantined": len(q_recs), "pass_rate_pct": round(len(p_recs)/len(df)*100,1)})
        print(f"  {ds_name}: {len(p_recs)} passed / {len(q_recs)} quarantined")

    llm_dq = MOCK_RESPONSES["Data Quality Gate Agent"]
    z0_a2 = {"passed_datasets": passed, "quarantine_records": quarantine,
              "dq_summary": dq_summary, "llm_dq_analysis": llm_dq, "processed_at": ts}
    print_output("DQ Summary", dq_summary)
    print_output("LLM DQ Analysis", llm_dq)
    total_q = sum(len(q) for q in quarantine.values() if isinstance(q, pd.DataFrame))
    print(f"\n{Fore.GREEN}  ✓ Agent 2 Complete: {total_q} records quarantined{Style.RESET_ALL}")
    save_output(z0_a2, "zone0_agent2_data_quality.json")

    # Agent 3: Source Anomaly
    print_agent_header("Source Anomaly Detection Agent", "Zone 0 — Pre-Ingestion")
    feed_meta = {k: len(v) for k, v in {
        "purchase_orders": data["purchase_orders"], "shipments": data["shipments"],
        "inventory": data["inventory"], "supplier_invoices": data["supplier_invoices"],
        "risk_feed": data["risk_feed"]}.items()}
    print_output("Input", feed_meta)

    risk_events = data["risk_feed"].to_dict(orient="records")
    anomalies, feed_health, tagged = [], [], []
    baselines = {"purchase_orders": 8, "shipments": 8, "inventory": 35, "supplier_invoices": 5, "risk_feed": 2}
    for feed, count in feed_meta.items():
        baseline = baselines.get(feed, 10)
        ratio = count / baseline
        status = "Normal"
        if ratio > ANOMALY_VOLUME_THRESHOLD:
            status = "Volume Spike"
            anomalies.append({"feed": feed, "type": "Volume Spike", "ratio": round(ratio, 2), "severity": "Medium"})
        feed_health.append({"feed": feed, "current": count, "baseline": baseline, "status": status})

    for _, row in data["shipments"].iterrows():
        rec = row.to_dict()
        risk_tag = None
        for ev in risk_events:
            if ev.get("active") and rec.get("port_risk_flag"):
                risk_tag = {"event_id": ev["event_id"], "event_type": ev["event_type"],
                            "severity": ev["severity"], "delay_days": ev["estimated_delay_days"]}
                break
        rec["_risk_tag"] = risk_tag
        rec["_priority"] = "HIGH" if risk_tag else "NORMAL"
        tagged.append(rec)

    tagged_count = sum(1 for r in tagged if r.get("_risk_tag"))
    llm_anomaly  = MOCK_RESPONSES["Source Anomaly Detection Agent"]
    z0_a3 = {"feed_health": feed_health, "anomalies": anomalies,
              "priority_tagged_shipments": tagged, "llm_risk_analysis": llm_anomaly, "processed_at": ts}
    print_output("Feed Health", feed_health)
    print_output("Anomalies", anomalies)
    print_output("LLM Risk Analysis", llm_anomaly)
    print(f"\n{Fore.GREEN}  ✓ Agent 3 Complete: {len(anomalies)} anomalies, {tagged_count} shipments risk-tagged{Style.RESET_ALL}")
    save_output(z0_a3, "zone0_agent3_source_anomaly.json")

    pipeline["clean_pos"]        = passed["purchase_orders"]
    pipeline["clean_shipments"]  = passed["shipments"]
    pipeline["clean_invoices"]   = passed["supplier_invoices"]
    pipeline["clean_inventory"]  = passed["inventory"]
    pipeline["scorecards"]       = structured_records
    pipeline["risk_events"]      = risk_events
    pipeline["anomalies"]        = anomalies


    # ZONE 1 — PROCESSING

    print_zone_header(1, "PROCESSING", "Reconciliation | Semantic Enrichment | Promotion")

    # Agent 4: Reconciliation 
    print_agent_header("Reconciliation Agent", "Zone 1 — Processing (Bronze → Silver)")
    print_output("Input", f"{len(pipeline['clean_shipments'])} shipment records (with TMS vs ERP status conflicts)")

    reconciled, conflict_log, po_grn = [], [], []
    for _, row in pipeline["clean_shipments"].iterrows():
        rec = row.to_dict()
        tms = rec.get("tms_status", "")
        erp = rec.get("erp_status", "")
        if tms and erp and tms != erp:
            res = MOCK_RESPONSES["Reconciliation Agent"].copy()
            rec["_reconciled_status"]   = tms
            rec["_reconciliation_note"] = res["resolution_rationale"]
            rec["_had_conflict"]        = True
            conflict_log.append({"shipment_id": rec["shipment_id"],
                                  "conflict": f"TMS='{tms}' vs ERP='{erp}'",
                                  "resolution": tms, "rationale": res["resolution_rationale"]})
            print(f"  Conflict resolved: {rec['shipment_id']} TMS='{tms}' vs ERP='{erp}' → '{tms}'")
        else:
            rec["_reconciled_status"] = tms or erp
            rec["_had_conflict"]      = False
        rec["_layer"] = "Silver"
        reconciled.append(rec)

    grn_df = data["goods_receipts"]
    for _, grn_row in grn_df.iterrows():
        grn = grn_row.to_dict()
        po_match = pipeline["clean_pos"][pipeline["clean_pos"]["po_id"] == grn["po_id"]]
        if not po_match.empty:
            ordered  = po_match.iloc[0]["ordered_qty"]
            received = grn["received_qty"]
            gap = ordered - received
            po_grn.append({"po_id": grn["po_id"], "ordered_qty": ordered, "received_qty": received,
                            "quantity_gap": gap, "status": "Short Delivery" if gap > 0 else "Full Delivery"})

    z1_a4 = {"silver_shipments": pd.DataFrame(reconciled), "conflict_log": conflict_log,
              "po_grn_reconciliation": po_grn, "processed_at": ts}
    print_output("Conflict Log", conflict_log)
    print_output("PO-GRN Reconciliation", po_grn)
    print(f"\n{Fore.GREEN}  ✓ Agent 4 Complete: {len(reconciled)} reconciled, {len(conflict_log)} conflicts resolved{Style.RESET_ALL}")
    save_output(z1_a4, "zone1_agent4_reconciliation.json")

    # Agent 5: Semantic Enrichment
    print_agent_header("Semantic Enrichment Agent", "Zone 1 — Processing (Silver → Gold)")
    print_output("Input", f"{len(reconciled)} silver records + 5 master tables")

    enriched = []
    carrier_map  = data["carrier_master"].set_index("carrier_id").to_dict(orient="index")
    wh_map       = data["warehouse_master"].set_index("warehouse_id").to_dict(orient="index")
    ret_map      = data["retailer_master"].set_index("retailer_id").to_dict(orient="index")
    po_map       = pipeline["clean_pos"].set_index("po_id").to_dict(orient="index")
    prd_map      = data["product_master"].set_index("sku").to_dict(orient="index")

    for rec in reconciled:
        r = rec.copy()
        car   = carrier_map.get(r.get("carrier_id"), {})
        wh    = wh_map.get(r.get("origin_warehouse"), {})
        ret   = ret_map.get(r.get("destination_retailer"), {})
        po    = po_map.get(r.get("po_id"), {})
        prd   = prd_map.get(po.get("sku", ""), {})
        llm_e = MOCK_RESPONSES["Semantic Enrichment Agent"]

        r["_carrier_name"]       = car.get("carrier_name", r.get("carrier_id"))
        r["_carrier_sla"]        = car.get("sla_ontime_pct")
        r["_carrier_score"]      = car.get("current_score")
        r["_origin_wh_name"]     = wh.get("warehouse_name", r.get("origin_warehouse"))
        r["_retailer_name"]      = ret.get("retailer_name", r.get("destination_retailer"))
        r["_retailer_priority"]  = ret.get("priority")
        r["_product_name"]       = prd.get("product_name", "")
        r["_product_category"]   = prd.get("category", "")
        r["_business_summary"]   = llm_e["business_summary"]
        r["_risk_level"]         = llm_e["risk_level"]
        r["_business_priority"]  = llm_e["business_priority"]
        r["_sla_at_risk"]        = llm_e["sla_at_risk"]
        r["_enrichment_tags"]    = llm_e["enrichment_tags"]
        r["_enriched_at"]        = ts
        r["_layer"]              = "Pre-Gold"
        enriched.append(r)
        print(f"  {r.get('shipment_id')} → {r['_carrier_name']} | {r['_product_name'] or 'N/A'} "
              f"| Priority: {r['_business_priority']} | Risk: {r['_risk_level']}")

    z1_a5 = {"enriched_records": enriched, "total_enriched": len(enriched), "processed_at": ts}
    print(f"\n{Fore.GREEN}  ✓ Agent 5 Complete: {len(enriched)} records enriched{Style.RESET_ALL}")
    save_output(z1_a5, "zone1_agent5_enrichment.json")

    # Agent 6: Promotion
    print_agent_header("Data Classification & Promotion Agent", "Zone 1 — Processing (Silver → Gold)")
    print_output("Input", f"{len(enriched)} enriched records awaiting promotion evaluation")

    gold, hold, audit = [], [], []
    for rec in enriched:
        score = 100
        issues = []
        if not rec.get("_carrier_name"): score -= 20; issues.append("Missing carrier enrichment")
        if not rec.get("_retailer_name"): score -= 15; issues.append("Missing retailer enrichment")
        if rec.get("_had_conflict") and not rec.get("_reconciled_status"): score -= 20; issues.append("Unresolved conflict")
        decision = "PROMOTED" if score >= PROMOTION_THRESHOLD else "HELD"
        rec["_promotion_score"] = score
        rec["_layer"]           = "Gold" if decision == "PROMOTED" else "Silver-Hold"
        (gold if decision == "PROMOTED" else hold).append(rec)
        audit.append({"shipment_id": rec.get("shipment_id"), "score": score, "decision": decision, "issues": issues})
        color = Fore.GREEN if decision == "PROMOTED" else Fore.YELLOW
        print(f"  {color}{rec.get('shipment_id')}: score={score} → {decision}{Style.RESET_ALL}")

    llm_promo = MOCK_RESPONSES["Data Classification & Promotion Agent"]
    z1_a6 = {"gold_records": gold, "hold_queue": hold, "audit_log": audit,
              "promotion_summary": llm_promo, "processed_at": ts}
    print_output("Promotion Summary", llm_promo)
    print_output("Audit Log", audit)
    print(f"\n{Fore.GREEN}  ✓ Agent 6 Complete: {len(gold)} promoted to Gold, {len(hold)} held{Style.RESET_ALL}")
    save_output(z1_a6, "zone1_agent6_promotion.json")

    pipeline["gold_records"]   = gold
    pipeline["conflict_log"]   = conflict_log

    # ZONE 2 — GOLD INTELLIGENCE
    print_zone_header(2, "GOLD LAYER INTELLIGENCE", "Demand-Supply | Performance | Anomaly & Exception")

    # Agent 7: Demand-Supply 
    print_agent_header("Demand & Supply Signal Agent", "Zone 2 — Gold Layer")
    print_output("Input", f"{len(pipeline['clean_inventory'])} inventory records, {len(pipeline['clean_pos'])} POs, {len(gold)} Gold shipments")

    inv_df = pipeline["clean_inventory"]
    pos_df = pipeline["clean_pos"]
    balance_signals, risk_flags, contingencies = [], [], []

    for _, inv_row in inv_df.iterrows():
        inv   = inv_row.to_dict()
        sku   = inv["sku"]
        wh_id = inv["warehouse_id"]
        qty   = inv["qty_on_hand"]
        reord = inv["reorder_point"]
        open_po_qty = pos_df[(pos_df["sku"]==sku)&(pos_df["destination_wh"]==wh_id)&
                              (pos_df["status"].isin(["Open","Confirmed","In Transit"]))]["ordered_qty"].sum()
        total_avail = qty + open_po_qty
        gap = total_avail - reord
        risk = "Critical" if gap < 0 and qty < reord*0.3 else "High" if gap < 0 else "Medium" if gap < reord*0.2 else "Low"
        sig = {"sku": sku, "warehouse_id": wh_id, "qty_on_hand": qty, "in_transit": int(open_po_qty),
               "total_available": int(total_avail), "reorder_point": reord, "supply_gap": int(gap), "risk_level": risk}
        balance_signals.append(sig)
        if risk in ["Critical", "High"]: risk_flags.append(sig)

    for flag in risk_flags:
        surplus = [s for s in balance_signals if s["sku"]==flag["sku"] and s["supply_gap"]>500 and s["warehouse_id"]!=flag["warehouse_id"]]
        if surplus:
            contingencies.append({"at_risk_wh": flag["warehouse_id"], "sku": flag["sku"],
                                   "shortfall": abs(flag["supply_gap"]),
                                   "contingency_from": surplus[0]["warehouse_id"],
                                   "available_qty": surplus[0]["qty_on_hand"]})

    llm_supply = MOCK_RESPONSES["Demand & Supply Signal Agent"]
    z2_a7 = {"balance_signals": balance_signals, "risk_flags": risk_flags,
              "contingencies": contingencies, "supply_analysis": llm_supply, "processed_at": ts}
    print_output("Risk Flags", len(risk_flags))
    print_output("Contingencies", contingencies[:3])
    print_output("LLM Supply Analysis", llm_supply)
    print(f"\n{Fore.GREEN}  ✓ Agent 7 Complete: {len(risk_flags)} supply risk flags, {len(contingencies)} contingencies{Style.RESET_ALL}")
    save_output(z2_a7, "zone2_agent7_demand_supply.json")

    # Agent 8: Performance
    print_agent_header("Supplier & Carrier Performance Agent", "Zone 2 — Gold Layer")
    print_output("Input", f"{len(data['carrier_master'])} carriers, {len(data['supplier_master'])} suppliers, {len(pipeline['scorecards'])} scorecards")

    carrier_perf = []
    perf_hist    = data["carrier_performance_history"]
    for _, car_row in data["carrier_master"].iterrows():
        car   = car_row.to_dict()
        cid   = car["carrier_id"]
        hist  = perf_hist[perf_hist["carrier_id"]==cid].sort_values("period_date")["ontime_pct"].tolist()
        c_sh  = [r for r in gold if r.get("carrier_id")==cid]
        delayed= [s for s in c_sh if s.get("status")=="Delayed"]
        sc_scores = [s["overall_score"] for s in pipeline["scorecards"] if s.get("carrier_id")==cid]
        trend = "Declining" if len(hist)>=3 and hist[-1]<hist[-3] else "Improving" if len(hist)>=3 and hist[-1]>hist[-3] else "Stable"
        carrier_perf.append({"carrier_id": cid, "carrier_name": car["carrier_name"],
                              "sla_ontime_pct": car["sla_ontime_pct"], "current_score": car["current_score"],
                              "trend": trend, "delayed_shipments": len(delayed),
                              "scorecard_avg": round(sum(sc_scores)/len(sc_scores),2) if sc_scores else None,
                              "sla_breached": car["current_score"] < car["sla_ontime_pct"]})

    at_risk = [c for c in carrier_perf if c["sla_breached"] or c["trend"]=="Declining"]
    llm_perf = MOCK_RESPONSES["Supplier & Carrier Performance Agent"]
    z2_a8 = {"carrier_performance": carrier_perf, "performance_analysis": llm_perf,
              "at_risk_carriers": at_risk, "processed_at": ts}
    print_output("Carrier Performance", carrier_perf)
    print_output("LLM Performance Analysis", llm_perf)
    print(f"\n{Fore.GREEN}  ✓ Agent 8 Complete: {len(at_risk)} carrier alerts{Style.RESET_ALL}")
    save_output(z2_a8, "zone2_agent8_performance.json")

    # Agent 9: Anomaly & Exception
    print_agent_header("Anomaly & Exception Agent", "Zone 2 — Gold Layer")
    print_output("Input", f"{len(gold)} Gold records, {len(pipeline['clean_invoices'])} invoices, {len(data['goods_receipts'])} GRNs")

    exceptions = []
    grn_df = data["goods_receipts"]
    inv_df2 = pipeline["clean_invoices"]

    for _, inv_row in inv_df2.iterrows():
        inv = inv_row.to_dict()
        grn_match = grn_df[grn_df["po_id"]==inv.get("po_id")]
        if not grn_match.empty:
            grn = grn_match.iloc[0].to_dict()
            billed = inv.get("billed_qty",0)
            recvd  = grn.get("received_qty",0)
            gap = billed - recvd
            if abs(gap) > 0:
                impact = round(abs(gap)*inv.get("unit_price",0),2)
                exceptions.append({"exception_id": f"EXC-{len(exceptions)+1:03d}",
                                   "type": "Invoice-GRN Quantity Mismatch",
                                   "severity": "High" if impact>10000 else "Medium",
                                   "po_id": inv["po_id"], "invoice_id": inv["invoice_id"],
                                   "supplier_id": inv["supplier_id"],
                                   "billed_qty": billed, "received_qty": recvd,
                                   "qty_gap": gap, "financial_impact_usd": impact,
                                   "owner": "Finance & Procurement"})

    delayed_shp = [r for r in gold if r.get("status")=="Delayed"]
    from collections import Counter
    lane_counts = Counter(f"{r.get('origin_warehouse')}-{r.get('destination_retailer')}" for r in delayed_shp)
    for lane, count in lane_counts.items():
        if count >= 2:
            exceptions.append({"exception_id": f"EXC-{len(exceptions)+1:03d}",
                               "type": "Lane Delay Cluster", "severity": "High" if count>=3 else "Medium",
                               "lane": lane, "delayed_count": count,
                               "financial_impact_usd": count*5000, "owner": "Logistics Operations"})

    damaged = grn_df[grn_df["damaged_qty"]>0]
    for _, dmg_row in damaged.iterrows():
        dmg = dmg_row.to_dict()
        po_match = pos_df[pos_df["po_id"]==dmg.get("po_id")]
        up = po_match.iloc[0]["unit_price"] if not po_match.empty else 0
        exceptions.append({"exception_id": f"EXC-{len(exceptions)+1:03d}",
                           "type": "Goods Damage on Receipt", "severity": "Medium",
                           "grn_id": dmg["grn_id"], "damaged_qty": dmg["damaged_qty"],
                           "financial_impact_usd": round(dmg["damaged_qty"]*up,2),
                           "owner": "Warehouse & Quality"})

    crit_inv = inv_df[inv_df["below_reorder"]==True]
    crit_inv = crit_inv[crit_inv["qty_on_hand"] < crit_inv["reorder_point"]*0.3]
    for _, inv_row in crit_inv.iterrows():
        inv = inv_row.to_dict()
        exceptions.append({"exception_id": f"EXC-{len(exceptions)+1:03d}",
                           "type": "Critical Inventory Depletion", "severity": "Critical",
                           "warehouse_id": inv["warehouse_id"], "sku": inv["sku"],
                           "qty_on_hand": inv["qty_on_hand"], "reorder_point": inv["reorder_point"],
                           "financial_impact_usd": inv["qty_on_hand"]*50, "owner": "Inventory Planning"})

    total_exposure = sum(e.get("financial_impact_usd",0) for e in exceptions)
    llm_exc = MOCK_RESPONSES["Anomaly & Exception Agent"]
    z2_a9 = {"exceptions": exceptions, "exception_analysis": llm_exc,
              "total_financial_exposure_usd": total_exposure, "processed_at": ts}
    print_output("Exception Log", exceptions)
    print_output("LLM Exception Analysis", llm_exc)
    print(f"\n{Fore.GREEN}  ✓ Agent 9 Complete: {len(exceptions)} exceptions, ${total_exposure:,.2f} exposure{Style.RESET_ALL}")
    save_output(z2_a9, "zone2_agent9_exceptions.json")

    pipeline["risk_flags"]           = risk_flags
    pipeline["contingencies"]        = contingencies
    pipeline["supply_analysis"]      = llm_supply
    pipeline["carrier_performance"]  = carrier_perf
    pipeline["performance_analysis"] = llm_perf
    pipeline["exceptions"]           = exceptions
    pipeline["total_financial_exposure_usd"] = total_exposure


    # ZONE 3 — REPORTING
    print_zone_header(3, "REPORTING & CONSUMPTION", "KPI Reports | Role-Based Copilot | Proactive Insights")

    # Agent 10: KPI Report
    print_agent_header("KPI Report Generation Agent", "Zone 3 — Reporting")

    delivered = [r for r in gold if r.get("status")=="Delivered"]
    delayed   = [r for r in gold if r.get("status")=="Delayed"]
    otd_rate  = round(len(delivered)/len(gold)*100,1) if gold else 0
    sla_scores= [c["current_score"] for c in carrier_perf]
    avg_sla   = round(sum(sla_scores)/len(sla_scores),1) if sla_scores else 0
    below_r   = len(inv_df[inv_df["below_reorder"]==True])
    inv_h_pct = round((1-below_r/len(inv_df))*100,1) if len(inv_df)>0 else 100
    closed_po = len(pos_df[pos_df["status"]=="Closed"])
    po_pct    = round(closed_po/len(pos_df)*100,1) if len(pos_df)>0 else 0

    kpi_values = {
        "on_time_delivery_rate_pct":   otd_rate,
        "avg_carrier_sla_compliance":  avg_sla,
        "inventory_health_pct":        inv_h_pct,
        "open_exceptions_total":       len(exceptions),
        "open_exceptions_critical":    len([e for e in exceptions if e.get("severity")=="Critical"]),
        "open_exceptions_high":        len([e for e in exceptions if e.get("severity")=="High"]),
        "po_fulfillment_rate_pct":     po_pct,
        "total_financial_exposure_usd":total_exposure,
        "active_shipments":            len(gold),
        "delayed_shipments":           len(delayed),
    }

    def kpi_status(k, v):
        th = {"on_time_delivery_rate_pct":(95,90),"avg_carrier_sla_compliance":(95,90),
              "inventory_health_pct":(90,80),"po_fulfillment_rate_pct":(90,75)}
        if k in th:
            g, a = th[k]
            return "🟢 On Target" if v>=g else "🟡 At Risk" if v>=a else "🔴 Breached"
        return "ℹ️ Informational"

    kpi_report = [{"kpi": k.replace("_"," ").title(), "value": v, "status": kpi_status(k,v)} for k,v in kpi_values.items()]
    llm_kpi    = MOCK_RESPONSES["KPI Report Generation Agent"]
    z3_a10 = {"kpi_values": kpi_values, "kpi_report": kpi_report, "report_narrative": llm_kpi, "processed_at": ts}
    print_output("KPI Scorecard", kpi_report)
    print_output("KPI Report Narrative", llm_kpi)
    print(f"\n{Fore.GREEN}  ✓ Agent 10 Complete: {len(kpi_report)} KPIs computed and reported{Style.RESET_ALL}")
    save_output(z3_a10, "zone3_agent10_kpi_report.json")
    pipeline["kpi_values"] = kpi_values

    # Agent 11: Copilot
    print_agent_header("Role-Based Copilot Agent", "Zone 3 — Reporting")
    queries = [
        ("logistics_planner",   "Which shipments are at risk of missing their delivery window this week?"),
        ("finance_manager",     "What is our total invoice discrepancy exposure this month?"),
        ("executive",           "What is the single biggest supply chain risk I need to know about right now?"),
        ("procurement_manager", "Which suppliers have open performance issues I should address this week?"),
    ]
    copilot_responses = []
    for role, query in queries:
        print(f"\n  {Fore.YELLOW}[{role.upper()}]{Style.RESET_ALL}: {query}")
        answer = MOCK_RESPONSES.get(f"Role-Based Copilot Agent_{role}", {"direct_answer": "N/A"})
        print(f"  {Fore.GREEN}Answer:{Style.RESET_ALL} {answer.get('direct_answer','')}")
        print(f"  {Fore.GREEN}Urgency:{Style.RESET_ALL} {answer.get('urgency','')}")
        print(f"  {Fore.GREEN}Actions:{Style.RESET_ALL} {answer.get('recommended_actions',[])} ")
        copilot_responses.append({"role": role, "query": query, "answer": answer})

    z3_a11 = {"copilot_responses": copilot_responses, "queries_answered": len(copilot_responses), "processed_at": ts}
    print(f"\n{Fore.GREEN}  ✓ Agent 11 Complete: {len(copilot_responses)} role-scoped queries answered{Style.RESET_ALL}")
    save_output(z3_a11, "zone3_agent11_copilot.json")

    # Agent 12: Proactive Insights
    print_agent_header("Proactive Insight Agent", "Zone 3 — Reporting")
    llm_insights = MOCK_RESPONSES["Proactive Insight Agent"]
    print(f"\n{Fore.YELLOW}  ── PROACTIVE INSIGHTS ──{Style.RESET_ALL}")
    for insight in llm_insights.get("insights", []):
        urgency_color = Fore.RED if insight["urgency"]=="Immediate" else Fore.YELLOW if insight["urgency"]=="This Week" else Fore.CYAN
        print(f"\n  {urgency_color}[{insight['urgency'].upper()}] → {insight['target_role'].upper()}{Style.RESET_ALL}")
        print(f"  Subject : {insight['subject']}")
        print(f"  Message : {insight['message']}")
        print(f"  Action  : {insight['action_required']}")
        print(f"  Impact  : {insight['financial_or_operational_impact']}")

    z3_a12 = {"proactive_insights": llm_insights, "total_alerts": llm_insights["total_proactive_alerts"], "processed_at": ts}
    print(f"\n{Fore.GREEN}  ✓ Agent 12 Complete: {llm_insights['total_proactive_alerts']} proactive insights{Style.RESET_ALL}")
    save_output(z3_a12, "zone3_agent12_proactive.json")

    # FINAL SUMMARY
    summary = {
        "pipeline_completed_at": ts,
        "zones_executed": 4, "agents_executed": 12,
        "zone_0_pre_ingestion": {
            "documents_structured": len(structured_records),
            "records_quarantined": total_q,
            "anomalies_detected": len(anomalies),
            "shipments_risk_tagged": tagged_count,
        },
        "zone_1_processing": {
            "conflicts_resolved": len(conflict_log),
            "records_enriched": len(enriched),
            "records_promoted_to_gold": len(gold),
            "records_held_in_silver": len(hold),
        },
        "zone_2_gold_intelligence": {
            "supply_risk_flags": len(risk_flags),
            "contingencies_identified": len(contingencies),
            "carrier_alerts": len(at_risk),
            "exceptions_detected": len(exceptions),
            "total_financial_exposure": f"${total_exposure:,.2f}",
        },
        "zone_3_reporting": {
            "kpis_computed": len(kpi_report),
            "copilot_queries_answered": len(copilot_responses),
            "proactive_insights_sent": llm_insights["total_proactive_alerts"],
        },
    }
    save_output(summary, "pipeline_summary.json")

    print(f"\n\n{Fore.CYAN}{'='*70}")
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print(f"{'='*70}{Style.RESET_ALL}")
    for zone, metrics in summary.items():
        if isinstance(metrics, dict):
            print(f"\n  {Fore.YELLOW}{zone.replace('_',' ').upper()}{Style.RESET_ALL}")
            for k, v in metrics.items():
                print(f"    • {k.replace('_',' ')}: {Fore.GREEN}{v}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}  All outputs saved to ./{OUTPUT_DIR}/")
    print(f"  Completed: {ts}")
    print(f"{'='*70}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    run_demo()
