# End-to-End Supply Chain AI Agent Pipeline Orchestrator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import pandas as pd
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)

# Synthetic Data
from synthetic_data.generator import load_all_data

# Zone 0: Pre-Ingestion
from zone0_pre_ingestion.agents import (
    DocumentAgent,
    DataQualityGateAgent,
    SourceAnomalyDetectionAgent,
)

# Zone 1: Processing
from zone1_processing.agents import (
    ReconciliationAgent,
    SemanticEnrichmentAgent,
    DataPromotionAgent,
)

# Zone 2: Gold Layer
from zone2_gold.agents import (
    DemandSupplySignalAgent,
    SupplierCarrierPerformanceAgent,
    AnomalyExceptionAgent,
)

# Zone 3: Reporting
from zone3_reporting.agents import (
    KPIReportAgent,
    RoleBasedCopilotAgent,
    ProactiveInsightAgent,
)

from config import OUTPUT_DIR


def print_zone_header(zone_num: int, zone_name: str, desc: str):
    print(f"\n\n{Fore.BLUE}{'█'*70}")
    print(f"{'█'*70}")
    print(f"  ZONE {zone_num} — {zone_name}")
    print(f"  {desc}")
    print(f"{'█'*70}")
    print(f"{'█'*70}{Style.RESET_ALL}\n")


def print_pipeline_header():
    print(f"\n{Fore.CYAN}{'='*70}")
    print("  AI-ACCELERATED SUPPLY CHAIN — END-TO-END AGENT PIPELINE")
    print("  Azure OpenAI GPT-5 | 12 Agents | 4 Zones")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{Style.RESET_ALL}\n")


def save_output(data: dict, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    def serialize(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=serialize)
    print(f"  {Fore.CYAN}Output saved: {filepath}{Style.RESET_ALL}")


def main():
    print_pipeline_header()

    # Load all synthetic data 
    print(f"{Fore.WHITE}Loading synthetic supply chain data...{Style.RESET_ALL}")
    data = load_all_data()
    print(f"  ✓ Loaded {len(data)} datasets")
    for name, dataset in data.items():
        count = len(dataset) if isinstance(dataset, (pd.DataFrame, list)) else "N/A"
        print(f"    • {name}: {count} records")

    pipeline_state = {}   # accumulates outputs across zones


    # ZONE 0 — Pre Ingestion

    print_zone_header(0, "PRE-INGESTION", "Document Extraction | Data Quality | Source Anomaly Detection")

    # Agent 1 — Document Agent
    agent1 = DocumentAgent()
    z0_a1_output = agent1.run({
        "raw_documents": data["retailer_scorecards"]
    })
    save_output(z0_a1_output, "zone0_agent1_document_extraction.json")

    # Agent 2 — DQ Gate Agent
    agent2 = DataQualityGateAgent()
    z0_a2_output = agent2.run({
        "datasets": {
            "purchase_orders":   data["purchase_orders"],
            "shipments":         data["shipments"],
            "supplier_invoices": data["supplier_invoices"],
            "inventory":         data["inventory"],
        }
    })
    save_output(z0_a2_output, "zone0_agent2_data_quality.json")

    # Agent 3 — Source Anomaly Agent
    agent3 = SourceAnomalyDetectionAgent()
    z0_a3_output = agent3.run({
        "feed_metadata": {
            "purchase_orders":   len(data["purchase_orders"]),
            "shipments":         len(data["shipments"]),
            "inventory":         len(data["inventory"]),
            "supplier_invoices": len(data["supplier_invoices"]),
            "risk_feed":         len(data["risk_feed"]),
        },
        "risk_events":   data["risk_feed"].to_dict(orient="records"),
        "shipments_df":  data["shipments"],
    })
    save_output(z0_a3_output, "zone0_agent3_source_anomaly.json")

    # Carry forward clean datasets
    pipeline_state["clean_purchase_orders"] = z0_a2_output["passed_datasets"].get("purchase_orders", data["purchase_orders"])
    pipeline_state["clean_shipments"]       = z0_a2_output["passed_datasets"].get("shipments", data["shipments"])
    pipeline_state["clean_invoices"]        = z0_a2_output["passed_datasets"].get("supplier_invoices", data["supplier_invoices"])
    pipeline_state["clean_inventory"]       = z0_a2_output["passed_datasets"].get("inventory", data["inventory"])
    pipeline_state["structured_scorecards"] = z0_a1_output["structured_records"]
    pipeline_state["risk_events"]           = data["risk_feed"].to_dict(orient="records")
    pipeline_state["anomalies"]             = z0_a3_output["anomalies"]

    print(f"\n{Fore.GREEN}  ✓ ZONE 0 COMPLETE{Style.RESET_ALL}")


    # Zone 1 — Processing (Bronze → Silver → Gold)

    print_zone_header(1, "PROCESSING", "Reconciliation | Semantic Enrichment | Data Promotion")

    # Agent 4 — Reconciliation Agent
    agent4 = ReconciliationAgent()
    z1_a4_output = agent4.run({
        "shipments_df":        pipeline_state["clean_shipments"],
        "purchase_orders_df":  pipeline_state["clean_purchase_orders"],
        "goods_receipts_df":   data["goods_receipts"],
    })
    save_output(z1_a4_output, "zone1_agent4_reconciliation.json")

    # Agent 5 — Semantic Enrichment Agent
    agent5 = SemanticEnrichmentAgent()
    z1_a5_output = agent5.run({
        "silver_shipments":    z1_a4_output["silver_shipments"],
        "supplier_master":     data["supplier_master"],
        "carrier_master":      data["carrier_master"],
        "product_master":      data["product_master"],
        "warehouse_master":    data["warehouse_master"],
        "retailer_master":     data["retailer_master"],
        "purchase_orders_df":  pipeline_state["clean_purchase_orders"],
    })
    save_output(z1_a5_output, "zone1_agent5_enrichment.json")

    # Agent 6 — Data Promotion Agent
    agent6 = DataPromotionAgent()
    z1_a6_output = agent6.run({
        "enriched_records": z1_a5_output["enriched_records"],
    })
    save_output(z1_a6_output, "zone1_agent6_promotion.json")

    pipeline_state["gold_records"]    = z1_a6_output["gold_records"]
    pipeline_state["conflict_log"]    = z1_a4_output["conflict_log"]
    pipeline_state["po_grn_recon"]    = z1_a4_output["po_grn_reconciliation"]

    print(f"\n{Fore.GREEN}  ✓ ZONE 1 COMPLETE — {len(pipeline_state['gold_records'])} records promoted to Gold{Style.RESET_ALL}")


    # Zone 2 — Gold Layer Intelligence

    print_zone_header(2, "GOLD LAYER INTELLIGENCE", "Demand-Supply Signal | Performance | Anomaly & Exception")

    # Agent 7 — Demand & Supply Signal Agent
    agent7 = DemandSupplySignalAgent()
    z2_a7_output = agent7.run({
        "inventory_df":        pipeline_state["clean_inventory"],
        "purchase_orders_df":  pipeline_state["clean_purchase_orders"],
        "shipments_df":        pipeline_state["clean_shipments"],
        "product_master":      data["product_master"],
        "warehouse_master":    data["warehouse_master"],
    })
    save_output(z2_a7_output, "zone2_agent7_demand_supply.json")

    # Agent 8 — Supplier & Carrier Performance Agent
    agent8 = SupplierCarrierPerformanceAgent()
    z2_a8_output = agent8.run({
        "gold_records":                pipeline_state["gold_records"],
        "carrier_master":              data["carrier_master"],
        "supplier_master":             data["supplier_master"],
        "goods_receipts_df":           data["goods_receipts"],
        "retailer_scorecards":         pipeline_state["structured_scorecards"],
        "carrier_performance_history": data["carrier_performance_history"],
    })
    save_output(z2_a8_output, "zone2_agent8_performance.json")

    # Agent 9 — Anomaly & Exception Agent
    agent9 = AnomalyExceptionAgent()
    z2_a9_output = agent9.run({
        "gold_records":         pipeline_state["gold_records"],
        "invoices_df":          pipeline_state["clean_invoices"],
        "goods_receipts_df":    data["goods_receipts"],
        "inventory_df":         pipeline_state["clean_inventory"],
        "purchase_orders_df":   pipeline_state["clean_purchase_orders"],
    })
    save_output(z2_a9_output, "zone2_agent9_exceptions.json")

    pipeline_state["risk_flags"]           = z2_a7_output["risk_flags"]
    pipeline_state["contingencies"]        = z2_a7_output["contingencies"]
    pipeline_state["supply_analysis"]      = z2_a7_output["supply_analysis"]
    pipeline_state["carrier_performance"]  = z2_a8_output["carrier_performance"]
    pipeline_state["supplier_performance"] = z2_a8_output["supplier_performance"]
    pipeline_state["performance_analysis"] = z2_a8_output["performance_analysis"]
    pipeline_state["exceptions"]           = z2_a9_output["exceptions"]
    pipeline_state["total_financial_exposure_usd"] = z2_a9_output["total_financial_exposure_usd"]

    print(f"\n{Fore.GREEN}  ✓ ZONE 2 COMPLETE — {len(pipeline_state['exceptions'])} exceptions, "
          f"${pipeline_state['total_financial_exposure_usd']:,.2f} total exposure{Style.RESET_ALL}")


    # Zone 3 — Reporting & Consumption

    print_zone_header(3, "REPORTING & CONSUMPTION", "KPI Reports | Role-Based Copilot | Proactive Insights")

    # Agent 10 — KPI Report Agent
    agent10 = KPIReportAgent()
    z3_a10_output = agent10.run({
        "gold_records":         pipeline_state["gold_records"],
        "carrier_performance":  pipeline_state["carrier_performance"],
        "exceptions":           pipeline_state["exceptions"],
        "inventory_df":         pipeline_state["clean_inventory"],
        "purchase_orders_df":   pipeline_state["clean_purchase_orders"],
    })
    save_output(z3_a10_output, "zone3_agent10_kpi_report.json")
    pipeline_state["kpi_values"] = z3_a10_output["kpi_values"]

    # Agent 11 — Role-Based Copilot Agent
    agent11 = RoleBasedCopilotAgent()
    z3_a11_output = agent11.run({
        **pipeline_state,
        "purchase_orders_df": pipeline_state["clean_purchase_orders"],
        "queries": [
            {"role": "logistics_planner",   "query": "Which shipments are at risk of missing their delivery window this week?"},
            {"role": "finance_manager",     "query": "What is our total invoice discrepancy exposure this month?"},
            {"role": "executive",           "query": "What is the single biggest supply chain risk I need to know about right now?"},
            {"role": "procurement_manager", "query": "Which suppliers have open performance issues I should address this week?"},
        ]
    })
    save_output(z3_a11_output, "zone3_agent11_copilot.json")

    # Agent 12 — Proactive Insight Agent
    agent12 = ProactiveInsightAgent()
    z3_a12_output = agent12.run({
        **pipeline_state,
    })
    save_output(z3_a12_output, "zone3_agent12_proactive.json")

    print(f"\n{Fore.GREEN}  ✓ ZONE 3 COMPLETE{Style.RESET_ALL}")

    # Pipeline Summary

    print(f"\n\n{Fore.CYAN}{'='*70}")
    print("  PIPELINE COMPLETE — SUMMARY")
    print(f"{'='*70}{Style.RESET_ALL}")

    summary = {
        "pipeline_completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zones_executed": 4,
        "agents_executed": 12,
        "zone_0_pre_ingestion": {
            "documents_structured":     z0_a1_output["total_extracted"],
            "records_quarantined":       sum(len(q) for q in z0_a2_output["quarantine_records"].values() if isinstance(q, pd.DataFrame)),
            "anomalies_detected":        len(z0_a3_output["anomalies"]),
            "shipments_risk_tagged":     sum(1 for r in z0_a3_output["priority_tagged_shipments"] if r.get("_risk_tag")),
        },
        "zone_1_processing": {
            "conflicts_resolved":        len(z1_a4_output["conflict_log"]),
            "records_enriched":          z1_a5_output["total_enriched"],
            "records_promoted_to_gold":  len(z1_a6_output["gold_records"]),
            "records_held_in_silver":    len(z1_a6_output["hold_queue"]),
        },
        "zone_2_gold_intelligence": {
            "supply_risk_flags":         len(z2_a7_output["risk_flags"]),
            "contingencies_identified":  len(z2_a7_output["contingencies"]),
            "carrier_alerts":            len(z2_a8_output["at_risk_carriers"]),
            "exceptions_detected":       len(z2_a9_output["exceptions"]),
            "total_financial_exposure":  f"${z2_a9_output['total_financial_exposure_usd']:,.2f}",
        },
        "zone_3_reporting": {
            "kpis_computed":             len(z3_a10_output["kpi_report"]),
            "copilot_queries_answered":  z3_a11_output["queries_answered"],
            "proactive_insights_sent":   z3_a12_output["total_alerts"],
        },
    }

    for zone, metrics in summary.items():
        if isinstance(metrics, dict):
            print(f"\n  {Fore.YELLOW}{zone.replace('_', ' ').upper()}{Style.RESET_ALL}")
            for k, v in metrics.items():
                print(f"    • {k.replace('_', ' ')}: {Fore.GREEN}{v}{Style.RESET_ALL}")
        else:
            print(f"  {k}: {v}")

    save_output(summary, "pipeline_summary.json")

    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"  All outputs saved to: ./{OUTPUT_DIR}/")
    print(f"  Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    return summary


if __name__ == "__main__":
    main()
