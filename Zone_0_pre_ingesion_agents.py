# Zone 0: Pre-Ingestion Agents
# Agents: Document & Unstructured Data, Data Quality Gate, Source Anomaly Detection

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from base_agent import BaseAgent
from config import DQ_SCORE_THRESHOLD, ANOMALY_VOLUME_THRESHOLD
from colorama import Fore, Style



# Agent 1: Document & Unstructured Data Agent

class DocumentAgent(BaseAgent):
    """
    INPUT : List of raw document dicts (simulating PDFs/Excel parsed to text)
    OUTPUT: Structured Delta-ready records with extraction metadata
    """

    def __init__(self):
        super().__init__("Document & Unstructured Data Agent", "Zone 0 — Pre-Ingestion")

    def run(self, input_data: dict) -> dict:
        self._print_header()
        raw_documents = input_data.get("raw_documents", [])
        self._print_section(f"Processing {len(raw_documents)} unstructured documents")

        structured_records = []
        extraction_log     = []

        for doc in raw_documents:
            self._print_section(f"Extracting: {doc.get('document_type')} — {doc.get('retailer_name', doc.get('supplier_id', ''))}")

            system_prompt = """You are a supply chain document extraction agent.
Your job is to validate and normalize extracted document data into clean structured records.
Respond ONLY with a valid JSON object. No extra text."""

            user_prompt = f"""
Validate and normalize the following supply chain document data.
Return a JSON object with:
- "normalized_record": the cleaned and validated data dict
- "extraction_quality": score 0-100
- "fields_extracted": count of non-null fields
- "issues_found": list of any data quality issues noticed
- "document_category": one of [retailer_scorecard, supplier_invoice, finance_report]

Raw document data:
{json.dumps(doc, indent=2)}
"""
            response_text = self.call_llm(system_prompt, user_prompt)
            try:
                parsed = json.loads(response_text)
            except Exception:
                parsed = {"normalized_record": doc, "extraction_quality": 70,
                          "fields_extracted": len(doc), "issues_found": [], "document_category": "unknown"}

            normalized = parsed.get("normalized_record", doc)
            normalized["_extraction_quality"] = parsed.get("extraction_quality", 85)
            normalized["_extracted_at"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            normalized["_document_category"]   = parsed.get("document_category", "unknown")

            structured_records.append(normalized)
            extraction_log.append({
                "document_type":     doc.get("document_type"),
                "source":            doc.get("retailer_name", doc.get("supplier_id", "unknown")),
                "fields_extracted":  parsed.get("fields_extracted", len(doc)),
                "extraction_quality":parsed.get("extraction_quality", 85),
                "issues_found":      parsed.get("issues_found", []),
                "document_category": parsed.get("document_category", "unknown"),
            })

            self._print_output("Extraction Quality", parsed.get("extraction_quality", 85))
            self._print_output("Issues Found", parsed.get("issues_found", []))

        output = {
            "structured_records": structured_records,
            "extraction_log":     extraction_log,
            "total_documents":    len(raw_documents),
            "total_extracted":    len(structured_records),
            "agent":              self.agent_name,
            "processed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"\n{Fore.GREEN}  ✓ Agent 1 Complete: {len(structured_records)} documents structured{Style.RESET_ALL}")
        return output


# Agent 2: data quality agent


class DataQualityGateAgent(BaseAgent):
    """
    INPUT : Dict of raw DataFrames (purchase_orders, shipments, invoices, etc.)
    OUTPUT: Cleaned records (passed), quarantine records (failed), DQ summary report
    """

    def __init__(self):
        super().__init__("Data Quality Gate Agent", "Zone 0 — Pre-Ingestion")

    def _score_record(self, record: dict, dataset_name: str) -> tuple[int, list]:
        """Rule-based DQ scoring before LLM enrichment."""
        issues = []
        score  = 100

        # Null checks
        null_fields = [k for k, v in record.items() if v is None or v == "" or (isinstance(v, float) and pd.isna(v))]
        if null_fields:
            score -= len(null_fields) * 15
            issues.append(f"Null fields: {null_fields}")

        # PO-specific checks
        if dataset_name == "purchase_orders":
            if record.get("ordered_qty", 0) <= 0:
                score -= 20; issues.append("Invalid ordered_qty <= 0")
            if record.get("unit_price", 0) <= 0:
                score -= 20; issues.append("Invalid unit_price <= 0")
            if not record.get("destination_wh"):
                score -= 25; issues.append("Missing destination warehouse")

        # Shipment-specific checks
        if dataset_name == "shipments":
            if not record.get("carrier_id"):
                score -= 20; issues.append("Missing carrier_id")
            if not record.get("dispatch_date"):
                score -= 15; issues.append("Missing dispatch_date")

        # Invoice-specific checks
        if dataset_name == "supplier_invoices":
            if record.get("total_billed", 0) <= 0:
                score -= 30; issues.append("Invalid total_billed")
            expected = round(record.get("billed_qty", 0) * record.get("unit_price", 0), 2)
            actual   = record.get("total_billed", 0)
            if abs(expected - actual) > 0.01:
                score -= 10; issues.append(f"Billing math mismatch: expected {expected}, got {actual}")

        return max(0, score), issues

    def run(self, input_data: dict) -> dict:
        self._print_header()
        datasets  = input_data.get("datasets", {})
        passed    = {}
        quarantine= {}
        dq_summary= []

        for dataset_name, df in datasets.items():
            if not isinstance(df, pd.DataFrame):
                continue

            self._print_section(f"Validating dataset: {dataset_name} ({len(df)} records)")
            passed_records     = []
            quarantine_records = []

            for _, row in df.iterrows():
                record = row.to_dict()
                score, issues = self._score_record(record, dataset_name)
                record["_dq_score"]   = score
                record["_dq_issues"]  = issues
                record["_dq_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if score >= DQ_SCORE_THRESHOLD:
                    passed_records.append(record)
                else:
                    record["_quarantine_reason"] = "; ".join(issues)
                    quarantine_records.append(record)

            passed[dataset_name]     = pd.DataFrame(passed_records)
            quarantine[dataset_name] = pd.DataFrame(quarantine_records) if quarantine_records else pd.DataFrame()

            dq_summary.append({
                "dataset":         dataset_name,
                "total_records":   len(df),
                "passed":          len(passed_records),
                "quarantined":     len(quarantine_records),
                "pass_rate_pct":   round(len(passed_records) / len(df) * 100, 1),
                "threshold":       DQ_SCORE_THRESHOLD,
            })

            print(f"  {dataset_name}: {len(passed_records)} passed / {len(quarantine_records)} quarantined")

        # LLM generates DQ executive summary
        system_prompt = """You are a data quality analyst for a supply chain system.
Respond ONLY with a valid JSON object."""

        user_prompt = f"""
Analyze this Data Quality Gate summary and respond with JSON:
{{
  "overall_health": "Good/Fair/Poor",
  "overall_health_score": <0-100>,
  "critical_issues": [list of critical issues],
  "recommended_actions": [list of actions],
  "narrative": "2-3 sentence executive summary"
}}

DQ Summary:
{json.dumps(dq_summary, indent=2)}
"""
        llm_response = self.call_llm(system_prompt, user_prompt)
        try:
            llm_analysis = json.loads(llm_response)
        except Exception:
            llm_analysis = {"overall_health": "Fair", "narrative": "DQ check completed."}

        self._print_output("DQ Summary Table", dq_summary)
        self._print_output("LLM DQ Analysis", llm_analysis)

        total_quarantined = sum(len(q) for q in quarantine.values() if isinstance(q, pd.DataFrame))
        print(f"\n{Fore.GREEN}  ✓ Agent 2 Complete: {total_quarantined} records quarantined, pipeline continues cleanly{Style.RESET_ALL}")

        return {
            "passed_datasets":    passed,
            "quarantine_records": quarantine,
            "dq_summary":         dq_summary,
            "llm_dq_analysis":    llm_analysis,
            "agent":              self.agent_name,
            "processed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 3: Source anamoly detection


class SourceAnomalyDetectionAgent(BaseAgent):
    """
    INPUT : Feed metadata (current volumes, timing) + risk feed events
    OUTPUT: Anomaly flags, priority tags on affected records, feed health log
    """

    def __init__(self):
        super().__init__("Source Anomaly Detection Agent", "Zone 0 — Pre-Ingestion")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        feed_metadata = input_data.get("feed_metadata", {})
        risk_events   = input_data.get("risk_events", [])
        shipments_df  = input_data.get("shipments_df", pd.DataFrame())

        # ── Rule-based anomaly detection
        anomalies   = []
        feed_health = []

        baselines = {
            "purchase_orders": 8,
            "shipments":       8,
            "inventory":       35,
            "supplier_invoices": 5,
            "risk_feed":       2,
        }

        for feed_name, current_count in feed_metadata.items():
            baseline = baselines.get(feed_name, 10)
            ratio    = current_count / baseline if baseline > 0 else 1.0
            status   = "Normal"
            anomaly_flag = False

            if ratio > ANOMALY_VOLUME_THRESHOLD:
                status = "Volume Spike"
                anomaly_flag = True
                anomalies.append({
                    "feed": feed_name, "type": "Volume Spike",
                    "current": current_count, "baseline": baseline,
                    "ratio": round(ratio, 2), "severity": "Medium"
                })
            elif ratio < 0.5:
                status = "Volume Drop"
                anomaly_flag = True
                anomalies.append({
                    "feed": feed_name, "type": "Volume Drop",
                    "current": current_count, "baseline": baseline,
                    "ratio": round(ratio, 2), "severity": "High"
                })

            feed_health.append({
                "feed": feed_name, "current_count": current_count,
                "baseline": baseline, "ratio": round(ratio, 2),
                "status": status, "anomaly_detected": anomaly_flag,
            })

        # ── Tag shipments with port risk
        priority_tagged = []
        if not shipments_df.empty:
            for _, row in shipments_df.iterrows():
                record = row.to_dict()
                risk_tag = None
                for event in risk_events:
                    if event.get("active") and event.get("severity") in ["High", "Medium"]:
                        affected_routes = event.get("affected_routes", "")
                        carrier_lane    = record.get("carrier_id", "")
                        # Tag shipments on affected routes
                        if (record.get("port_risk_flag") or
                            "Asia" in affected_routes and carrier_lane in ["CAR-002", "CAR-003"]):
                            risk_tag = {
                                "event_id":     event["event_id"],
                                "event_type":   event["event_type"],
                                "severity":     event["severity"],
                                "delay_days":   event["estimated_delay_days"],
                            }
                record["_risk_tag"]    = risk_tag
                record["_priority"]    = "HIGH" if risk_tag else "NORMAL"
                priority_tagged.append(record)

        tagged_count = sum(1 for r in priority_tagged if r.get("_risk_tag"))

        # ── LLM reasoning over anomalies
        system_prompt = """You are a supply chain source monitoring agent.
Respond ONLY with a valid JSON object."""

        user_prompt = f"""
Analyze the following feed anomalies and risk events.
Return JSON with:
{{
  "overall_feed_health": "Healthy/Degraded/Critical",
  "high_priority_actions": [list of immediate actions needed],
  "risk_exposure_summary": "1-2 sentences on total risk exposure",
  "affected_shipment_count": <number>,
  "estimated_financial_exposure_usd": <number>
}}

Feed Anomalies: {json.dumps(anomalies, indent=2)}
Active Risk Events: {json.dumps([e for e in risk_events if e.get('active')], indent=2)}
Shipments tagged with risk: {tagged_count}
"""
        llm_response = self.call_llm(system_prompt, user_prompt)
        try:
            llm_analysis = json.loads(llm_response)
        except Exception:
            llm_analysis = {"overall_feed_health": "Healthy", "risk_exposure_summary": "Monitoring active."}

        self._print_output("Feed Health", feed_health)
        self._print_output("Anomalies Detected", anomalies)
        self._print_output("Risk-Tagged Shipments", tagged_count)
        self._print_output("LLM Risk Analysis", llm_analysis)

        print(f"\n{Fore.GREEN}  ✓ Agent 3 Complete: {len(anomalies)} anomalies, {tagged_count} shipments risk-tagged{Style.RESET_ALL}")

        return {
            "feed_health":        feed_health,
            "anomalies":          anomalies,
            "priority_tagged_shipments": priority_tagged,
            "llm_risk_analysis":  llm_analysis,
            "agent":              self.agent_name,
            "processed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
