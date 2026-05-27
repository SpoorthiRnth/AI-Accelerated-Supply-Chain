# Zone 1: Processing Agents (Bronze, Silver, Gold)
# Agents: Reconciliation, Semantic Enrichment, Data Classification and Promotion

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from base_agent import BaseAgent
from config import PROMOTION_THRESHOLD
from colorama import Fore, Style



# Agent 4: Reconciliation Agent


class ReconciliationAgent(BaseAgent):
    """
    INPUT : Bronze layer shipments (with tms_status vs erp_status conflicts),
            purchase orders, goods receipts
    OUTPUT: Reconciled Silver records, conflict log with resolution rationale
    """

    def __init__(self):
        super().__init__("Reconciliation Agent", "Zone 1 — Processing (Bronze → Silver)")

    def _detect_conflicts(self, row: dict) -> list:
        conflicts = []
        tms = row.get("tms_status", "")
        erp = row.get("erp_status", "")
        if tms and erp and tms != erp:
            conflicts.append({
                "field":      "status",
                "tms_value":  tms,
                "erp_value":  erp,
                "conflict":   f"TMS says '{tms}' but ERP says '{erp}'",
            })
        return conflicts

    def run(self, input_data: dict) -> dict:
        self._print_header()

        shipments_df = input_data.get("shipments_df", pd.DataFrame())
        pos_df       = input_data.get("purchase_orders_df", pd.DataFrame())
        grn_df       = input_data.get("goods_receipts_df", pd.DataFrame())

        reconciled_records = []
        conflict_log       = []

        for _, row in shipments_df.iterrows():
            record    = row.to_dict()
            conflicts = self._detect_conflicts(record)

            if conflicts:
                # LLM reconciliation reasoning
                system_prompt = """You are a supply chain data reconciliation agent.
                    When data conflicts exist between systems, apply reconciliation logic.
                    TMS is ground truth for physical shipment status.
                    ERP is ground truth for financial/order status.
                    Respond ONLY with valid JSON."""

                user_prompt = f"""
                    Reconcile the following supply chain data conflict.
                    Return JSON with:
                    {{
                    "reconciled_status": "<final authoritative status>",
                    "resolution_rationale": "<why this resolution was chosen>",
                    "erp_correction_needed": <true/false>,
                    "confidence": <0-100>
                    }}

                        Shipment ID: {record.get('shipment_id')}
                        PO ID: {record.get('po_id')}
                        Conflicts detected: {json.dumps(conflicts, indent=2)}
                        Full record: {json.dumps({k: str(v) for k, v in record.items()}, indent=2)}
                        """
                response = self.call_llm(system_prompt, user_prompt)
                try:
                    resolution = json.loads(response)
                except Exception:
                    resolution = {
                        "reconciled_status":    record.get("tms_status"),
                        "resolution_rationale": "TMS is ground truth for physical status",
                        "erp_correction_needed": True,
                        "confidence": 80,
                    }

                record["_reconciled_status"]    = resolution.get("reconciled_status", record.get("tms_status"))
                record["_reconciliation_note"]  = resolution.get("resolution_rationale", "")
                record["_erp_correction_needed"]= resolution.get("erp_correction_needed", False)
                record["_reconciliation_confidence"] = resolution.get("confidence", 80)
                record["_had_conflict"]         = True

                conflict_log.append({
                    "shipment_id":   record.get("shipment_id"),
                    "conflicts":     conflicts,
                    "resolution":    resolution,
                    "resolved_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                print(f"  Conflict resolved on {record.get('shipment_id')}: "
                      f"'{conflicts[0]['tms_value']}' vs '{conflicts[0]['erp_value']}' "
                      f"→ '{record['_reconciled_status']}'")
            else:
                record["_reconciled_status"]   = record.get("tms_status", record.get("status"))
                record["_reconciliation_note"] = "No conflict — single source of truth"
                record["_had_conflict"]        = False

            record["_reconciled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record["_layer"]         = "Silver"
            reconciled_records.append(record)

        # PO-GRN quantity reconciliation
        po_grn_reconciliation = []
        if not pos_df.empty and not grn_df.empty:
            for _, grn_row in grn_df.iterrows():
                grn = grn_row.to_dict()
                po  = pos_df[pos_df["po_id"] == grn["po_id"]]
                if not po.empty:
                    po_row = po.iloc[0].to_dict()
                    ordered  = po_row.get("ordered_qty", 0)
                    received = grn.get("received_qty", 0)
                    gap      = ordered - received
                    po_grn_reconciliation.append({
                        "po_id":         grn["po_id"],
                        "grn_id":        grn["grn_id"],
                        "ordered_qty":   ordered,
                        "received_qty":  received,
                        "quantity_gap":  gap,
                        "gap_pct":       round(gap / ordered * 100, 1) if ordered > 0 else 0,
                        "status":        "Short Delivery" if gap > 0 else "Over Delivery" if gap < 0 else "Full Delivery",
                    })

        self._print_output("Conflicts Resolved", len(conflict_log))
        self._print_output("Conflict Log", conflict_log)
        self._print_output("PO-GRN Reconciliation", po_grn_reconciliation)

        print(f"\n{Fore.GREEN}  ✓ Agent 4 Complete: {len(reconciled_records)} records reconciled, "
              f"{len(conflict_log)} conflicts resolved{Style.RESET_ALL}")

        return {
            "silver_shipments":      pd.DataFrame(reconciled_records),
            "conflict_log":          conflict_log,
            "po_grn_reconciliation": po_grn_reconciliation,
            "agent":                 self.agent_name,
            "processed_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 5: Semantic Enrichment Agent


class SemanticEnrichmentAgent(BaseAgent):
    """
    INPUT : Silver shipment records (codes/IDs) + all master data tables
    OUTPUT: Gold records in full business language with contextual attributes
    """

    def __init__(self):
        super().__init__("Semantic Enrichment Agent", "Zone 1 — Processing (Silver → Gold)")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        silver_df       = input_data.get("silver_shipments", pd.DataFrame())
        supplier_master = input_data.get("supplier_master", pd.DataFrame())
        carrier_master  = input_data.get("carrier_master", pd.DataFrame())
        product_master  = input_data.get("product_master", pd.DataFrame())
        warehouse_master= input_data.get("warehouse_master", pd.DataFrame())
        retailer_master = input_data.get("retailer_master", pd.DataFrame())
        pos_df          = input_data.get("purchase_orders_df", pd.DataFrame())

        enriched_records = []

        for _, row in silver_df.iterrows():
            record = row.to_dict()

            # Join carrier context
            carrier_ctx = {}
            if not carrier_master.empty and record.get("carrier_id"):
                c = carrier_master[carrier_master["carrier_id"] == record["carrier_id"]]
                if not c.empty:
                    carrier_ctx = c.iloc[0].to_dict()

            # Join warehouse context
            wh_ctx = {}
            if not warehouse_master.empty and record.get("origin_warehouse"):
                w = warehouse_master[warehouse_master["warehouse_id"] == record["origin_warehouse"]]
                if not w.empty:
                    wh_ctx = w.iloc[0].to_dict()

            # Join retailer context
            ret_ctx = {}
            if not retailer_master.empty and record.get("destination_retailer"):
                r = retailer_master[retailer_master["retailer_id"] == record["destination_retailer"]]
                if not r.empty:
                    ret_ctx = r.iloc[0].to_dict()

            # Join PO + product context
            po_ctx  = {}
            prd_ctx = {}
            if not pos_df.empty and record.get("po_id"):
                p = pos_df[pos_df["po_id"] == record["po_id"]]
                if not p.empty:
                    po_ctx = p.iloc[0].to_dict()
                    sku = po_ctx.get("sku", "")
                    if not product_master.empty and sku:
                        pr = product_master[product_master["sku"] == sku]
                        if not pr.empty:
                            prd_ctx = pr.iloc[0].to_dict()

            # Build enrichment context for LLM
            system_prompt = """You are a semantic enrichment agent for a supply chain.
                Your job is to translate technical record data into business-meaningful context.
                Respond ONLY with valid JSON."""

                            user_prompt = f"""
                Enrich this supply chain shipment record with business context.
                Return JSON with:
                {{
                "business_summary": "1-2 sentence plain English description of this shipment",
                "risk_level": "Low/Medium/High",
                "business_priority": "Standard/Elevated/Critical",
                "key_context_flags": [list of important business flags],
                "sla_at_risk": <true/false>,
                "enrichment_tags": [relevant business tags]
                }}

                Shipment: {json.dumps({k: str(v) for k, v in record.items() if not k.startswith('_')}, indent=2)}
                Carrier: {json.dumps(carrier_ctx, indent=2, default=str)}
                Origin Warehouse: {json.dumps(wh_ctx, indent=2, default=str)}
                Destination Retailer: {json.dumps(ret_ctx, indent=2, default=str)}
                Product: {json.dumps(prd_ctx, indent=2, default=str)}
                """
            response = self.call_llm(system_prompt, user_prompt)
            try:
                enrichment = json.loads(response)
            except Exception:
                enrichment = {"business_summary": "Shipment in progress", "risk_level": "Low",
                              "business_priority": "Standard", "sla_at_risk": False}

            # Attach enriched context
            record["_carrier_name"]        = carrier_ctx.get("carrier_name", record.get("carrier_id"))
            record["_carrier_sla"]         = carrier_ctx.get("sla_ontime_pct")
            record["_carrier_score"]       = carrier_ctx.get("current_score")
            record["_origin_wh_name"]      = wh_ctx.get("warehouse_name", record.get("origin_warehouse"))
            record["_origin_wh_util_pct"]  = wh_ctx.get("current_utilization_pct")
            record["_retailer_name"]       = ret_ctx.get("retailer_name", record.get("destination_retailer"))
            record["_retailer_priority"]   = ret_ctx.get("priority")
            record["_product_name"]        = prd_ctx.get("product_name", "")
            record["_product_category"]    = prd_ctx.get("category", "")
            record["_unit_cost"]           = prd_ctx.get("unit_cost")
            record["_business_summary"]    = enrichment.get("business_summary", "")
            record["_risk_level"]          = enrichment.get("risk_level", "Low")
            record["_business_priority"]   = enrichment.get("business_priority", "Standard")
            record["_sla_at_risk"]         = enrichment.get("sla_at_risk", False)
            record["_enrichment_tags"]     = enrichment.get("enrichment_tags", [])
            record["_key_context_flags"]   = enrichment.get("key_context_flags", [])
            record["_enriched_at"]         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record["_layer"]               = "Pre-Gold"

            enriched_records.append(record)

            print(f"  {record.get('shipment_id')} → {record['_carrier_name']} | "
                  f"{record['_product_name']} | Priority: {record['_business_priority']} | "
                  f"Risk: {record['_risk_level']}")

        self._print_output("Sample Enriched Record", {
            k: v for k, v in enriched_records[0].items() if k.startswith("_")
        } if enriched_records else {})

        print(f"\n{Fore.GREEN}  ✓ Agent 5 Complete: {len(enriched_records)} records enriched with business context{Style.RESET_ALL}")

        return {
            "enriched_records":  enriched_records,
            "total_enriched":    len(enriched_records),
            "agent":             self.agent_name,
            "processed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 6: Data Classification and Promotion Agent


class DataPromotionAgent(BaseAgent):
    """
    INPUT : Enriched pre-Gold records with DQ scores, reconciliation flags
    OUTPUT: Gold layer records (promoted), hold queue (not ready), promotion audit log
    """

    def __init__(self):
        super().__init__("Data Classification & Promotion Agent", "Zone 1 — Processing (Silver → Gold)")

    def _compute_promotion_score(self, record: dict) -> tuple[int, list]:
        score  = 100
        issues = []

        dq_score = record.get("_dq_score", 100)
        if dq_score < 75:
            score -= 30; issues.append(f"DQ score too low: {dq_score}")

        if not record.get("_carrier_name"):
            score -= 20; issues.append("Missing carrier enrichment")
        if not record.get("_retailer_name"):
            score -= 15; issues.append("Missing retailer enrichment")
        if not record.get("_product_name"):
            score -= 15; issues.append("Missing product enrichment")
        if record.get("_had_conflict") and not record.get("_reconciled_status"):
            score -= 20; issues.append("Unresolved conflict")

        return max(0, score), issues

    def run(self, input_data: dict) -> dict:
        self._print_header()

        enriched_records = input_data.get("enriched_records", [])
        gold_records     = []
        hold_queue       = []
        audit_log        = []

        for record in enriched_records:
            score, issues = self._compute_promotion_score(record)
            decision      = "PROMOTED" if score >= PROMOTION_THRESHOLD else "HELD"

            if decision == "PROMOTED":
                record["_promotion_score"] = score
                record["_layer"]           = "Gold"
                record["_promoted_at"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                gold_records.append(record)
            else:
                record["_promotion_score"]  = score
                record["_hold_reasons"]     = issues
                record["_layer"]            = "Silver-Hold"
                hold_queue.append(record)

            audit_log.append({
                "shipment_id":      record.get("shipment_id"),
                "promotion_score":  score,
                "decision":         decision,
                "hold_reasons":     issues if decision == "HELD" else [],
                "evaluated_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            status_color = Fore.GREEN if decision == "PROMOTED" else Fore.YELLOW
            print(f"  {status_color}{record.get('shipment_id')}: score={score} → {decision}{Style.RESET_ALL}"
                  + (f" | {issues}" if issues else ""))

        # LLM generates promotion summary
        system_prompt = """You are a data governance agent for a supply chain lakehouse.
            Respond ONLY with valid JSON."""

                    user_prompt = f"""
            Summarize the data promotion cycle results.
            Return JSON with:
            {{
            "gold_layer_health": "Excellent/Good/Fair/Poor",
            "promotion_rate_pct": <number>,
            "summary": "2-3 sentence narrative",
            "data_gaps_identified": [list of patterns in held records]
            }}

            Promoted: {len(gold_records)}
            Held: {len(hold_queue)}
            Audit log: {json.dumps(audit_log, indent=2)}
            """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            summary = json.loads(response)
        except Exception:
            summary = {"gold_layer_health": "Good",
                       "promotion_rate_pct": round(len(gold_records)/len(enriched_records)*100, 1) if enriched_records else 0}

        self._print_output("Promotion Summary", summary)
        self._print_output("Audit Log", audit_log)

        print(f"\n{Fore.GREEN}  ✓ Agent 6 Complete: {len(gold_records)} promoted to Gold, "
              f"{len(hold_queue)} held in Silver{Style.RESET_ALL}")

        return {
            "gold_records":   gold_records,
            "hold_queue":     hold_queue,
            "audit_log":      audit_log,
            "promotion_summary": summary,
            "agent":          self.agent_name,
            "processed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
