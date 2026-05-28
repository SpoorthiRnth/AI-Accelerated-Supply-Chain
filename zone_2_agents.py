# Zone 2: Gold Layer Intelligence Agents
# Agents: Demand & Supply Signal, Supplier & Carrier Performance, Anomaly & Exception

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from base_agent import BaseAgent
from colorama import Fore, Style


# Agent 7: Demand & Supply Signal Agent

class DemandSupplySignalAgent(BaseAgent):
    """
    INPUT : Gold inventory, purchase orders, shipments in transit, product master
    OUTPUT: Live demand-supply balance signal, risk flags, recommended actions
    """

    def __init__(self):
        super().__init__("Demand & Supply Signal Agent", "Zone 2 — Gold Layer")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        inventory_df    = input_data.get("inventory_df", pd.DataFrame())
        pos_df          = input_data.get("purchase_orders_df", pd.DataFrame())
        shipments_df    = input_data.get("shipments_df", pd.DataFrame())
        product_master  = input_data.get("product_master", pd.DataFrame())
        warehouse_master= input_data.get("warehouse_master", pd.DataFrame())

        # Compute supply-demand balance per SKU per warehouse
        balance_signals = []
        risk_flags      = []

        if not inventory_df.empty:
            for _, inv_row in inventory_df.iterrows():
                inv       = inv_row.to_dict()
                sku       = inv.get("sku")
                wh_id     = inv.get("warehouse_id")
                qty_hand  = inv.get("qty_on_hand", 0)
                reorder   = inv.get("reorder_point", 0)

                # In-transit supply for this SKU to this warehouse
                in_transit_qty = 0
                if not shipments_df.empty:
                    transit = shipments_df[
                        (shipments_df.get("destination_retailer", pd.Series(dtype=str)) == wh_id) |
                        (shipments_df.get("origin_warehouse", pd.Series(dtype=str)) == wh_id)
                    ] if "destination_retailer" in shipments_df.columns else pd.DataFrame()
                    # Approximate: open POs for this SKU = inbound supply
                    if not pos_df.empty:
                        open_pos = pos_df[
                            (pos_df["sku"] == sku) &
                            (pos_df["destination_wh"] == wh_id) &
                            (pos_df["status"].isin(["Open", "Confirmed", "In Transit"]))
                        ]
                        in_transit_qty = open_pos["ordered_qty"].sum() if not open_pos.empty else 0

                total_available = qty_hand + in_transit_qty
                gap             = total_available - reorder
                risk_level      = "Critical" if gap < 0 and qty_hand < reorder * 0.3 else \
                                  "High" if gap < 0 else \
                                  "Medium" if gap < reorder * 0.2 else "Low"

                signal = {
                    "sku":              sku,
                    "warehouse_id":     wh_id,
                    "qty_on_hand":      qty_hand,
                    "in_transit_qty":   in_transit_qty,
                    "total_available":  total_available,
                    "reorder_point":    reorder,
                    "supply_gap":       gap,
                    "risk_level":       risk_level,
                }
                balance_signals.append(signal)

                if risk_level in ["Critical", "High"]:
                    risk_flags.append(signal)

        # Focus LLM on critical signals only
        critical_signals = [s for s in balance_signals if s["risk_level"] in ["Critical", "High"]]

        # Find contingency options (same SKU, different warehouse with surplus)
        contingencies = []
        for flag in risk_flags:
            sku = flag["sku"]
            surplus_whs = [s for s in balance_signals
                           if s["sku"] == sku and s["supply_gap"] > 500 and s["warehouse_id"] != flag["warehouse_id"]]
            if surplus_whs:
                contingencies.append({
                    "at_risk_warehouse": flag["warehouse_id"],
                    "sku":               sku,
                    "shortfall":         abs(flag["supply_gap"]),
                    "contingency_from":  surplus_whs[0]["warehouse_id"],
                    "available_qty":     surplus_whs[0]["qty_on_hand"],
                })

        system_prompt = """You are a supply chain demand-supply balancing agent.
                    Analyze inventory signals and recommend actions to prevent stockouts.
                    Respond ONLY with valid JSON."""

                            user_prompt = f"""
                    Analyze these supply-demand signals and return JSON with:
                    {{
                    "overall_supply_health": "Healthy/At Risk/Critical",
                    "stockout_risks": [
                        {{
                        "sku": "<sku>",
                        "warehouse": "<wh>",
                        "days_to_stockout": <number>,
                        "impact": "<business impact>",
                        "recommended_action": "<specific action>",
                        "action_deadline": "<deadline>"
                        }}
                    ],
                    "contingency_recommendations": [list of cross-warehouse transfer recommendations],
                    "executive_summary": "2-3 sentence summary of supply position"
                    }}

Critical/High Risk Signals ({len(critical_signals)} items):
{json.dumps(critical_signals[:10], indent=2)}

Contingency Options Available:
{json.dumps(contingencies, indent=2)}
"""
        response = self.call_llm(system_prompt, user_prompt)
        try:
            analysis = json.loads(response)
        except Exception:
            analysis = {"overall_supply_health": "At Risk",
                        "executive_summary": f"{len(critical_signals)} critical supply signals detected."}

        self._print_output("Critical Risk Signals", len(critical_signals))
        self._print_output("Contingency Options", contingencies)
        self._print_output("LLM Supply Analysis", analysis)

        print(f"\n{Fore.GREEN}  ✓ Agent 7 Complete: {len(risk_flags)} supply risk flags raised, "
              f"{len(contingencies)} contingencies identified{Style.RESET_ALL}")

        return {
            "balance_signals":    balance_signals,
            "risk_flags":         risk_flags,
            "contingencies":      contingencies,
            "supply_analysis":    analysis,
            "agent":              self.agent_name,
            "processed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 8: Supplier & Carrier Performance Agent


class SupplierCarrierPerformanceAgent(BaseAgent):
    """
    INPUT : Gold shipments, carrier/supplier master, GRNs, retailer scorecards,
            carrier performance history
    OUTPUT: Performance scores, trend analysis, risk alerts, recommended actions
    """

    def __init__(self):
        super().__init__("Supplier & Carrier Performance Agent", "Zone 2 — Gold Layer")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        gold_records        = input_data.get("gold_records", [])
        carrier_master      = input_data.get("carrier_master", pd.DataFrame())
        supplier_master     = input_data.get("supplier_master", pd.DataFrame())
        grn_df              = input_data.get("goods_receipts_df", pd.DataFrame())
        scorecards          = input_data.get("retailer_scorecards", [])
        perf_history        = input_data.get("carrier_performance_history", pd.DataFrame())

        # Carrier Performance Analysis
        carrier_performance = []
        if not carrier_master.empty:
            for _, car_row in carrier_master.iterrows():
                car      = car_row.to_dict()
                car_id   = car["carrier_id"]

                # Historical trend
                history = []
                if not perf_history.empty:
                    hist = perf_history[perf_history["carrier_id"] == car_id].sort_values("period_date")
                    history = hist["ontime_pct"].tolist()

                # Shipment-level delays
                car_shipments = [r for r in gold_records if r.get("carrier_id") == car_id]
                delayed       = [s for s in car_shipments if s.get("status") == "Delayed"]
                delay_rate    = round(len(delayed) / len(car_shipments) * 100, 1) if car_shipments else 0

                # Retailer scorecard data for this carrier
                scorecard_scores = [s["overall_score"] for s in scorecards if s.get("carrier_id") == car_id]
                avg_scorecard    = round(sum(scorecard_scores) / len(scorecard_scores), 2) if scorecard_scores else None

                # Trend direction
                trend = "Stable"
                if len(history) >= 3:
                    if history[-1] < history[-3]:
                        trend = "Declining"
                    elif history[-1] > history[-3]:
                        trend = "Improving"

                carrier_performance.append({
                    "carrier_id":          car_id,
                    "carrier_name":        car["carrier_name"],
                    "sla_ontime_pct":      car["sla_ontime_pct"],
                    "current_score":       car["current_score"],
                    "delay_rate_pct":      delay_rate,
                    "performance_trend":   trend,
                    "history":             history,
                    "scorecard_avg":       avg_scorecard,
                    "sla_breached":        car["current_score"] < car["sla_ontime_pct"],
                    "active_shipments":    len(car_shipments),
                    "delayed_shipments":   len(delayed),
                })

        # Supplier Performance Analysis
        supplier_performance = []
        if not supplier_master.empty and not grn_df.empty:
            for _, sup_row in supplier_master.iterrows():
                sup    = sup_row.to_dict()
                sup_id = sup["supplier_id"]

                grns   = grn_df[grn_df.get("po_id", pd.Series()).isin([])] if grn_df.empty else grn_df
                # Simplified: check GRN conditions
                sup_grns     = grn_df.copy() if not grn_df.empty else pd.DataFrame()
                damage_issues = len(sup_grns[sup_grns["condition"] != "Good"]) if not sup_grns.empty else 0

                supplier_performance.append({
                    "supplier_id":          sup_id,
                    "supplier_name":        sup["supplier_name"],
                    "sla_fulfillment_pct":  sup["sla_fulfillment_pct"],
                    "risk_rating":          sup["risk_rating"],
                    "category":             sup["category"],
                    "damage_issues":        damage_issues,
                })

        # LLM Performance Analysis
        at_risk_carriers = [c for c in carrier_performance
                            if c["sla_breached"] or c["performance_trend"] == "Declining"]

        system_prompt = """You are a supply chain performance management agent.
                Analyze carrier and supplier performance data and identify risks.
                Respond ONLY with valid JSON."""

        user_prompt = f"""
                        Analyze the following performance data and return JSON with:
                        {{
                        "carrier_alerts": [
                            {{
                            "carrier_id": "<id>",
                            "carrier_name": "<name>",
                            "alert_type": "<SLA Breach/Declining Trend/etc>",
                            "severity": "High/Medium/Low",
                            "root_cause_hypothesis": "<hypothesis>",
                            "recommended_action": "<action>",
                            "timeline": "<when to act>"
                            }}
                        ],
                        "supplier_alerts": [similar structure],
                        "performance_narrative": "2-3 sentence executive summary"
                        }}

                        At-Risk Carriers ({len(at_risk_carriers)}):
                        {json.dumps(at_risk_carriers, indent=2)}

                        All Carrier Performance:
                        {json.dumps(carrier_performance, indent=2)}

                        Supplier Performance:
                        {json.dumps(supplier_performance[:3], indent=2)}

                        Retailer Scorecards:
                        {json.dumps(scorecards, indent=2)}
                        """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            analysis = json.loads(response)
        except Exception:
            analysis = {"carrier_alerts": [], "supplier_alerts": [],
                        "performance_narrative": "Performance analysis completed."}

        self._print_output("Carrier Performance", carrier_performance)
        self._print_output("At-Risk Carriers", len(at_risk_carriers))
        self._print_output("LLM Performance Analysis", analysis)

        print(f"\n{Fore.GREEN}  ✓ Agent 8 Complete: {len(at_risk_carriers)} carrier alerts, "
              f"performance trends analysed{Style.RESET_ALL}")

        return {
            "carrier_performance":  carrier_performance,
            "supplier_performance": supplier_performance,
            "performance_analysis": analysis,
            "at_risk_carriers":     at_risk_carriers,
            "agent":                self.agent_name,
            "processed_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 9: Anamoly & Exception Agent

class AnomalyExceptionAgent(BaseAgent):
    """
    INPUT : Gold shipments, invoices, GRNs, inventory
    OUTPUT: Exception log with severity, financial impact, owner, recommended action
    """

    def __init__(self):
        super().__init__("Anomaly & Exception Agent", "Zone 2 — Gold Layer")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        gold_records  = input_data.get("gold_records", [])
        invoices_df   = input_data.get("invoices_df", pd.DataFrame())
        grn_df        = input_data.get("goods_receipts_df", pd.DataFrame())
        inventory_df  = input_data.get("inventory_df", pd.DataFrame())
        pos_df        = input_data.get("purchase_orders_df", pd.DataFrame())

        exceptions = []

        # Exception 1: Invoice quantity vs GRN quantity mismatch
        if not invoices_df.empty and not grn_df.empty:
            for _, inv_row in invoices_df.iterrows():
                inv = inv_row.to_dict()
                grn_match = grn_df[grn_df["po_id"] == inv.get("po_id")]
                if not grn_match.empty:
                    grn         = grn_match.iloc[0].to_dict()
                    billed_qty  = inv.get("billed_qty", 0)
                    received_qty= grn.get("received_qty", 0)
                    gap         = billed_qty - received_qty
                    if abs(gap) > 0:
                        financial_impact = round(abs(gap) * inv.get("unit_price", 0), 2)
                        exceptions.append({
                            "exception_id":      f"EXC-{len(exceptions)+1:03d}",
                            "type":              "Invoice-GRN Quantity Mismatch",
                            "severity":          "High" if financial_impact > 10000 else "Medium",
                            "po_id":             inv.get("po_id"),
                            "invoice_id":        inv.get("invoice_id"),
                            "supplier_id":       inv.get("supplier_id"),
                            "billed_qty":        billed_qty,
                            "received_qty":      received_qty,
                            "qty_gap":           gap,
                            "unit_price":        inv.get("unit_price"),
                            "financial_impact_usd": financial_impact,
                            "owner":             "Finance & Procurement",
                            "detected_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

        # Exception 2: Shipment delays on same lane
        if gold_records:
            from collections import Counter
            delayed    = [r for r in gold_records if r.get("status") == "Delayed"]
            lane_counts= Counter(f"{r.get('origin_warehouse')}-{r.get('destination_retailer')}"
                                  for r in delayed)
            for lane, count in lane_counts.items():
                if count >= 2:
                    exceptions.append({
                        "exception_id":  f"EXC-{len(exceptions)+1:03d}",
                        "type":          "Lane Delay Cluster",
                        "severity":      "High" if count >= 3 else "Medium",
                        "lane":          lane,
                        "delayed_count": count,
                        "financial_impact_usd": count * 5000,  # estimated impact
                        "owner":         "Logistics Operations",
                        "detected_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

        # Exception 3: Goods damage on receipt
        if not grn_df.empty:
            damaged = grn_df[grn_df["damaged_qty"] > 0]
            for _, dmg_row in damaged.iterrows():
                dmg = dmg_row.to_dict()
                # Get unit cost from PO
                unit_price = 0
                if not pos_df.empty:
                    po_match = pos_df[pos_df["po_id"] == dmg.get("po_id")]
                    if not po_match.empty:
                        unit_price = po_match.iloc[0].get("unit_price", 0)
                financial_impact = round(dmg.get("damaged_qty", 0) * unit_price, 2)
                exceptions.append({
                    "exception_id":        f"EXC-{len(exceptions)+1:03d}",
                    "type":                "Goods Damage on Receipt",
                    "severity":            "Medium",
                    "grn_id":              dmg.get("grn_id"),
                    "po_id":               dmg.get("po_id"),
                    "damaged_qty":         dmg.get("damaged_qty"),
                    "financial_impact_usd":financial_impact,
                    "owner":               "Warehouse & Quality",
                    "detected_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Exception 4: Inventory depletion anomaly
        if not inventory_df.empty:
            critical_inv = inventory_df[
                (inventory_df["below_reorder"] == True) &
                (inventory_df["qty_on_hand"] < inventory_df["reorder_point"] * 0.3)
            ]
            for _, inv_row in critical_inv.iterrows():
                inv = inv_row.to_dict()
                exceptions.append({
                    "exception_id":        f"EXC-{len(exceptions)+1:03d}",
                    "type":                "Critical Inventory Depletion",
                    "severity":            "Critical",
                    "warehouse_id":        inv.get("warehouse_id"),
                    "sku":                 inv.get("sku"),
                    "qty_on_hand":         inv.get("qty_on_hand"),
                    "reorder_point":       inv.get("reorder_point"),
                    "depletion_pct":       round(inv["qty_on_hand"] / inv["reorder_point"] * 100, 1),
                    "financial_impact_usd":inv.get("qty_on_hand", 0) * 50,
                    "owner":               "Inventory Planning",
                    "detected_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        total_financial_exposure = sum(e.get("financial_impact_usd", 0) for e in exceptions)

        # LLM exception analysis
        system_prompt = """You are a supply chain exception management agent.
                Prioritize exceptions, quantify risk, and assign ownership.
                Respond ONLY with valid JSON."""

                        user_prompt = f"""
                Analyze these supply chain exceptions and return JSON with:
                {{
                "total_exceptions": <number>,
                "critical_count": <number>,
                "total_financial_exposure_usd": <number>,
                "top_priority_exceptions": [top 3 exceptions with recommended resolution steps],
                "exception_narrative": "2-3 sentence executive summary of exception landscape"
                }}

                    Exceptions detected ({len(exceptions)}):
                    {json.dumps(exceptions, indent=2)}
                    """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            analysis = json.loads(response)
        except Exception:
            analysis = {"total_exceptions": len(exceptions),
                        "total_financial_exposure_usd": total_financial_exposure,
                        "exception_narrative": f"{len(exceptions)} exceptions detected."}

        self._print_output("Exceptions Detected", len(exceptions))
        self._print_output("Total Financial Exposure", f"${total_financial_exposure:,.2f}")
        self._print_output("Exception Log", exceptions)
        self._print_output("LLM Exception Analysis", analysis)

        print(f"\n{Fore.GREEN}  ✓ Agent 9 Complete: {len(exceptions)} exceptions, "
              f"${total_financial_exposure:,.2f} total exposure{Style.RESET_ALL}")

        return {
            "exceptions":            exceptions,
            "exception_analysis":    analysis,
            "total_financial_exposure_usd": total_financial_exposure,
            "agent":                 self.agent_name,
            "processed_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
