# Zone 3: Reporting Layer Agents
# Agents: KPI Report Generation, Role-Based Copilot, Proactive Insight

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from base_agent import BaseAgent
from colorama import Fore, Style



# Agent 10: KPI Report Generation Agent


class KPIReportAgent(BaseAgent):
    """
    INPUT : KPI definitions + Gold layer data + agent outputs from zones 0-2
    OUTPUT: Formatted KPI report with actuals, trends, narrative commentary
    """

    def __init__(self):
        super().__init__("KPI Report Generation Agent", "Zone 3 — Reporting")

    def _compute_kpis(self, data: dict) -> dict:
        """Rule-based KPI computation from Gold data."""
        gold_records        = data.get("gold_records", [])
        carrier_performance = data.get("carrier_performance", [])
        exceptions          = data.get("exceptions", [])
        inventory_df        = data.get("inventory_df", pd.DataFrame())
        pos_df              = data.get("purchase_orders_df", pd.DataFrame())

        # KPI 1: On-time delivery rate
        delivered  = [r for r in gold_records if r.get("status") == "Delivered"]
        delayed    = [r for r in gold_records if r.get("status") == "Delayed"]
        total_shp  = len(gold_records)
        otd_rate   = round(len(delivered) / total_shp * 100, 1) if total_shp > 0 else 0

        # KPI 2: Average carrier SLA compliance
        sla_scores = [c["current_score"] for c in carrier_performance]
        avg_sla    = round(sum(sla_scores) / len(sla_scores), 1) if sla_scores else 0

        # KPI 3: Inventory health — % SKUs below reorder
        if not inventory_df.empty:
            total_sku_wh   = len(inventory_df)
            below_reorder  = len(inventory_df[inventory_df["below_reorder"] == True])
            inv_health_pct = round((1 - below_reorder / total_sku_wh) * 100, 1) if total_sku_wh else 100
        else:
            inv_health_pct = 100

        # KPI 4: Open exceptions by severity
        critical_exc = len([e for e in exceptions if e.get("severity") == "Critical"])
        high_exc     = len([e for e in exceptions if e.get("severity") == "High"])
        total_exc    = len(exceptions)

        # KPI 5: PO fulfillment rate
        if not pos_df.empty:
            closed_pos     = len(pos_df[pos_df["status"] == "Closed"])
            po_fulfill_pct = round(closed_pos / len(pos_df) * 100, 1)
        else:
            po_fulfill_pct = 0

        # KPI 6: Financial exposure from exceptions
        total_exposure = sum(e.get("financial_impact_usd", 0) for e in exceptions)

        return {
            "on_time_delivery_rate_pct":   otd_rate,
            "avg_carrier_sla_compliance":  avg_sla,
            "inventory_health_pct":        inv_health_pct,
            "open_exceptions_total":       total_exc,
            "open_exceptions_critical":    critical_exc,
            "open_exceptions_high":        high_exc,
            "po_fulfillment_rate_pct":     po_fulfill_pct,
            "total_financial_exposure_usd": total_exposure,
            "active_shipments":            total_shp,
            "delayed_shipments":           len(delayed),
        }

    def run(self, input_data: dict) -> dict:
        self._print_header()

        kpi_definitions = input_data.get("kpi_definitions", [
            "On-Time Delivery Rate by Carrier",
            "Inventory Coverage & Health",
            "Open Exception Count by Severity",
            "Purchase Order Fulfillment Rate",
            "Total Financial Exposure from Exceptions",
            "Average Carrier SLA Compliance",
        ])

        kpi_values = self._compute_kpis(input_data)

        # KPI status thresholds
        def kpi_status(metric, value):
            thresholds = {
                "on_time_delivery_rate_pct":  (95, 90),
                "avg_carrier_sla_compliance": (95, 90),
                "inventory_health_pct":       (90, 80),
                "po_fulfillment_rate_pct":    (90, 75),
            }
            if metric in thresholds:
                green, amber = thresholds[metric]
                return "🟢 On Target" if value >= green else "🟡 At Risk" if value >= amber else "🔴 Breached"
            return "ℹ️ Informational"

        kpi_report = []
        for key, value in kpi_values.items():
            kpi_report.append({
                "kpi":    key.replace("_", " ").title(),
                "value":  value,
                "status": kpi_status(key, value),
                "unit":   "%" if "pct" in key or "rate" in key else "USD" if "usd" in key else "count",
            })

        # LLM generates narrative report
        system_prompt = """You are a supply chain analytics reporting agent.
            Generate a professional weekly KPI report with clear business narrative.
            Respond ONLY with valid JSON."""

        user_prompt = f"""
            Generate a comprehensive weekly supply chain KPI report.
            Return JSON with:
            {{
            "report_title": "Weekly Supply Chain Performance Report",
            "report_date": "{datetime.now().strftime('%B %d, %Y')}",
            "executive_summary": "3-4 sentence overall performance narrative",
            "kpi_commentary": {{
                "<kpi_name>": "1-2 sentence commentary explaining the KPI result and trend"
            }},
            "top_concerns": [top 3 concerns requiring management attention],
            "positive_highlights": [top 2 positive performance highlights],
            "recommended_actions": [specific actions for this week]
            }}

            KPI Values:
            {json.dumps(kpi_values, indent=2)}

            KPI Definitions Tracked:
            {json.dumps(kpi_definitions, indent=2)}

            Context - Carrier Performance:
            {json.dumps(input_data.get('carrier_performance', [])[:3], indent=2)}

            Context - Active Exceptions:
            {json.dumps(input_data.get('exceptions', [])[:5], indent=2)}
            """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            report_narrative = json.loads(response)
        except Exception:
            report_narrative = {"executive_summary": "Weekly KPI report generated.", "report_title": "Weekly Supply Chain KPI Report"}

        self._print_output("KPI Scorecard", kpi_report)
        self._print_output("Report Narrative", report_narrative)

        print(f"\n{Fore.GREEN}  ✓ Agent 10 Complete: KPI report generated with {len(kpi_report)} KPIs{Style.RESET_ALL}")

        return {
            "kpi_values":       kpi_values,
            "kpi_report":       kpi_report,
            "report_narrative": report_narrative,
            "report_date":      datetime.now().strftime("%Y-%m-%d"),
            "agent":            self.agent_name,
            "processed_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }



# Agent 11: Role-Based Copilot Agent


class RoleBasedCopilotAgent(BaseAgent):
    """
    INPUT : User query + role context + Gold layer data + all agent outputs
    OUTPUT: Role-scoped, actionable answer in plain language
    """

    ROLE_PERMISSIONS = {
        "logistics_planner": ["shipments", "carriers", "inventory", "risk_flags", "supply_signals"],
        "finance_manager":   ["invoices", "exceptions_financial", "po_values", "exposure"],
        "executive":         ["kpi_summary", "risk_summary", "financial_exposure", "strategic_alerts"],
        "procurement_manager": ["purchase_orders", "supplier_performance", "invoice_discrepancies"],
    }

    ROLE_CONTEXT = {
        "logistics_planner":    "You are helping a Logistics Planner who manages shipments and carrier relationships.",
        "finance_manager":      "You are helping a Finance Manager focused on invoice accuracy, cost control, and financial exposure.",
        "executive":            "You are helping a C-level Executive who needs strategic supply chain risk intelligence.",
        "procurement_manager":  "You are helping a Procurement Manager who manages supplier relationships and purchase orders.",
    }

    def __init__(self):
        super().__init__("Role-Based Copilot Agent", "Zone 3 — Reporting")

    def _build_role_context(self, role: str, data: dict) -> dict:
        """Filter and prepare data relevant to the role."""
        permissions = self.ROLE_PERMISSIONS.get(role, [])
        role_data   = {}

        if "shipments" in permissions:
            role_data["active_shipments"]  = [
                {k: v for k, v in r.items() if not k.startswith("_dq") and not k.startswith("_prom")}
                for r in data.get("gold_records", [])[:5]
            ]
            role_data["delayed_shipments"] = [r for r in data.get("gold_records", []) if r.get("status") == "Delayed"]

        if "carriers" in permissions:
            role_data["carrier_performance"] = data.get("carrier_performance", [])

        if "risk_flags" in permissions:
            role_data["supply_risk_flags"]   = data.get("risk_flags", [])[:5]
            role_data["feed_anomalies"]      = data.get("anomalies", [])

        if "invoices" in permissions or "exceptions_financial" in permissions:
            role_data["financial_exceptions"] = [
                e for e in data.get("exceptions", [])
                if e.get("type") in ["Invoice-GRN Quantity Mismatch", "Goods Damage on Receipt"]
            ]
            role_data["total_exposure_usd"]  = data.get("total_financial_exposure_usd", 0)

        if "kpi_summary" in permissions:
            role_data["kpi_values"]          = data.get("kpi_values", {})
            role_data["performance_analysis"] = data.get("performance_analysis", {})

        if "risk_summary" in permissions:
            role_data["all_exceptions"]       = data.get("exceptions", [])
            role_data["supply_analysis"]      = data.get("supply_analysis", {})
            role_data["carrier_alerts"]       = data.get("performance_analysis", {}).get("carrier_alerts", [])

        if "purchase_orders" in permissions:
            pos = data.get("purchase_orders_df")
            if isinstance(pos, pd.DataFrame):
                role_data["purchase_orders"] = pos.head(5).to_dict(orient="records")

        if "supplier_performance" in permissions:
            role_data["supplier_performance"] = data.get("supplier_performance", [])

        return role_data

    def answer(self, query: str, role: str, data: dict) -> dict:
        """Answer a single query for a given role."""
        self._print_section(f"Query from [{role.upper()}]: {query}")

        role_context   = self._build_role_context(role, data)
        role_prompt    = self.ROLE_CONTEXT.get(role, "You are helping a supply chain professional.")

        system_prompt  = f"""{role_prompt}
                Answer the user's question using ONLY the data provided — nothing outside their authorization.
                Be specific, actionable, and concise.
                Respond ONLY with valid JSON."""

                        user_prompt = f"""
                Answer this supply chain query for a {role.replace('_', ' ').title()}.
                Return JSON with:
                {{
                "direct_answer": "clear, direct answer to the question",
                "key_facts": [3-5 specific facts from the data supporting the answer],
                "recommended_actions": [1-3 specific actions the user should take],
                "urgency": "Immediate/This Week/Monitor",
                "data_scope_note": "brief note on what data was used to answer"
                }}

                User Query: {query}

                Authorized Data for this role:
                {json.dumps(role_context, indent=2, default=str)[:3000]}
                """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            answer = json.loads(response)
        except Exception:
            answer = {"direct_answer": "Unable to process query at this time.", "key_facts": []}

        self._print_output(f"Answer for {role}", answer)
        return {"role": role, "query": query, "answer": answer}

    def run(self, input_data: dict) -> dict:
        self._print_header()

        queries = input_data.get("queries", [
            {"role": "logistics_planner",   "query": "Which shipments are at risk of missing their delivery window this week?"},
            {"role": "finance_manager",     "query": "What is our total invoice discrepancy exposure this month?"},
            {"role": "executive",           "query": "What is the single biggest supply chain risk I need to know about right now?"},
            {"role": "procurement_manager", "query": "Which suppliers have open performance issues I should address?"},
        ])

        responses = []
        for q in queries:
            result = self.answer(q["query"], q["role"], input_data)
            responses.append(result)

        print(f"\n{Fore.GREEN}  ✓ Agent 11 Complete: {len(responses)} role-scoped queries answered{Style.RESET_ALL}")

        return {
            "copilot_responses": responses,
            "queries_answered":  len(responses),
            "agent":             self.agent_name,
            "processed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# Agent 12: Proactive Insight Agent


class ProactiveInsightAgent(BaseAgent):
    """
    INPUT : All agent outputs — risk flags, exceptions, performance alerts,
            supply signals, KPIs, decision windows
    OUTPUT: Proactive, time-sensitive narrative alerts pushed to the right person
    """

    def __init__(self):
        super().__init__("Proactive Insight Agent", "Zone 3 — Reporting")

    def run(self, input_data: dict) -> dict:
        self._print_header()

        risk_flags       = input_data.get("risk_flags", [])
        exceptions       = input_data.get("exceptions", [])
        carrier_alerts   = input_data.get("performance_analysis", {}).get("carrier_alerts", [])
        supply_analysis  = input_data.get("supply_analysis", {})
        kpi_values       = input_data.get("kpi_values", {})
        contingencies    = input_data.get("contingencies", [])
        anomalies        = input_data.get("anomalies", [])

        # ── Rule-based trigger detection
        triggers = []

        # Trigger 1: Stockout with contingency window
        for cont in contingencies:
            triggers.append({
                "trigger_type":  "Stockout Contingency Window",
                "target_role":   "logistics_planner",
                "urgency":       "Immediate",
                "sku":           cont.get("sku"),
                "warehouse":     cont.get("at_risk_warehouse"),
                "contingency_wh":cont.get("contingency_from"),
                "available_qty": cont.get("available_qty"),
                "shortfall":     cont.get("shortfall"),
            })

        # Trigger 2: Invoice discrepancies with aging
        invoice_exceptions = [e for e in exceptions if e.get("type") == "Invoice-GRN Quantity Mismatch"]
        if invoice_exceptions:
            total_inv_exposure = sum(e.get("financial_impact_usd", 0) for e in invoice_exceptions)
            triggers.append({
                "trigger_type":  "Invoice Discrepancy Pattern",
                "target_role":   "procurement_manager",
                "urgency":       "This Week",
                "count":         len(invoice_exceptions),
                "total_exposure":total_inv_exposure,
                "supplier_ids":  list(set(e.get("supplier_id") for e in invoice_exceptions)),
            })

        # Trigger 3: Carrier SLA trajectory breach
        for alert in carrier_alerts:
            if alert.get("severity") == "High":
                triggers.append({
                    "trigger_type":  "Carrier SLA Trajectory",
                    "target_role":   "logistics_planner",
                    "urgency":       "This Week",
                    "carrier":       alert.get("carrier_name"),
                    "alert_detail":  alert.get("alert_type"),
                    "action":        alert.get("recommended_action"),
                })

        # Trigger 4: Critical inventory depletion
        critical_exc = [e for e in exceptions if e.get("severity") == "Critical"]
        if critical_exc:
            triggers.append({
                "trigger_type":  "Critical Inventory Depletion",
                "target_role":   "executive",
                "urgency":       "Immediate",
                "count":         len(critical_exc),
                "details":       critical_exc[:2],
            })

        # Trigger 5: KPI breach
        if kpi_values.get("on_time_delivery_rate_pct", 100) < 90:
            triggers.append({
                "trigger_type":  "OTD KPI Below Threshold",
                "target_role":   "executive",
                "urgency":       "This Week",
                "current_otd":   kpi_values.get("on_time_delivery_rate_pct"),
                "threshold":     90,
            })

        # ── LLM generates insight messages for each trigger
        system_prompt = """You are a proactive supply chain insight agent.
                Generate clear, urgent, actionable insight messages for supply chain professionals.
                Each message should tell them what happened, why it matters, and what to do.
                Respond ONLY with valid JSON."""

                        user_prompt = f"""
                Generate proactive insight messages for these supply chain triggers.
                Return JSON with:
                {{
                "insights": [
                    {{
                    "target_role": "<role>",
                    "subject": "<brief subject line>",
                    "message": "2-3 sentence insight message with specific data points",
                    "urgency": "Immediate/This Week/Monitor",
                    "action_required": "<specific action with deadline if applicable>",
                    "financial_or_operational_impact": "<quantified impact>"
                    }}
                ],
                "total_proactive_alerts": <number>,
                "highest_urgency_count": <number of Immediate alerts>
                }}

                Active Triggers ({len(triggers)}):
                {json.dumps(triggers, indent=2, default=str)}

                Supply Analysis Context:
                {json.dumps(supply_analysis.get('stockout_risks', [])[:3], indent=2, default=str)}
                """
        response = self.call_llm(system_prompt, user_prompt)
        try:
            insights = json.loads(response)
        except Exception:
            insights = {"insights": [], "total_proactive_alerts": len(triggers)}

        # ── Print insights clearly
        print(f"\n{Fore.YELLOW}  ── PROACTIVE INSIGHTS GENERATED ──{Style.RESET_ALL}")
        for insight in insights.get("insights", []):
            urgency_color = Fore.RED if insight.get("urgency") == "Immediate" else \
                           Fore.YELLOW if insight.get("urgency") == "This Week" else Fore.CYAN
            print(f"\n  {urgency_color}[{insight.get('urgency', 'Monitor').upper()}] "
                  f"→ {insight.get('target_role', '').upper()}{Style.RESET_ALL}")
            print(f"  Subject : {insight.get('subject', '')}")
            print(f"  Message : {insight.get('message', '')}")
            print(f"  Action  : {insight.get('action_required', '')}")
            print(f"  Impact  : {insight.get('financial_or_operational_impact', '')}")

        print(f"\n{Fore.GREEN}  ✓ Agent 12 Complete: {len(insights.get('insights', []))} proactive insights generated{Style.RESET_ALL}")

        return {
            "triggers":             triggers,
            "proactive_insights":   insights,
            "total_alerts":         len(insights.get("insights", [])),
            "agent":                self.agent_name,
            "processed_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
