## AI-Accelerated Supply Chain

## The Problem

Modern supply chains are drowning in data and still making decisions blind.
A single purchase order touches five systems:
1. ERP creates it
2. TMS tracks the shipment 
3. Finance processes the invoice
4. The warehouse logs the receipt 
5. The retailer scores the delivery. 

Each system captures the same event differently, at different times, in different formats. Nobody reconciles them automatically. Nobody is watching for the moment things go wrong.

The result:

1. Invoice overbillings discovered at month end, days after payment has already run
2. Stockouts identified after shelves are already empty
3. Carrier SLA breaches measured retrospectively, never predicted
4. Dashboards that require a human to open them insight is pull-based, never proactive
5. Conflicting data from TMS and ERP on the same shipment, with no authoritative truth

A single undetected stockout at a high priority retail account during peak season costs €40,000–70,000 in lost sales. A systematic invoice overbilling pattern left undetected costs tens of thousands per month. These are not edge cases, they are the default state of most supply chain operations.

## Objective of building this project

Supply chain is the backbone of the German economy, manufacturing, automotive, retail, logistics. SAP was built here. The problems above are not hypothetical, they are the daily reality for operations teams across the country and across Europe.

I wanted to build something that demonstrated two things together:

Agentic AI applied to a real business domain: Not a chatbot, not a wrapper around an API, but a system of autonomous agents that each have a defined role, reason over real data, and produce structured outputs that feed the next stage of a pipeline

Enterprise data architecture: The Medallion pattern (Bronze -> Silver -> Gold), data quality gating, conflict reconciliation, role-based access.The patterns that actually matter in production data systems

**This project is my answer to the question: What does AI engineering look like when it solves a problem worth solving?**

## What This System Does

Twelve AI agents, each powered by Azure OpenAI GPT-5 govern every stage of the supply chain data flow. 
Agents do not simply move data. 
They validate, reconcile, enrich, analyse, and proactively surface decisions to the right human at the right time.

## Architecture — 4-Zone Medallion

```
SOURCE SYSTEMS
ERP · TMS · Finance · Retailer Scorecards · Risk Feed
        │
        ▼
┌─────────────────────────────┐
│  ZONE 0 — PRE-INGESTION     │
│  Agent 1: Document Agent    │  Extracts PDFs and Excel into structured records
│  Agent 2: Data Quality Gate │  Scores and quarantines bad data at the boundary
│  Agent 3: Source Anomaly    │  Monitors feed behaviour; tags risk-affected records
└─────────────┬───────────────┘
              │ Validated, DQ-scored records
              ▼
┌─────────────────────────────┐
│  ZONE 1 — PROCESSING        │
│  Agent 4: Reconciliation    │  Resolves TMS vs ERP conflicts → Silver layer
│  Agent 5: Semantic Enrich.  │  Translates codes and IDs into business language
│  Agent 6: Data Promotion    │  Gates Gold access — only clean records pass
└─────────────┬───────────────┘
              │ Trusted, enriched Gold records
              ▼
┌─────────────────────────────┐
│  ZONE 2 — GOLD INTELLIGENCE │
│  Agent 7: Demand & Supply   │  Stockout risk signals + contingency options
│  Agent 8: Supplier/Carrier  │  SLA scoring and declining trend detection
│  Agent 9: Anomaly/Exception │  Invoice mismatches, damage, depletion — with £ impact
└─────────────┬───────────────┘
              │ Intelligence signals + exceptions
              ▼
┌─────────────────────────────┐
│  ZONE 3 — REPORTING         │
│  Agent 10: KPI Reports      │  10 KPIs computed and narrated automatically
│  Agent 11: Role Copilot     │  Natural language Q&A, scoped by persona
│  Agent 12: Proactive Alerts │  Pushes insights to the right human before they ask
└─────────────┬───────────────┘
              │
    Logistics Planner · Finance Manager · Executive · Procurement
```

**Core design principle:** Every agent runs deterministic rule-based logic first scoring, thresholds, conflict detection then calls GPT-5 to add reasoning, narrative, and judgment on top of structured facts. **The LLM never does arithmetic. Rules do. The LLM makes the rules think.**

## The 12 Agents

|   | Agent | Zone | What It Does |
|---|-------|------|-------------|
| 1 | Document Agent | Pre-Ingestion | Extracts tables from PDFs and Excel files into structured records |
| 2 | Data Quality Gate | Pre-Ingestion | Scores every record 0–100; quarantines anything below threshold |
| 3 | Source Anomaly Detection | Pre-Ingestion | Detects feed volume spikes and silent failures; tags risk-exposed shipments |
| 4 | Reconciliation Agent | Processing | Resolves TMS vs ERP conflicts; produces one authoritative truth per record |
| 5 | Semantic Enrichment | Processing | Replaces IDs and codes with full business language using master data |
| 6 | Data Promotion | Processing | Quality gate for the Gold layer scores and promotes or holds each record |
| 7 | Demand & Supply Signal | Gold Intelligence | Computes supply balance per SKU per warehouse; finds contingency options with deadlines |
| 8 | Supplier & Carrier Performance | Gold Intelligence | Scores SLA compliance; detects declining trends before breaches occur |
| 9 | Anomaly & Exception | Gold Intelligence | Detects invoice mismatches, lane delay clusters, damage, depletion with financial impact |
| 10 | KPI Report Generator | Reporting | Computes 10 KPIs with status indicators; GPT-5 authors the full narrative automatically |
| 11 | Role-Based Copilot | Reporting | Answers the same question differently for a Finance Manager vs a Logistics Planner |
| 12 | Proactive Insight Agent | Reporting | Pushes time sensitive alerts with deadlines to the right persona without being asked |

## One Full Pipeline Run — Results

| Zone | What Was Found | Result |
|------|---------------|--------|
| Zone 0 | Retailer scorecards extracted from PDFs | 3 documents → structured records, 97% quality |
| Zone 0 | Financial exposure from active risk events | **$312,000** across 2 risk-tagged shipments |
| Zone 1 | TMS/ERP status conflicts resolved | 2 shipments reconciled (SHP-5002, SHP-5006) |
| Zone 1 | Records promoted to Gold | 8 of 8 — 100% promotion rate |
| Zone 2 | Critical stockout deadlines identified | 2 SKUs with action window closing same day |
| Zone 2 | Invoice overbilling caught | **$7,200** (INV-3002, NordicParts GmbH) |
| Zone 2 | Total financial exposure quantified | **$204,950** across 10 exceptions |
| Zone 3 | KPIs computed and scored | 10 with Green / Amber / Red status |
| Zone 3 | Proactive alerts pushed | 4 (2 Immediate, 2 This Week) |


## Synthetic Data & Deliberate Anomalies

The data generator produces 13 datasets with embedded anomalies — because a supply chain without exceptions is not a supply chain.

| Anomaly | What It Triggers |
|---------|-----------------|
| `PO-10009` has null `destination_wh` | Agent 2 DQ quarantine |
| `SHP-5002`, `SHP-5006`: TMS status ≠ ERP status | Agent 4 reconciliation conflict |
| `INV-3002`: billed qty 400 vs received qty 340 | Agent 9 overbilling exception ($7,200) |
| `WH-NYC-001 PRD-1001` at 24% of reorder point | Agent 7 critical stockout + Agent 12 immediate alert |
| `CAR-001` score declining across 9 periods | Agent 8 declining trend alert |
| 2 active high-severity risk events | Agent 3 risk-tagging of affected shipments |

Every anomaly is traceable from Zone 0 detection through to Zone 3 alert delivery.


## Stack

| Component | Technology |
|-----------|------------|
| LLM | Azure OpenAI GPT-5 — `response_format: json_object`, `temperature: 0.2` |
| Data layer | Pandas DataFrames simulating Bronze / Silver / Gold Delta Lake |
| Agent base | Python `BaseAgent` — shared client, logging, fail-safe LLM calls |
| Orchestration | Sequential zone execution with shared `pipeline_state` dict |
| Synthetic data | `generator.py` — 13 datasets, `seed=42` |


