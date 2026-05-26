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