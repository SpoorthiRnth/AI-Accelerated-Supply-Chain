import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

def generate_supplier_master() -> pd.DataFrame:
    return pd.DataFrame([
        {"supplier_id": "SUP-001", "supplier_name": "GlobalTex Industries",   "location": "Shanghai, China",      "category": "Apparel",      "sla_fulfillment_pct": 96, "payment_terms_days": 30, "risk_rating": "Low"},
        {"supplier_id": "SUP-002", "supplier_name": "NordicParts GmbH",        "location": "Hamburg, Germany",     "category": "Electronics",  "sla_fulfillment_pct": 98, "payment_terms_days": 45, "risk_rating": "Low"},
        {"supplier_id": "SUP-003", "supplier_name": "SunRise Packaging Co.",   "location": "Mumbai, India",        "category": "Packaging",    "sla_fulfillment_pct": 88, "payment_terms_days": 30, "risk_rating": "Medium"},
        {"supplier_id": "SUP-004", "supplier_name": "Apex Metals Ltd.",        "location": "Detroit, USA",         "category": "Raw Material", "sla_fulfillment_pct": 94, "payment_terms_days": 60, "risk_rating": "Low"},
        {"supplier_id": "SUP-005", "supplier_name": "FastComp Electronics",    "location": "Shenzhen, China",      "category": "Electronics",  "sla_fulfillment_pct": 91, "payment_terms_days": 30, "risk_rating": "Medium"},
        {"supplier_id": "SUP-006", "supplier_name": "MedTex Fabrics",          "location": "Istanbul, Turkey",     "category": "Apparel",      "sla_fulfillment_pct": 85, "payment_terms_days": 45, "risk_rating": "Medium"},
    ])

def generate_carrier_master() -> pd.DataFrame:
    return pd.DataFrame([
        {"carrier_id": "CAR-001", "carrier_name": "FastFreight Logistics",  "mode": "Road",  "lanes": "Chicago-Northeast",    "sla_ontime_pct": 98, "current_score": 91, "avg_transit_days": 2},
        {"carrier_id": "CAR-002", "carrier_name": "BlueOcean Shipping",     "mode": "Sea",   "lanes": "Asia-US West Coast",   "sla_ontime_pct": 95, "current_score": 94, "avg_transit_days": 21},
        {"carrier_id": "CAR-003", "carrier_name": "SkyExpress Air Cargo",   "mode": "Air",   "lanes": "International",        "sla_ontime_pct": 97, "current_score": 96, "avg_transit_days": 1},
        {"carrier_id": "CAR-004", "carrier_name": "MidWest Truckers Inc.",  "mode": "Road",  "lanes": "Dallas-Southeast",     "sla_ontime_pct": 92, "current_score": 89, "avg_transit_days": 3},
        {"carrier_id": "CAR-005", "carrier_name": "EuroLink Freight",       "mode": "Road",  "lanes": "Europe Cross-Border",  "sla_ontime_pct": 96, "current_score": 95, "avg_transit_days": 4},
    ])

def generate_product_master() -> pd.DataFrame:
    return pd.DataFrame([
        {"sku": "PRD-1001", "product_name": "Winter Jacket - Premium",   "category": "Apparel",     "unit_cost": 45.00,  "lead_time_days": 14, "reorder_point": 500,  "substitute_sku": "PRD-1002"},
        {"sku": "PRD-1002", "product_name": "Winter Jacket - Standard",  "category": "Apparel",     "unit_cost": 28.00,  "lead_time_days": 10, "reorder_point": 800,  "substitute_sku": None},
        {"sku": "PRD-2001", "product_name": "Circuit Board X200",        "category": "Electronics", "unit_cost": 120.00, "lead_time_days": 21, "reorder_point": 200,  "substitute_sku": "PRD-2002"},
        {"sku": "PRD-2002", "product_name": "Circuit Board X150",        "category": "Electronics", "unit_cost": 95.00,  "lead_time_days": 18, "reorder_point": 300,  "substitute_sku": None},
        {"sku": "PRD-3001", "product_name": "Packaging Box - Large",     "category": "Packaging",   "unit_cost": 2.50,   "lead_time_days": 7,  "reorder_point": 5000, "substitute_sku": None},
        {"sku": "PRD-4001", "product_name": "Steel Rod Grade A",         "category": "Raw Material","unit_cost": 8.75,   "lead_time_days": 30, "reorder_point": 2000, "substitute_sku": "PRD-4002"},
        {"sku": "PRD-4002", "product_name": "Steel Rod Grade B",         "category": "Raw Material","unit_cost": 6.50,   "lead_time_days": 25, "reorder_point": 2500, "substitute_sku": None},
    ])

def generate_warehouse_master() -> pd.DataFrame:
    return pd.DataFrame([
        {"warehouse_id": "WH-CHI-001", "warehouse_name": "Chicago Midwest DC",      "city": "Chicago",    "state": "IL", "capacity_units": 50000, "current_utilization_pct": 84},
        {"warehouse_id": "WH-DAL-001", "warehouse_name": "Dallas South DC",         "city": "Dallas",     "state": "TX", "capacity_units": 40000, "current_utilization_pct": 61},
        {"warehouse_id": "WH-NYC-001", "warehouse_name": "New York East DC",        "city": "Newark",     "state": "NJ", "capacity_units": 35000, "current_utilization_pct": 79},
        {"warehouse_id": "WH-LAX-001", "warehouse_name": "Los Angeles West DC",     "city": "Los Angeles","state": "CA", "capacity_units": 45000, "current_utilization_pct": 72},
        {"warehouse_id": "WH-ATL-001", "warehouse_name": "Atlanta Southeast DC",    "city": "Atlanta",    "state": "GA", "capacity_units": 38000, "current_utilization_pct": 55},
    ])

def generate_retailer_master() -> pd.DataFrame:
    return pd.DataFrame([
        {"retailer_id": "RET-001", "retailer_name": "NorthStar Retail Group",  "city": "New York",    "priority": "High",   "annual_volume_usd": 12000000},
        {"retailer_id": "RET-002", "retailer_name": "SunCoast Stores",         "city": "Miami",       "priority": "Medium", "annual_volume_usd": 5500000},
        {"retailer_id": "RET-003", "retailer_name": "MidWest Mart",            "city": "Chicago",     "priority": "High",   "annual_volume_usd": 9800000},
        {"retailer_id": "RET-004", "retailer_name": "Pacific Retail Co.",      "city": "Los Angeles", "priority": "Medium", "annual_volume_usd": 7200000},
        {"retailer_id": "RET-005", "retailer_name": "Dixie Distribution",      "city": "Atlanta",     "priority": "Low",    "annual_volume_usd": 3100000},
    ])

#transactional data

def generate_purchase_orders() -> pd.DataFrame:
    base_date = datetime(2025, 10, 1)
    orders = []
    po_configs = [
        ("PO-10001", "SUP-001", "PRD-1001", "WH-CHI-001", 1200, 45.00, 0),
        ("PO-10002", "SUP-002", "PRD-2001", "WH-NYC-001", 400,  120.00, 2),
        ("PO-10003", "SUP-003", "PRD-3001", "WH-DAL-001", 8000, 2.50,  5),
        ("PO-10004", "SUP-004", "PRD-4001", "WH-LAX-001", 3000, 8.75,  3),
        ("PO-10005", "SUP-005", "PRD-2002", "WH-ATL-001", 600,  95.00, 7),
        ("PO-10006", "SUP-006", "PRD-1002", "WH-CHI-001", 2000, 28.00, 1),
        ("PO-10007", "SUP-001", "PRD-1001", "WH-DAL-001", 800,  45.00, 10),
        ("PO-10008", "SUP-002", "PRD-2001", "WH-LAX-001", 350,  120.00, 4),
        # Intentionally missing warehouse_id to trigger DQ agent
        ("PO-10009", "SUP-003", "PRD-3001", None,          5000, 2.50,  6),
        ("PO-10010", "SUP-004", "PRD-4002", "WH-NYC-001",  2500, 6.50,  8),
    ]
    statuses = ["Open", "Confirmed", "In Transit", "Partially Received", "Closed"]
    for po_id, sup_id, sku, wh_id, qty, price, days_offset in po_configs:
        order_date = base_date + timedelta(days=days_offset)
        promised_date = order_date + timedelta(days=random.randint(10, 21))
        status = random.choice(statuses)
        orders.append({
            "po_id":              po_id,
            "supplier_id":        sup_id,
            "sku":                sku,
            "destination_wh":     wh_id,
            "ordered_qty":        qty,
            "unit_price":         price,
            "total_value":        round(qty * price, 2),
            "order_date":         order_date.strftime("%Y-%m-%d"),
            "promised_date":      promised_date.strftime("%Y-%m-%d"),
            "status":             status,
            "currency":           "USD",
        })
    return pd.DataFrame(orders)


def generate_shipments() -> pd.DataFrame:
    base_date = datetime(2025, 10, 5)
    shipments = []
    configs = [
        ("SHP-5001", "PO-10001", "CAR-001", "WH-CHI-001", "RET-001", 1200, 0,  "Delivered",     False),
        ("SHP-5002", "PO-10002", "CAR-003", "WH-NYC-001", "RET-001", 400,  2,  "In Transit",    True),   # port risk flag
        ("SHP-5003", "PO-10003", "CAR-004", "WH-DAL-001", "RET-005", 8000, 3,  "Delayed",       False),
        ("SHP-5004", "PO-10004", "CAR-002", "WH-LAX-001", "RET-004", 3000, 1,  "In Transit",    True),   # port risk flag
        ("SHP-5005", "PO-10005", "CAR-004", "WH-ATL-001", "RET-005", 600,  5,  "Delayed",       False),
        ("SHP-5006", "PO-10006", "CAR-001", "WH-CHI-001", "RET-003", 2000, 0,  "In Transit",    False),
        ("SHP-5007", "PO-10007", "CAR-004", "WH-DAL-001", "RET-002", 800,  7,  "Delayed",       False),
        ("SHP-5008", "PO-10008", "CAR-003", "WH-LAX-001", "RET-004", 350,  4,  "Delivered",     False),
    ]
    for shp_id, po_id, car_id, origin_wh, dest_ret, qty, days_offset, status, port_risk in configs:
        dispatch_date = base_date + timedelta(days=days_offset)
        eta = dispatch_date + timedelta(days=random.randint(1, 7))
        actual_delivery = (eta + timedelta(days=random.randint(-1, 3))).strftime("%Y-%m-%d") if status == "Delivered" else None
        freight_cost = round(qty * random.uniform(0.8, 2.5), 2)
        shipments.append({
            "shipment_id":       shp_id,
            "po_id":             po_id,
            "carrier_id":        car_id,
            "origin_warehouse":  origin_wh,
            "destination_retailer": dest_ret,
            "shipped_qty":       qty,
            "dispatch_date":     dispatch_date.strftime("%Y-%m-%d"),
            "eta":               eta.strftime("%Y-%m-%d"),
            "actual_delivery":   actual_delivery,
            "status":            status,
            "freight_cost_usd":  freight_cost,
            "port_risk_flag":    port_risk,
            # TMS and ERP intentionally conflict on 2 records for Reconciliation Agent
            "tms_status":        status,
            "erp_status":        "Pending Dispatch" if shp_id in ["SHP-5002", "SHP-5006"] else status,
        })
    return pd.DataFrame(shipments)


def generate_inventory() -> pd.DataFrame:
    records = []
    warehouse_ids = ["WH-CHI-001", "WH-DAL-001", "WH-NYC-001", "WH-LAX-001", "WH-ATL-001"]
    skus = ["PRD-1001", "PRD-1002", "PRD-2001", "PRD-2002", "PRD-3001", "PRD-4001", "PRD-4002"]
    for wh in warehouse_ids:
        for sku in skus:
            qty = random.randint(100, 3000)
            reorder = {"PRD-1001": 500, "PRD-1002": 800, "PRD-2001": 200, "PRD-2002": 300,
                       "PRD-3001": 5000, "PRD-4001": 2000, "PRD-4002": 2500}.get(sku, 500)
            records.append({
                "warehouse_id":     wh,
                "sku":              sku,
                "qty_on_hand":      qty,
                "reorder_point":    reorder,
                "below_reorder":    qty < reorder,
                "last_updated":     (datetime(2025, 10, 14) - timedelta(hours=random.randint(0, 12))).strftime("%Y-%m-%d %H:%M"),
            })
    # Force NYC PRD-1001 to be critically low to trigger Demand-Supply agent
    df = pd.DataFrame(records)
    df.loc[(df.warehouse_id == "WH-NYC-001") & (df.sku == "PRD-1001"), "qty_on_hand"] = 120
    df.loc[(df.warehouse_id == "WH-NYC-001") & (df.sku == "PRD-1001"), "below_reorder"] = True
    return df


def generate_goods_receipts() -> pd.DataFrame:
    return pd.DataFrame([
        {"grn_id": "GRN-001", "po_id": "PO-10001", "shipment_id": "SHP-5001", "received_qty": 1200, "damaged_qty": 0,  "receipt_date": "2025-10-10", "warehouse_id": "WH-CHI-001", "condition": "Good"},
        {"grn_id": "GRN-002", "po_id": "PO-10008", "shipment_id": "SHP-5008", "received_qty": 340,  "damaged_qty": 10, "receipt_date": "2025-10-12", "warehouse_id": "WH-LAX-001", "condition": "Partial Damage"},
        {"grn_id": "GRN-003", "po_id": "PO-10005", "shipment_id": "SHP-5005", "received_qty": 580,  "damaged_qty": 0,  "receipt_date": "2025-10-13", "warehouse_id": "WH-ATL-001", "condition": "Good"},
    ])


def generate_supplier_invoices() -> pd.DataFrame:
    return pd.DataFrame([
        # Correct invoice
        {"invoice_id": "INV-3001", "po_id": "PO-10001", "supplier_id": "SUP-001", "billed_qty": 1200, "unit_price": 45.00, "total_billed": 54000.00, "invoice_date": "2025-10-11", "due_date": "2025-11-10", "status": "Pending"},
        # Over-billed invoice — triggers Anomaly Agent
        {"invoice_id": "INV-3002", "po_id": "PO-10008", "supplier_id": "SUP-002", "billed_qty": 400,  "unit_price": 120.00,"total_billed": 48000.00, "invoice_date": "2025-10-13", "due_date": "2025-11-27", "status": "Pending"},
        {"invoice_id": "INV-3003", "po_id": "PO-10003", "supplier_id": "SUP-003", "billed_qty": 8000, "unit_price": 2.50,  "total_billed": 20000.00, "invoice_date": "2025-10-12", "due_date": "2025-11-11", "status": "Pending"},
        {"invoice_id": "INV-3004", "po_id": "PO-10005", "supplier_id": "SUP-005", "billed_qty": 620,  "unit_price": 95.00, "total_billed": 58900.00, "invoice_date": "2025-10-14", "due_date": "2025-11-13", "status": "Pending"},
        {"invoice_id": "INV-3005", "po_id": "PO-10006", "supplier_id": "SUP-006", "billed_qty": 2000, "unit_price": 28.00, "total_billed": 56000.00, "invoice_date": "2025-10-11", "due_date": "2025-11-25", "status": "Pending"},
    ])


def generate_retailer_scorecards() -> list:
    """Returns list of dicts simulating parsed PDF scorecard data."""
    return [
        {
            "document_type":     "Retailer Scorecard",
            "retailer_id":       "RET-001",
            "retailer_name":     "NorthStar Retail Group",
            "period":            "September 2025",
            "carrier_id":        "CAR-001",
            "carrier_name":      "FastFreight Logistics",
            "ontime_delivery_pct": 87.4,
            "fill_rate_pct":     94.2,
            "damage_rate_pct":   1.8,
            "overall_score":     3.2,
            "comments":          "Delays on Northeast corridor impacting shelf availability. Requesting SLA review.",
            "extraction_confidence": 0.96,
        },
        {
            "document_type":     "Retailer Scorecard",
            "retailer_id":       "RET-003",
            "retailer_name":     "MidWest Mart",
            "period":            "September 2025",
            "carrier_id":        "CAR-004",
            "carrier_name":      "MidWest Truckers Inc.",
            "ontime_delivery_pct": 91.0,
            "fill_rate_pct":     96.5,
            "damage_rate_pct":   0.9,
            "overall_score":     4.1,
            "comments":          "Generally satisfied. Minor delays on 2 shipments.",
            "extraction_confidence": 0.98,
        },
        {
            "document_type":     "Retailer Scorecard",
            "retailer_id":       "RET-004",
            "retailer_name":     "Pacific Retail Co.",
            "period":            "September 2025",
            "carrier_id":        "CAR-002",
            "carrier_name":      "BlueOcean Shipping",
            "ontime_delivery_pct": 93.8,
            "fill_rate_pct":     97.1,
            "damage_rate_pct":   0.5,
            "overall_score":     4.4,
            "comments":          "Good performance. Expect higher volumes in Q4.",
            "extraction_confidence": 0.99,
        },
    ]


def generate_risk_feed() -> pd.DataFrame:
    return pd.DataFrame([
        {"event_id": "EVT-001", "event_type": "Port Disruption",  "severity": "High",   "location": "Port of Singapore",     "affected_routes": "Asia-US West Coast", "estimated_delay_days": 6, "start_date": "2025-10-13", "active": True},
        {"event_id": "EVT-002", "event_type": "Weather",          "severity": "Medium", "location": "Chicago, IL",           "affected_routes": "Chicago-Northeast",  "estimated_delay_days": 2, "start_date": "2025-10-12", "active": True},
        {"event_id": "EVT-003", "event_type": "Strike",           "severity": "Low",    "location": "Hamburg, Germany",      "affected_routes": "Europe Cross-Border","estimated_delay_days": 3, "start_date": "2025-10-10", "active": False},
        {"event_id": "EVT-004", "event_type": "Geopolitical",     "severity": "Medium", "location": "Shenzhen, China",       "affected_routes": "Asia-US West Coast", "estimated_delay_days": 4, "start_date": "2025-10-14", "active": True},
    ])


def generate_carrier_performance_history() -> pd.DataFrame:
    """90-day rolling performance history for trend analysis."""
    records = []
    carriers = [
        ("CAR-001", "FastFreight Logistics",  [97, 96, 96, 95, 94, 93, 92, 91, 91]),
        ("CAR-002", "BlueOcean Shipping",      [95, 95, 94, 94, 94, 93, 94, 94, 94]),
        ("CAR-003", "SkyExpress Air Cargo",    [98, 97, 97, 97, 96, 96, 96, 96, 96]),
        ("CAR-004", "MidWest Truckers Inc.",   [93, 92, 91, 90, 90, 89, 89, 88, 89]),
        ("CAR-005", "EuroLink Freight",        [96, 96, 95, 96, 95, 95, 95, 95, 95]),
    ]
    for car_id, car_name, scores in carriers:
        for i, score in enumerate(scores):
            month_date = (datetime(2025, 10, 1) - timedelta(days=(8-i)*10)).strftime("%Y-%m-%d")
            records.append({"carrier_id": car_id, "carrier_name": car_name,
                            "period_date": month_date, "ontime_pct": score})
    return pd.DataFrame(records)


# load all the data


def load_all_data() -> dict:
    return {
        "supplier_master":              generate_supplier_master(),
        "carrier_master":               generate_carrier_master(),
        "product_master":               generate_product_master(),
        "warehouse_master":             generate_warehouse_master(),
        "retailer_master":              generate_retailer_master(),
        "purchase_orders":              generate_purchase_orders(),
        "shipments":                    generate_shipments(),
        "inventory":                    generate_inventory(),
        "goods_receipts":               generate_goods_receipts(),
        "supplier_invoices":            generate_supplier_invoices(),
        "retailer_scorecards":          generate_retailer_scorecards(),
        "risk_feed":                    generate_risk_feed(),
        "carrier_performance_history":  generate_carrier_performance_history(),
    }


if __name__ == "__main__":
    data = load_all_data()
    for name, dataset in data.items():
        if isinstance(dataset, pd.DataFrame):
            print(f"\n{name}: {len(dataset)} records")
            print(dataset.head(2).to_string())
        elif isinstance(dataset, list):
            print(f"\n{name}: {len(dataset)} documents")
