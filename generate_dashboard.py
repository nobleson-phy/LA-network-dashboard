"""
Interactive Network Graph Dashboard Generator

Reads gaze activity CSV data, computes network graph data for every Task_ID x modality combination,
and generates a standalone HTML file with embedded Cytoscape.js.

Usage:
    python generate_dashboard.py <csv_file> [options]

Example:
    python generate_dashboard.py 68fd29b9babf3bbc48f57506_Integrated_Gaze_Activity.csv
"""

import pandas as pd
import numpy as np
import networkx as nx
import re
import os
import sys
import json
import math
import argparse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate an interactive Cytoscape.js network graph dashboard from gaze activity CSV."
    )
    parser.add_argument("csv_file", help="Path to the CSV file to process")
    parser.add_argument("--user-filter", "-u", default="6",
                        help='Filter users by ID prefix (default: "6")')
    parser.add_argument("--min-duration", "-d", type=float, default=0,
                        help="Minimum duration threshold for nodes (seconds)")
    parser.add_argument("--min-frequency", "-f", type=int, default=2,
                        help="Minimum frequency threshold for edges")
    parser.add_argument("--edge-weight", "-e", choices=["average", "total"], default="total",
                        help="Edge weight calculation method (default: total)")
    parser.add_argument("--edge-representation", "-r", choices=["time", "frequency"], default="time",
                        help="Edge representation: time or frequency (default: time)")
    parser.add_argument("--output", "-o", default="dashboard.html",
                        help="Output HTML file path (default: dashboard.html)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preprocessing (from timeseries.network6.3.py lines 75-194)
# ---------------------------------------------------------------------------

def preprocess_data(csv_file, user_filter="6"):
    print(f"\n{'='*80}")
    print(f"PREPROCESSING RAW CSV: {csv_file}")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(csv_file, skiprows=1, low_memory=False)
        print(f"Successfully loaded CSV file with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    column_names = [
        "User_ID", "Timestamp", "X", "Y", "Panel_Title", "Task_ID",
        "Activity_ID", "Screen_ID", "Activity", "Task_Type", "Element_ID",
        "Element_Type", "Verb", "CourseID", "Duration", "Attempted",
    ]
    df.columns = column_names

    if "User_ID" not in df.columns:
        print("Error: User_ID column not found!")
        sys.exit(1)

    # Keep all users matching the prefix filter
    df["User_ID"] = df["User_ID"].astype(str)
    initial_rows = len(df)
    df = df[df["User_ID"].str.startswith(user_filter)]
    print(f"Filtered rows (prefix '{user_filter}'): {initial_rows} -> {len(df)}")

    user_ids = sorted(df["User_ID"].unique())
    print(f"Users found: {len(user_ids)}")
    for uid in user_ids:
        print(f"  {uid}: {len(df[df['User_ID'] == uid])} rows")

    # Modality
    df["modality"] = np.where(df["X"].isna() | (df["X"] == ""), "mclick", "eTrack")

    # Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S:%f", errors="coerce")
    df = df.sort_values("Timestamp")

    # Clean Panel_Title
    def clean_panel_title(title):
        if pd.isna(title) or title == "":
            return ""
        cleaned = re.sub(r"[^\x00-\x7F]+", "", str(title))
        cleaned = re.sub(r"[^\w\s\-()/.,]+", "", cleaned)
        return cleaned.strip()

    df["Panel_Title"] = df["Panel_Title"].apply(clean_panel_title)

    # Fill 'outside' for eTrack rows with empty Panel_Title
    df["X_numeric"] = pd.to_numeric(df["X"], errors="coerce")
    fill_condition = df["X_numeric"].notna() & (df["X_numeric"] != 0) & (df["Panel_Title"] == "")
    df.loc[fill_condition, "Panel_Title"] = "outside"

    additional_fill = (
        (df["modality"] == "eTrack")
        & df["X"].notna()
        & (df["X"] != "")
        & (df["Panel_Title"] == "")
    )
    df.loc[additional_fill, "Panel_Title"] = "outside"
    df = df.drop("X_numeric", axis=1)

    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"Preprocessing complete: {len(df)} rows, {len(user_ids)} user(s)")
    return df, user_ids


# ---------------------------------------------------------------------------
# Fixed classification & colours (from timeseries.network6.3.py)
# ---------------------------------------------------------------------------

FIXED_CLASSIFICATION = {
    "Mobile phones breakup details": "picture",
    "Mobile phones for teaching": "picture",
    "Desktop breakup details": "picture",
    "Laptop breakup details": "picture",
    "Laptop / Notebook": "picture",
    "Distribution of schools": "picture",
    "Dropout - secondary": "picture",
    "Desktop": "picture",
    "Projector": "picture",
    "Digital Library": "picture",
    "Pupil-teacher ratio (PTR)": "picture",
    "Dropout - middle": "picture",
    "Dropout - preparatory": "picture",
    "Distribution of teachers": "picture",
    "ICT labs": "picture",
    "Infrastructure": "picture",
    "Distribution of students": "picture",
    "Tablet breakup details": "picture",
    "Tablet": "picture",
    "1st chart in the report": "text",
    "2nd chart in the report": "text",
    "3rd chart in the report": "text",
    "Ordering three charts as evidence": "instruction",
    "outside": "outside",
}

FIXED_COLORS = {
    "picture": "#2ca02c",
    "text": "#C227F5",
    "instruction": "#FFA500",
    "outside": "#cccccc",
    "other": "#7f7f7f",
}

SPECIAL_TEXT_NODES = {
    "1st chart in the report",
    "2nd chart in the report",
    "3rd chart in the report",
}


# ---------------------------------------------------------------------------
# Computation helpers (from timeseries.network6.3.py)
# ---------------------------------------------------------------------------

def calculate_time_spent(task_data):
    task_sorted = task_data.sort_values("Timestamp").reset_index(drop=True)
    time_spent = {}
    for i in range(len(task_sorted) - 1):
        panel = task_sorted.iloc[i]["Panel_Title"]
        if pd.isna(panel) or panel == "":
            continue
        diff = (task_sorted.iloc[i + 1]["Timestamp"] - task_sorted.iloc[i]["Timestamp"]).total_seconds()
        if diff > 0:
            time_spent[panel] = time_spent.get(panel, 0) + diff
    return time_spent


def calculate_edge_weights(task_data, edge_weight_method="total"):
    task_sorted = task_data.sort_values("Timestamp").reset_index(drop=True)
    edge_frequencies = {}
    edge_times = {}
    edge_counts = {}

    for i in range(len(task_sorted) - 1):
        cur = task_sorted.iloc[i]["Panel_Title"]
        nxt = task_sorted.iloc[i + 1]["Panel_Title"]
        if pd.isna(cur) or cur == "" or pd.isna(nxt) or nxt == "":
            continue
        if cur == nxt:
            continue

        key = tuple(sorted([cur, nxt]))
        diff = (task_sorted.iloc[i + 1]["Timestamp"] - task_sorted.iloc[i]["Timestamp"]).total_seconds()

        if key not in edge_times:
            edge_times[key] = 0
            edge_counts[key] = 0
            edge_frequencies[key] = 0

        edge_times[key] += diff
        edge_counts[key] += 1
        edge_frequencies[key] += 1

    edge_time_weights = {}
    for key in edge_times:
        if edge_weight_method == "average":
            edge_time_weights[key] = edge_times[key] / edge_counts[key] if edge_counts[key] else 0
        else:
            edge_time_weights[key] = edge_times[key]

    return edge_frequencies, edge_time_weights


def classify_panel(panel):
    if panel in FIXED_CLASSIFICATION:
        return FIXED_CLASSIFICATION[panel]
    lower = panel.lower()
    if any(k in lower for k in ["chart", "text", "report"]):
        return "text"
    if any(k in lower for k in ["instruction", "order", "evidence"]):
        return "instruction"
    if panel == "outside":
        return "outside"
    return "other"


# ---------------------------------------------------------------------------
# Network metric helpers
# ---------------------------------------------------------------------------

def compute_graph_metrics(G):
    """Compute structural metrics for a NetworkX graph. Safe for empty graphs."""
    if G.number_of_nodes() == 0:
        return {
            "density": 0.0,
            "num_components": 0,
            "diameter": -1,
            "avg_clustering": 0.0,
            "avg_degree": 0.0,
        }

    components = list(nx.connected_components(G))
    num_components = len(components)

    if G.number_of_nodes() > 1:
        largest = max(components, key=len)
        sub = G.subgraph(largest)
        try:
            diameter = nx.diameter(sub)
        except nx.NetworkXError:
            diameter = -1
    else:
        diameter = -1

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

    return {
        "density": round(nx.density(G), 4),
        "num_components": num_components,
        "diameter": diameter,
        "avg_clustering": round(nx.average_clustering(G), 4),
        "avg_degree": round(avg_degree, 2),
    }


def compute_node_metrics(G):
    """Compute per-node metrics for a NetworkX graph. Returns dict keyed by node id."""
    if G.number_of_nodes() == 0:
        return {}

    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)

    return {
        node: {
            "degree": G.degree(node),
            "betweenness_centrality": round(betweenness[node], 4),
            "closeness_centrality": round(closeness[node], 4),
            "clustering": round(clustering[node], 4),
        }
        for node in G.nodes()
    }


def create_master_layout(df_all):
    """Circular layout with outside at 12 o'clock, text at 6."""
    all_panels = sorted(df_all["Panel_Title"].dropna().unique())
    all_panels = [p for p in all_panels if p != ""]

    panel_classification = {p: classify_panel(p) for p in all_panels}

    classification_order = ["outside", "text", "instruction", "picture", "other"]
    grouped_panels = []
    for cls in classification_order:
        grouped_panels.extend(sorted(p for p in all_panels if panel_classification[p] == cls))

    G = nx.Graph()
    for p in grouped_panels:
        G.add_node(p)

    pos = nx.circular_layout(G, scale=1.0)

    # Rotate so 'outside' is at 12 o'clock (-pi/2)
    if "outside" in pos:
        ox, oy = pos["outside"]
        cur_angle = math.atan2(oy, ox)
        rot = -math.pi / 2 - cur_angle
        for p in pos:
            x, y = pos[p]
            r = math.sqrt(x * x + y * y)
            a = math.atan2(y, x) + rot
            pos[p] = (r * math.cos(a), r * math.sin(a))

    # Partial rotation to push text elements toward 6 o'clock
    text_panels = [p for p in grouped_panels if panel_classification[p] == "text"]
    if text_panels:
        tx = sum(pos[p][0] for p in text_panels) / len(text_panels)
        ty = sum(pos[p][1] for p in text_panels) / len(text_panels)
        cur_text_angle = math.atan2(ty, tx)
        partial_rot = (math.pi / 2 - cur_text_angle) * 0.3
        for p in pos:
            x, y = pos[p]
            r = math.sqrt(x * x + y * y)
            a = math.atan2(y, x) + partial_rot
            pos[p] = (r * math.cos(a), r * math.sin(a))

    print(f"Master layout: {len(pos)} nodes")
    return pos, grouped_panels, panel_classification


# ---------------------------------------------------------------------------
# Graph data pipeline
# ---------------------------------------------------------------------------

def compute_all_graph_data(df, user_ids, master_pos, all_panel_titles, panel_classification,
                           min_duration, min_frequency, edge_weight_method, edge_representation):
    """Return dict keyed by 'userId|taskId|modality' with nodes/edges for Cytoscape,
    plus a user_tasks mapping of userId -> [taskIds], and available_modalities."""

    df_network = df[df["Panel_Title"].notna() & (df["Panel_Title"] != "")].copy()
    modalities = ["combined", "mclick", "eTrack"]

    all_graphs = {}
    user_tasks = {}  # userId -> sorted list of task IDs
    available_modalities = {}  # userId|taskId -> list of available modalities

    # Convert positions once: scale by 300, centre at 400,400, invert Y
    pixel_pos = {}
    for p in all_panel_titles:
        x, y = master_pos[p]
        pixel_pos[p] = (round(x * 300 + 400, 2), round(-y * 300 + 400, 2))

    for uid in user_ids:
        df_user = df_network[df_network["User_ID"] == uid]
        unique_tasks = sorted([t for t in df_user["Task_ID"].dropna().unique() if t != ""])
        user_tasks[uid] = unique_tasks
        print(f"\n  User {uid}: {len(unique_tasks)} task(s)")

        for task_id in unique_tasks:
            task_data = df_user[df_user["Task_ID"] == task_id].copy()
            if len(task_data) < 2:
                continue

            for mod in modalities:
                if mod == "combined":
                    mod_data = task_data
                else:
                    mod_data = task_data[task_data["modality"] == mod].copy()

                if len(mod_data) < 2:
                    continue

                time_spent = calculate_time_spent(mod_data)
                edge_freq, edge_tw = calculate_edge_weights(mod_data, edge_weight_method)

                # Filter nodes by min_duration
                filtered_ts = {k: v for k, v in time_spent.items() if v >= min_duration}
                if not filtered_ts:
                    continue

                # Filter edges by min_frequency and node membership
                filtered_ef = {}
                filtered_etw = {}
                for ek, freq in edge_freq.items():
                    n1, n2 = ek
                    if freq >= min_frequency and n1 in filtered_ts and n2 in filtered_ts:
                        filtered_ef[ek] = freq
                        filtered_etw[ek] = edge_tw[ek]

                # Choose edge weights for width scaling
                if edge_representation == "frequency":
                    weight_source = filtered_ef
                else:
                    weight_source = filtered_etw

                # Min-max scaling for edge widths (1-10 px)
                width_map = {}
                if weight_source:
                    vals = list(weight_source.values())
                    mn, mx = min(vals), max(vals)
                    for ek, w in weight_source.items():
                        if mx > mn:
                            width_map[ek] = 1 + 9 * (w - mn) / (mx - mn)
                        else:
                            width_map[ek] = 5

                # Build node list
                nodes = []
                for p in all_panel_titles:
                    cls = panel_classification.get(p, "other")
                    color = FIXED_COLORS.get(cls, "#7f7f7f")
                    active = p in filtered_ts
                    px, py = pixel_pos[p]
                    nodes.append({
                        "id": p,
                        "classification": cls,
                        "color": color,
                        "active": active,
                        "time_spent": round(filtered_ts.get(p, 0), 2),
                        "x": px,
                        "y": py,
                    })

                # Build edge list
                edges = []
                for ek, freq in filtered_ef.items():
                    n1, n2 = ek
                    tw = round(filtered_etw.get(ek, 0), 2)
                    tw_avg = round(tw / freq if freq > 0 else 0, 2)
                    w = round(width_map.get(ek, 1), 2)
                    is_text_edge = n1 in SPECIAL_TEXT_NODES or n2 in SPECIAL_TEXT_NODES
                    is_outside_edge = n1 == "outside" or n2 == "outside"
                    edges.append({
                        "source": n1,
                        "target": n2,
                        "frequency": freq,
                        "time_weight": tw,
                        "time_weight_avg": tw_avg,
                        "width": w,
                        "color": "#C227F5" if is_text_edge else "#cccccc",
                        "is_outside_edge": is_outside_edge,
                    })

                # Build NetworkX subgraph from active nodes/edges for metric computation
                G_active = nx.Graph()
                G_active.add_nodes_from(filtered_ts.keys())
                for ek in filtered_ef:
                    G_active.add_edge(ek[0], ek[1])

                graph_metrics = compute_graph_metrics(G_active)
                node_metrics = compute_node_metrics(G_active)

                # Merge node metrics into node dicts
                zero_metrics = {
                    "degree": 0,
                    "betweenness_centrality": 0.0,
                    "closeness_centrality": 0.0,
                    "clustering": 0.0,
                }
                for node_dict in nodes:
                    node_dict.update(node_metrics.get(node_dict["id"], zero_metrics))

                key = f"{uid}|{task_id}|{mod}"
                all_graphs[key] = {"nodes": nodes, "edges": edges, "graph_metrics": graph_metrics}

                # Track available modalities
                task_key = f"{uid}|{task_id}"
                if task_key not in available_modalities:
                    available_modalities[task_key] = []
                available_modalities[task_key].append(mod)

                print(f"    {task_id}|{mod}: {sum(1 for n in nodes if n['active'])} active nodes, {len(edges)} edges")

    return all_graphs, user_tasks, available_modalities


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Network Graph Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f5f6fa;color:#333}
#topbar{display:flex;align-items:center;gap:18px;padding:10px 24px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#topbar h1{font-size:18px;font-weight:700;white-space:nowrap}
#topbar label{font-size:13px;font-weight:600}
#topbar select{padding:4px 8px;font-size:13px;border:1px solid #bbb;border-radius:4px}
#main{display:flex;height:calc(100vh - 50px)}
#cy{flex:1;background:#fff;border-right:1px solid #ddd}
#sidebar{width:280px;overflow-y:auto;padding:16px;background:#fafbfc;font-size:13px}
#sidebar h3{margin:0 0 8px;font-size:14px;border-bottom:1px solid #ddd;padding-bottom:4px}
.legend-item{display:flex;align-items:center;gap:8px;margin:4px 0}
.legend-swatch{width:16px;height:16px;border-radius:3px;border:1.5px solid #555;flex-shrink:0}
.legend-swatch.hollow{background:#fff}
.stat-row{display:flex;justify-content:space-between;padding:3px 0}
.stat-label{color:#666}
.stat-value{font-weight:600}
.section{margin-bottom:18px}
#tooltip{position:absolute;display:none;background:rgba(30,30,30,.92);color:#fff;padding:8px 12px;border-radius:6px;font-size:12px;pointer-events:none;z-index:999;max-width:280px;line-height:1.5}
.complexity-indicator{display:inline-block;width:12px;height:12px;border-radius:50%;border:1px solid #ccc}
.complexity-low{background:#C8E6C9}
.complexity-medium{background:#FFE0B2}
.complexity-high{background:#FFCDD2}
.collapsible-header{cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none}
.collapsible-header:hover{color:#4a90d9}
.collapsible-arrow{font-size:10px;transition:transform .2s}
.collapsible-arrow.collapsed{transform:rotate(-90deg)}
.dist-channel{margin-bottom:10px}
.dist-channel-label{font-size:11px;font-weight:600;color:#555;margin-bottom:3px;border-bottom:1px solid #eee;padding-bottom:2px}
.dist-stats{display:flex;gap:6px;flex-wrap:wrap;font-size:11px;margin-bottom:4px}
.dist-stat{color:#666}
.dist-stat span{font-weight:600;color:#333}
.dist-sparkline{width:100%;height:32px;display:block}
.dist-empty{color:#999;font-size:12px;font-style:italic}
</style>
</head>
<body>

<div id="topbar">
  <h1>Network Graph Dashboard</h1>
  <div>
    <label for="selUser">User</label>
    <select id="selUser"></select>
  </div>
  <div>
    <label for="selTask">Task ID</label>
    <select id="selTask"></select>
  </div>
  <div>
    <label for="selMod">Modality</label>
    <select id="selMod">
      <option value="mclick">mclick</option>
      <option value="eTrack">eTrack</option>
    </select>
  </div>
  <div style="border:1px solid #ddd;padding:8px;border-radius:4px;background:#f9f9f9">
    <div style="font-weight:600;margin-bottom:8px">Interactive Filters</div>
    <div>
      <label>Min Duration (sec):</label>
      <input type="range" id="sliderMinDuration" min="0" max="10" step="0.5" value="0" style="width:100px">
      <span id="valMinDuration">0</span>s
    </div>
    <div>
      <label>Min Frequency:</label>
      <input type="range" id="sliderMinFrequency" min="1" max="10" step="1" value="1" style="width:100px">
      <span id="valMinFrequency">1</span>
    </div>
    <div>
      <label>Edge Weight:</label>
      <label style="margin-left:12px"><input type="radio" name="edgeWeight" value="total" checked> Total</label>
      <label style="margin-left:12px"><input type="radio" name="edgeWeight" value="average"> Average</label>
    </div>
    <div>
      <label>Edge Representation:</label>
      <label style="margin-left:12px"><input type="radio" name="edgeRepr" value="time" checked> Time</label>
      <label style="margin-left:12px"><input type="radio" name="edgeRepr" value="frequency"> Frequency</label>
    </div>
  </div>
  <div>
    <label>
      <input type="checkbox" id="chkShowOutsideEdges" checked>
      Show edges to "outside"
    </label>
  </div>
</div>

<div id="main">
  <div id="cy"></div>
  <div id="sidebar">

    <div class="section" id="legendSection">
      <h3>Legend</h3>
      <div id="dynamicLegend"></div>
      <div style="height:6px"></div>
      <div class="legend-item"><div class="legend-swatch hollow" style="border-color:#2ca02c"></div> Inactive node (hollow, coloured border)</div>
      <div style="height:6px"></div>
      <div class="legend-item"><div class="legend-swatch" style="background:#C227F5;border-color:#C227F5"></div> Edge to/from node</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#cccccc;border-color:#999"></div> Other edge</div>
    </div>

    <div class="section" id="statsSection">
      <h3>Statistics</h3>
      <div class="stat-row"><span class="stat-label">Active nodes</span><span class="stat-value" id="statActive">-</span></div>
      <div class="stat-row"><span class="stat-label">Total nodes</span><span class="stat-value" id="statTotal">-</span></div>
      <div class="stat-row"><span class="stat-label">Edges</span><span class="stat-value" id="statEdges">-</span></div>
      <div class="stat-row"><span class="stat-label">Total time</span><span class="stat-value" id="statTime">-</span></div>
      <div class="stat-row"><span class="stat-label">Total transitions</span><span class="stat-value" id="statTrans">-</span></div>
      <div style="height:8px"></div>
      <div class="stat-row"><span class="stat-label">Complexity Score</span><span class="stat-value" id="statComplexity">-</span></div>
      <div class="stat-row"><span class="stat-label">Group</span><span class="stat-value" id="statGroup" style="display:flex;align-items:center;gap:6px">-</span></div>
    </div>

    <div class="section" id="graphMetricsSection">
      <h3 class="collapsible-header" onclick="toggleCollapsible('graphMetricsBody','graphMetricsArrow')">
        Graph Metrics (precomputed)
        <span class="collapsible-arrow" id="graphMetricsArrow">&#9660;</span>
      </h3>
      <div id="graphMetricsBody">
        <div class="stat-row"><span class="stat-label">Density</span><span class="stat-value" id="gmDensity">-</span></div>
        <div class="stat-row"><span class="stat-label">Components</span><span class="stat-value" id="gmComponents">-</span></div>
        <div class="stat-row"><span class="stat-label">Diameter</span><span class="stat-value" id="gmDiameter">-</span></div>
        <div class="stat-row"><span class="stat-label">Avg Clustering</span><span class="stat-value" id="gmClustering">-</span></div>
        <div class="stat-row"><span class="stat-label">Avg Degree</span><span class="stat-value" id="gmAvgDegree">-</span></div>
      </div>
    </div>

    <div class="section" id="edgeDistSection">
      <h3 class="collapsible-header" onclick="toggleCollapsible('edgeDistBody','edgeDistArrow')">
        Edge Distribution (filtered)
        <span class="collapsible-arrow" id="edgeDistArrow">&#9660;</span>
      </h3>
      <div id="edgeDistBody">
        <div id="edgeDistContent"><span class="dist-empty">Load a graph to see distribution.</span></div>
      </div>
    </div>

    <div class="section" id="detailSection">
      <h3>Details</h3>
      <p id="detailContent" style="color:#999">Click a node or edge for details.</p>
    </div>

  </div>
</div>

<div id="tooltip"></div>

<script>
// ---- Embedded graph data ----
var GRAPHS = {{GRAPH_JSON}};
var USER_TASKS = {{USER_TASKS}};
var USER_IDS = {{USER_IDS}};
var AVAILABLE_MODALITIES = {{AVAILABLE_MODALITIES}};

// ---- State ----
var cy = null;
var highlightActive = false;
var allComplexityScores = {};  // Store all user scores for grouping
var groupThresholds = { low: 0, med: 0 };  // Tertile thresholds
var currentRawData = null;  // Store raw unfiltered data

// ---- Complexity calculation ----
function calculateComplexityScore(activeNodes, edgeCount){
  return activeNodes + edgeCount;
}

function calculateTertiles(scores){
  if(scores.length === 0) return [0, 0];
  var sorted = scores.slice().sort(function(a, b){ return a - b; });
  var p33 = sorted[Math.floor(sorted.length * 0.33)];
  var p66 = sorted[Math.floor(sorted.length * 0.66)];
  return [p33, p66];
}

function determineGroup(score, thLow, thMed){
  if(score < thLow) return 'low';
  if(score < thMed) return 'medium';
  return 'high';
}

function precomputeAllComplexityScores(){
  allComplexityScores = {};
  USER_IDS.forEach(function(uid){
    var tasks = USER_TASKS[uid] || [];
    tasks.forEach(function(task){
      ['mclick', 'eTrack'].forEach(function(mod){
        var key = uid + '|' + task + '|' + mod;
        var data = GRAPHS[key];
        if(data){
          var activeCount = data.nodes.filter(function(n){ return n.active; }).length;
          allComplexityScores[key] = calculateComplexityScore(activeCount, data.edges.length);
        }
      });
    });
  });
  // Calculate thresholds based on all scores
  var allScores = Object.values(allComplexityScores);
  var tertiles = calculateTertiles(allScores);
  groupThresholds.low = tertiles[0];
  groupThresholds.med = tertiles[1];
}

// ---- Populate user dropdown ----
(function(){
  var sel = document.getElementById('selUser');
  USER_IDS.forEach(function(u){
    var o = document.createElement('option');
    o.value = u; o.textContent = u;
    sel.appendChild(o);
  });
})();

function populateTasks(){
  var uid = document.getElementById('selUser').value;
  var sel = document.getElementById('selTask');
  sel.innerHTML = '';
  var tasks = USER_TASKS[uid] || [];
  tasks.forEach(function(t){
    var o = document.createElement('option');
    o.value = t; o.textContent = t;
    sel.appendChild(o);
  });
  populateModalities();
}

function populateModalities(){
  var uid = document.getElementById('selUser').value;
  var taskId = document.getElementById('selTask').value;
  var modSel = document.getElementById('selMod');
  modSel.innerHTML = '';

  var key = uid + '|' + taskId;
  var availableMods = AVAILABLE_MODALITIES[key] || [];

  availableMods.forEach(function(m){
    var o = document.createElement('option');
    o.value = m; o.textContent = m;
    modSel.appendChild(o);
  });

  loadGraph();
}

// ---- Apply filters and redraw ----
function applyFilters(){
  if(!currentRawData || !cy) return;

  var minDuration = parseFloat(document.getElementById('sliderMinDuration').value);
  var minFrequency = parseInt(document.getElementById('sliderMinFrequency').value);
  var edgeWeight = document.querySelector('input[name="edgeWeight"]:checked').value;
  var edgeRepr = document.querySelector('input[name="edgeRepr"]:checked').value;

  document.getElementById('valMinDuration').textContent = minDuration;
  document.getElementById('valMinFrequency').textContent = minFrequency;

  // Step 1: Determine which nodes pass the min duration filter
  var visibleNodeIds = new Set();
  currentRawData.timeSpent.forEach(function(item){
    if(item.time >= minDuration){
      visibleNodeIds.add(item.node);
    }
  });

  // Step 2: Filter edges - both nodes must be visible AND meet min frequency
  var filteredEdges = [];
  currentRawData.edges.forEach(function(edge){
    var bothNodesVisible = visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target);
    var meetsFrequency = edge.frequency >= minFrequency;

    if(bothNodesVisible && meetsFrequency){
      // Choose weight for display (for label and thickness)
      var displayWeight;
      var displayLabel;

      if(edgeRepr === 'frequency'){
        displayWeight = edge.frequency;
        displayLabel = String(edge.frequency);
      } else {
        displayWeight = edgeWeight === 'average' ? edge.timeWeightAvg : edge.timeWeight;
        displayLabel = displayWeight > 0 ? displayWeight + 's' : '';
      }

      filteredEdges.push({
        ...edge,
        displayWeight: displayWeight,
        displayLabel: displayLabel
      });
    }
  });

  // Step 3: Remove all old edges and nodes
  cy.elements('edge').remove();

  // Step 4: Update node visibility (show/hide)
  cy.elements('node').forEach(function(node){
    var nodeId = node.data('id');
    if(visibleNodeIds.has(nodeId)){
      node.show();
    } else {
      node.hide();
    }
  });

  // Step 5: Calculate edge widths based on displayWeight
  var edgeWidthMap = {};
  if(filteredEdges.length > 0){
    var vals = filteredEdges.map(function(e){ return e.displayWeight; });
    var mn = Math.min.apply(null, vals);
    var mx = Math.max.apply(null, vals);
    filteredEdges.forEach(function(e){
      var key = e.source + '|' + e.target;
      edgeWidthMap[key] = mx > mn ? 1 + 9 * (e.displayWeight - mn) / (mx - mn) : 5;
    });
  }

  // Step 6: Add back filtered edges only
  filteredEdges.forEach(function(e, i){
    var key = e.source + '|' + e.target;
    cy.add({
      group: 'edges',
      data: {
        id: 'e' + i,
        source: e.source,
        target: e.target,
        frequency: e.frequency,
        time_weight: e.displayWeight,
        width: edgeWidthMap[key] || 5,
        color: e.color,
        label: e.displayLabel,
        isOutsideEdge: e.isOutsideEdge
      }
    });
  });

  // Apply outside edges visibility
  var showOutside = document.getElementById('chkShowOutsideEdges').checked;
  cy.elements('edge[isOutsideEdge = true]').forEach(function(edge){
    if(!showOutside) edge.hide(); else edge.show();
  });
}

document.getElementById('selUser').addEventListener('change', populateTasks);
document.getElementById('selTask').addEventListener('change', populateModalities);
document.getElementById('selMod').addEventListener('change', loadGraph);

// Filter controls
document.getElementById('sliderMinDuration').addEventListener('change', applyFilters);
document.getElementById('sliderMinFrequency').addEventListener('change', applyFilters);
document.querySelectorAll('input[name="edgeWeight"]').forEach(function(radio){
  radio.addEventListener('change', applyFilters);
});
document.querySelectorAll('input[name="edgeRepr"]').forEach(function(radio){
  radio.addEventListener('change', applyFilters);
});

// ---- Cytoscape init ----
function loadGraph(){
  var userId = document.getElementById('selUser').value;
  var taskId = document.getElementById('selTask').value;
  var mod    = document.getElementById('selMod').value;
  var key    = userId + '|' + taskId + '|' + mod;
  var data   = GRAPHS[key];

  if(!data){
    if(cy){ cy.destroy(); cy = null; }
    currentRawData = null;
    document.getElementById('statActive').textContent = '-';
    document.getElementById('statTotal').textContent = '-';
    document.getElementById('statEdges').textContent = '-';
    document.getElementById('statTime').textContent = '-';
    document.getElementById('statTrans').textContent = '-';
    document.getElementById('statComplexity').textContent = '-';
    document.getElementById('statGroup').textContent = '-';
    document.getElementById('detailContent').textContent = 'No data for this combination.';
    return;
  }

  var elements = [];

  data.nodes.forEach(function(n){
    elements.push({
      group: 'nodes',
      data: {
        id: n.id,
        classification: n.classification,
        color: n.color,
        active: n.active,
        time_spent: n.time_spent,
        bgColor: n.active ? n.color : '#fff',
        borderColor: n.color,
        borderWidth: n.active ? 0 : 2,
        nodeOpacity: n.active ? 1 : 0.5,
        label: n.id
      },
      position: { x: n.x, y: n.y }
    });
  });

  // Store raw data for filtering
  currentRawData = {
    timeSpent: [],
    edges: []
  };
  data.nodes.forEach(function(n){
    currentRawData.timeSpent.push({node: n.id, time: n.time_spent});
  });
  data.edges.forEach(function(e){
    currentRawData.edges.push({
      id: e.source + '|' + e.target,
      source: e.source,
      target: e.target,
      frequency: e.frequency,
      timeWeight: e.time_weight,
      timeWeightAvg: e.time_weight_avg || (e.time_weight / e.frequency),
      color: e.color,
      isOutsideEdge: e.source === 'outside' || e.target === 'outside'
    });
  });

  data.edges.forEach(function(e, i){
    var isOutsideEdge = e.source === 'outside' || e.target === 'outside';
    elements.push({
      group: 'edges',
      data: {
        id: 'e' + i,
        source: e.source,
        target: e.target,
        frequency: e.frequency,
        time_weight: e.time_weight,
        width: e.width,
        color: e.color,
        label: e.time_weight > 0 ? e.time_weight + 's' : String(e.frequency),
        isOutsideEdge: isOutsideEdge
      }
    });
  });

  if(cy){ cy.destroy(); }
  highlightActive = false;

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: elements,
    layout: { name: 'preset' },
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'text-wrap': 'wrap',
          'text-max-width': '100px',
          'font-size': '10px',
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-margin-y': 6,
          'width': 28,
          'height': 28,
          'background-color': 'data(bgColor)',
          'border-color': 'data(borderColor)',
          'border-width': 'data(borderWidth)',
          'opacity': 'data(nodeOpacity)',
          'text-opacity': 1
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 'data(width)',
          'line-color': 'data(color)',
          'curve-style': 'bezier',
          'opacity': 0.7,
          'label': 'data(label)',
          'font-size': '8px',
          'text-rotation': 'autorotate',
          'text-margin-y': -8,
          'color': '#555'
        }
      },
      // Dimmed classes for highlight mode
      {
        selector: '.dimmed',
        style: {
          'opacity': 0.08,
          'text-opacity': 0.15
        }
      },
      {
        selector: '.highlighted-node',
        style: {
          'opacity': 1,
          'text-opacity': 1,
          'border-width': 3,
          'border-color': '#333',
          'z-index': 10
        }
      },
      {
        selector: '.highlighted-neighbor',
        style: {
          'opacity': 1,
          'text-opacity': 1,
          'z-index': 9
        }
      },
      {
        selector: '.highlighted-edge',
        style: {
          'opacity': 0.85,
          'z-index': 9
        }
      }
    ],
    minZoom: 0.3,
    maxZoom: 3,
    userPanningEnabled: true,
    userZoomingEnabled: true
  });

  // ---- Click node -> highlight ----
  cy.on('tap', 'node', function(evt){
    var node = evt.target;
    highlightActive = true;

    cy.elements().removeClass('highlighted-node highlighted-neighbor highlighted-edge').addClass('dimmed');

    node.removeClass('dimmed').addClass('highlighted-node');

    var connEdges = node.connectedEdges();
    connEdges.removeClass('dimmed').addClass('highlighted-edge');

    var neighbors = node.neighborhood('node');
    neighbors.removeClass('dimmed').addClass('highlighted-neighbor');

    // Show detail
    var d = node.data();
    document.getElementById('detailContent').innerHTML =
      '<b>' + d.id + '</b><br>' +
      'Classification: ' + d.classification + '<br>' +
      'Active: ' + (d.active ? 'Yes' : 'No') + '<br>' +
      'Time spent: ' + d.time_spent + 's<br>' +
      'Connected edges: ' + connEdges.length;
  });

  // ---- Click edge -> show detail ----
  cy.on('tap', 'edge', function(evt){
    var d = evt.target.data();
    document.getElementById('detailContent').innerHTML =
      '<b>' + d.source + '</b> &harr; <b>' + d.target + '</b><br>' +
      'Frequency: ' + d.frequency + '<br>' +
      'Time weight: ' + d.time_weight + 's<br>' +
      'Width: ' + d.width + 'px';
  });

  // ---- Click background -> restore ----
  cy.on('tap', function(evt){
    if(evt.target === cy){
      if(highlightActive){
        cy.elements().removeClass('dimmed highlighted-node highlighted-neighbor highlighted-edge');
        // Remove inline style overrides so stylesheet/data-mapped values take effect again
        cy.elements().removeStyle('opacity text-opacity');
        highlightActive = false;
        document.getElementById('detailContent').innerHTML = '<span style="color:#999">Click a node or edge for details.</span>';
      }
    }
  });

  // ---- Hover tooltip ----
  var tooltip = document.getElementById('tooltip');

  cy.on('mouseover', 'node', function(evt){
    var d = evt.target.data();
    tooltip.innerHTML = '<b>' + d.id + '</b><br>Class: ' + d.classification +
      '<br>Active: ' + (d.active ? 'Yes' : 'No') +
      '<br>Time: ' + d.time_spent + 's';
    tooltip.style.display = 'block';
  });

  cy.on('mouseover', 'edge', function(evt){
    var d = evt.target.data();
    tooltip.innerHTML = d.source + ' &harr; ' + d.target +
      '<br>Freq: ' + d.frequency +
      '<br>Time: ' + d.time_weight + 's';
    tooltip.style.display = 'block';
  });

  cy.on('mousemove', function(evt){
    if(evt.originalEvent){
      tooltip.style.left = (evt.originalEvent.clientX + 14) + 'px';
      tooltip.style.top  = (evt.originalEvent.clientY + 14) + 'px';
    }
  });

  cy.on('mouseout', 'node, edge', function(){
    tooltip.style.display = 'none';
  });

  // ---- Setup outside edges toggle ----
  var chk = document.getElementById('chkShowOutsideEdges');
  function updateOutsideEdgesVisibility(){
    if(cy){
      var showOutside = chk.checked;
      cy.elements('edge').forEach(function(edge){
        var isOutside = edge.data('isOutsideEdge');
        if(isOutside){
          if(showOutside){
            edge.show();
          } else {
            edge.hide();
          }
        }
      });
    }
  }
  chk.removeEventListener('change', updateOutsideEdgesVisibility);
  chk.addEventListener('change', updateOutsideEdgesVisibility);

  // ---- Update legend ----
  var legendDiv = document.getElementById('dynamicLegend');
  legendDiv.innerHTML = '';
  data.nodes.forEach(function(n){
    var item = document.createElement('div');
    item.className = 'legend-item';
    var swatch = document.createElement('div');
    swatch.className = 'legend-swatch';
    swatch.style.background = n.color;
    if(!n.active){
      swatch.className += ' hollow';
      swatch.style.borderColor = n.color;
      swatch.style.background = '#fff';
    }
    var label = document.createElement('span');
    label.textContent = n.id + (n.active ? ' (active)' : ' (inactive)');
    item.appendChild(swatch);
    item.appendChild(label);
    legendDiv.appendChild(item);
  });

  // ---- Update stats ----
  var activeCount = 0, totalTime = 0, totalTrans = 0;
  data.nodes.forEach(function(n){ if(n.active){ activeCount++; totalTime += n.time_spent; } });
  data.edges.forEach(function(e){ totalTrans += e.frequency; });

  var complexityScore = calculateComplexityScore(activeCount, data.edges.length);
  var group = determineGroup(complexityScore, groupThresholds.low, groupThresholds.med);
  var groupColors = { low: 'C8E6C9', medium: 'FFE0B2', high: 'FFCDD2' };
  var groupLabels = { low: 'Low', medium: 'Medium', high: 'High' };

  document.getElementById('statActive').textContent = activeCount;
  document.getElementById('statTotal').textContent  = data.nodes.length;
  document.getElementById('statEdges').textContent   = data.edges.length;
  document.getElementById('statTime').textContent    = totalTime.toFixed(1) + 's';
  document.getElementById('statTrans').textContent   = totalTrans;
  document.getElementById('statComplexity').textContent = complexityScore;
  document.getElementById('statGroup').innerHTML = '<div class="complexity-indicator complexity-' + group + '" style="background:#' + groupColors[group] + '"></div>' + groupLabels[group];
  document.getElementById('detailContent').innerHTML = '<span style="color:#999">Click a node or edge for details.</span>';
}

// ---- Read ?user= query param for deep-linking ----
(function(){
  var params = new URLSearchParams(window.location.search);
  var qUser = params.get('user');
  if(qUser){
    var sel = document.getElementById('selUser');
    for(var i = 0; i < sel.options.length; i++){
      if(sel.options[i].value === qUser){
        sel.value = qUser;
        break;
      }
    }
  }
})();

// Initial load — precompute complexity scores, populate tasks for first user, then load graph
precomputeAllComplexityScores();
populateTasks();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Snapshot template (all-users grid overview)
# ---------------------------------------------------------------------------

SNAPSHOT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Snapshot Overview</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f5f6fa;color:#333}
#topbar{position:sticky;top:0;z-index:100;padding:12px 24px;background:#fff;border-bottom:1px solid #ddd}
#topbar h1{font-size:18px;font-weight:700;margin-bottom:10px}
.toggle-group{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.toggle-group label{font-size:13px;font-weight:600;margin-right:4px}
.pill{display:inline-block;padding:4px 14px;font-size:12px;border:1px solid #bbb;border-radius:16px;cursor:pointer;background:#fff;color:#555;transition:all .15s}
.pill:hover{background:#eee}
.pill.active{background:#4a90d9;color:#fff;border-color:#4a90d9}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;padding:20px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;cursor:pointer;transition:box-shadow .15s}
.card:hover{box-shadow:0 2px 12px rgba(0,0,0,.12)}
.card-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#fafbfc;border-bottom:1px solid #eee;font-size:13px;transition:background-color .2s}
.card-header .uid{font-weight:700}
.card-header .stats{color:#666;font-size:12px}
.card-header .complexity{font-size:11px;color:#888;margin-left:8px}
.card-cy{width:100%;height:400px}
.card-nodata{width:100%;height:400px;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:14px}
.group-low{background:#C8E6C9}
.group-medium{background:#FFE0B2}
.group-high{background:#FFCDD2}
.controls-section{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:12px;padding:12px;background:#f9f9f9;border-radius:4px;border:1px solid #eee}
.controls-section label{font-size:12px;font-weight:600;color:#555}
.threshold-input{width:60px;padding:4px 6px;font-size:12px;border:1px solid #ccc;border-radius:4px}
.reset-btn{padding:6px 12px;font-size:12px;background:#fff;border:1px solid #bbb;border-radius:4px;cursor:pointer;transition:all .15s}
.reset-btn:hover{background:#eee}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;padding:12px;background:#f9f9f9;border-radius:4px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px}
.legend-color{width:24px;height:24px;border:1px solid #ccc;border-radius:2px}
.group-stats{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;padding:12px;background:#f9f9f9;border-radius:4px;font-size:12px}
.group-stat-item{display:flex;align-items:center;gap:6px}
.group-stat-count{font-weight:600}
</style>
</head>
<body>

<div id="topbar">
  <h1>Snapshot Overview — All Users</h1>
  <div class="toggle-group">
    <label>Task ID:</label>
    <span id="taskToggles"></span>
  </div>
  <div class="toggle-group">
    <label>Modality:</label>
    <span id="modToggles"></span>
  </div>
  <div class="controls-section">
    <div>
      <label>Low threshold:</label>
      <input type="number" id="thresholdLow" class="threshold-input" value="0" min="0">
    </div>
    <div>
      <label>Medium threshold:</label>
      <input type="number" id="thresholdMed" class="threshold-input" value="0" min="0">
    </div>
    <button class="reset-btn" id="resetBtn">Reset to Tertiles</button>
    <label style="margin-left:12px">
      <input type="checkbox" id="chkShowOutsideEdgesSnapshot" checked>
      Show edges to "outside"
    </label>
  </div>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-color" style="background:#C8E6C9"></div>
      <span>Low Complexity</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background:#FFE0B2"></div>
      <span>Medium Complexity</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background:#FFCDD2"></div>
      <span>High Complexity</span>
    </div>
  </div>
  <div class="group-stats">
    <div class="group-stat-item">
      <span style="color:#C8E6C9;font-weight:700">●</span>
      <span>Low: <span class="group-stat-count" id="countLow">0</span></span>
    </div>
    <div class="group-stat-item">
      <span style="color:#FFB74D;font-weight:700">●</span>
      <span>Medium: <span class="group-stat-count" id="countMed">0</span></span>
    </div>
    <div class="group-stat-item">
      <span style="color:#EF9A9A;font-weight:700">●</span>
      <span>High: <span class="group-stat-count" id="countHigh">0</span></span>
    </div>
  </div>
</div>

<div id="grid"></div>

<script>
var GRAPHS = {{GRAPH_JSON}};
var USER_TASKS = {{USER_TASKS}};
var USER_IDS = {{USER_IDS}};
var ALL_TASK_IDS = {{ALL_TASK_IDS}};
var AVAILABLE_MODALITIES = {{AVAILABLE_MODALITIES}};

var currentTask = ALL_TASK_IDS.length ? ALL_TASK_IDS[0] : '';
var currentMod = '';  // Will be set dynamically
var cyInstances = {};  // userId -> cytoscape instance
var complexityScores = {};  // userId -> complexity score for current task/mod
var thresholdLow = 0;
var thresholdMed = 0;

// ---- Complexity calculation ----
function calculateComplexityScore(activeNodes, edgeCount){
  return activeNodes + edgeCount;
}

function calculateTertiles(scores){
  if(scores.length === 0) return [0, 0];
  var sorted = scores.slice().sort(function(a, b){ return a - b; });
  var p33 = sorted[Math.floor(sorted.length * 0.33)];
  var p66 = sorted[Math.floor(sorted.length * 0.66)];
  return [p33, p66];
}

function determineGroup(score, thLow, thMed){
  if(score < thLow) return 'low';
  if(score < thMed) return 'medium';
  return 'high';
}

function updateAllComplexityScores(){
  complexityScores = {};
  USER_IDS.forEach(function(uid){
    var data = null;
    if(currentTask === 'combined' || currentMod === 'combined'){
      var tasks = currentTask === 'combined' ? (USER_TASKS[uid] || []) : [currentTask];
      var modalities = currentMod === 'combined' ? ['mclick', 'eTrack'] : [currentMod];
      var mergedNodes = {};
      var edgeCount = 0;
      tasks.forEach(function(task){
        modalities.forEach(function(mod){
          var key = uid + '|' + task + '|' + mod;
          var taskData = GRAPHS[key];
          if(taskData){
            taskData.nodes.forEach(function(n){
              if(n.active) mergedNodes[n.id] = true;
            });
            edgeCount += taskData.edges.length;
          }
        });
      });
      if(Object.keys(mergedNodes).length > 0){
        complexityScores[uid] = calculateComplexityScore(Object.keys(mergedNodes).length, edgeCount);
      }
    } else {
      var key = uid + '|' + currentTask + '|' + currentMod;
      data = GRAPHS[key];
      if(data){
        var activeCount = data.nodes.filter(function(n){ return n.active; }).length;
        complexityScores[uid] = calculateComplexityScore(activeCount, data.edges.length);
      }
    }
  });
}

function calculateAndSetTertiles(){
  var scores = Object.values(complexityScores);
  var tertiles = calculateTertiles(scores);
  thresholdLow = tertiles[0];
  thresholdMed = tertiles[1];
  document.getElementById('thresholdLow').value = Math.round(thresholdLow);
  document.getElementById('thresholdMed').value = Math.round(thresholdMed);
}

function getSortedUsersByComplexity(){
  var usersWithScores = USER_IDS.map(function(uid){
    return { uid: uid, score: complexityScores[uid] || 0 };
  });
  usersWithScores.sort(function(a, b){ return a.score - b.score; });
  return usersWithScores.map(function(item){ return item.uid; });
}

function reorganizeGrid(){
  var grid = document.getElementById('grid');
  var sortedUserIds = getSortedUsersByComplexity();
  sortedUserIds.forEach(function(uid){
    var card = document.querySelector('[data-uid="' + uid + '"]');
    if(card){
      grid.appendChild(card);  // Move to end (reorders in DOM)
    }
  });
}

function updateGroupColors(){
  var counts = { low: 0, medium: 0, high: 0 };
  USER_IDS.forEach(function(uid){
    var score = complexityScores[uid] || 0;
    var group = determineGroup(score, thresholdLow, thresholdMed);
    counts[group]++;
    var header = document.querySelector('[data-uid="' + uid + '"] .card-header');
    if(header){
      header.className = 'card-header group-' + group;
    }
  });
  reorganizeGrid();
  document.getElementById('countLow').textContent = counts.low;
  document.getElementById('countMed').textContent = counts.medium;
  document.getElementById('countHigh').textContent = counts.high;
}

// ---- Build toggle pills ----
function buildToggles(){
  var taskC = document.getElementById('taskToggles');
  ALL_TASK_IDS.forEach(function(t){
    var span = document.createElement('span');
    span.className = 'pill' + (t === currentTask ? ' active' : '');
    span.textContent = t;
    span.dataset.task = t;
    span.addEventListener('click', function(){
      currentTask = t;
      document.querySelectorAll('#taskToggles .pill').forEach(function(p){ p.classList.toggle('active', p.dataset.task === t); });
      updateModalityToggles();
      updateAllComplexityScores();
      calculateAndSetTertiles();
      refreshAll();
      updateGroupColors();
    });
    taskC.appendChild(span);
  });

  // Build modality toggles based on available modalities for first task
  updateModalityToggles();
}

function updateModalityToggles(){
  var modC = document.getElementById('modToggles');
  modC.innerHTML = '';

  // Get available modalities for current task
  var availableMods = new Set();
  availableMods.add('combined');  // Always include combined

  if(currentTask === 'combined'){
    // For combined task, get modalities from all actual tasks
    var allTasks = [];
    USER_IDS.forEach(function(uid){
      var tasks = USER_TASKS[uid] || [];
      tasks.forEach(function(t){ allTasks.push(t); });
    });
    allTasks = Array.from(new Set(allTasks));

    allTasks.forEach(function(task){
      USER_IDS.forEach(function(uid){
        var key = uid + '|' + task;
        var mods = AVAILABLE_MODALITIES[key] || [];
        mods.forEach(function(m){ availableMods.add(m); });
      });
    });
  } else {
    // For specific task, get modalities from that task
    USER_IDS.forEach(function(uid){
      var key = uid + '|' + currentTask;
      var mods = AVAILABLE_MODALITIES[key] || [];
      mods.forEach(function(m){ availableMods.add(m); });
    });
  }

  availableMods = Array.from(availableMods).sort();
  if(availableMods.length === 0) availableMods = ['combined', 'mclick', 'eTrack'];

  if(!currentMod || availableMods.indexOf(currentMod) === -1){
    currentMod = availableMods[0];
  }

  availableMods.forEach(function(m){
    var span = document.createElement('span');
    span.className = 'pill' + (m === currentMod ? ' active' : '');
    span.textContent = m;
    span.dataset.mod = m;
    span.addEventListener('click', function(){
      currentMod = m;
      document.querySelectorAll('#modToggles .pill').forEach(function(p){ p.classList.toggle('active', p.dataset.mod === m); });
      updateAllComplexityScores();
      calculateAndSetTertiles();
      refreshAll();
      updateGroupColors();
    });
    modC.appendChild(span);
  });
}

// ---- Build grid cards ----
function buildGrid(){
  var grid = document.getElementById('grid');
  USER_IDS.forEach(function(uid){
    var card = document.createElement('div');
    card.className = 'card';
    card.dataset.uid = uid;
    card.addEventListener('click', function(){ window.open('dashboard.html?user=' + encodeURIComponent(uid), '_blank'); });

    var header = document.createElement('div');
    header.className = 'card-header';
    var shortId = uid.length > 8 ? '...' + uid.slice(-8) : uid;
    header.innerHTML = '<span class="uid" title="' + uid + '">' + shortId + '</span><span class="stats" id="stats-' + uid + '"></span>';
    card.appendChild(header);

    var cyDiv = document.createElement('div');
    cyDiv.className = 'card-cy';
    cyDiv.id = 'cy-' + uid;
    card.appendChild(cyDiv);

    var nodata = document.createElement('div');
    nodata.className = 'card-nodata';
    nodata.id = 'nodata-' + uid;
    nodata.textContent = 'No data';
    nodata.style.display = 'none';
    card.appendChild(nodata);

    grid.appendChild(card);
  });
}

// ---- Load / refresh a single user card ----
function loadUserGraph(uid){
  var cyDiv = document.getElementById('cy-' + uid);
  var nodata = document.getElementById('nodata-' + uid);
  var statsEl = document.getElementById('stats-' + uid);

  // Destroy existing instance
  if(cyInstances[uid]){ cyInstances[uid].destroy(); cyInstances[uid] = null; }

  // Handle "combined" task or "combined" modality - merge data
  var data = null;
  if(currentTask === 'combined' || currentMod === 'combined'){
    var tasks = currentTask === 'combined' ? (USER_TASKS[uid] || []) : [currentTask];
    var modalities = currentMod === 'combined' ? ['mclick', 'eTrack'] : [currentMod];
    var mergedData = { nodes: [], edges: [] };
    var nodeMap = {};
    var edgeMap = {};

    tasks.forEach(function(task){
      modalities.forEach(function(mod){
        var key = uid + '|' + task + '|' + mod;
        var taskData = GRAPHS[key];
        if(taskData){
          // Merge nodes
          taskData.nodes.forEach(function(n){
            if(!nodeMap[n.id]){
              nodeMap[n.id] = JSON.parse(JSON.stringify(n));
            } else {
              // Update active status and sum time
              if(n.active) nodeMap[n.id].active = true;
              nodeMap[n.id].time_spent = (nodeMap[n.id].time_spent || 0) + n.time_spent;
            }
          });

          // Merge edges
          taskData.edges.forEach(function(e){
            var ek = e.source + '|' + e.target;
            if(!edgeMap[ek]){
              edgeMap[ek] = JSON.parse(JSON.stringify(e));
            } else {
              // Sum frequencies and time weights
              edgeMap[ek].frequency += e.frequency;
              edgeMap[ek].time_weight += e.time_weight;
              edgeMap[ek].time_weight_avg = edgeMap[ek].time_weight / edgeMap[ek].frequency;
            }
          });
        }
      });
    });

    mergedData.nodes = Object.values(nodeMap);
    mergedData.edges = Object.values(edgeMap);
    data = mergedData.nodes.length > 0 ? mergedData : null;
  } else {
    var key = uid + '|' + currentTask + '|' + currentMod;
    data = GRAPHS[key];
  }

  if(!data){
    cyDiv.style.display = 'none';
    nodata.style.display = 'flex';
    statsEl.textContent = '';
    return;
  }

  cyDiv.style.display = 'block';
  nodata.style.display = 'none';

  var activeCount = 0;
  var elements = [];

  data.nodes.forEach(function(n){
    if(n.active) activeCount++;
    elements.push({
      group: 'nodes',
      data: {
        id: n.id,
        color: n.color,
        active: n.active,
        bgColor: n.active ? n.color : '#fff',
        borderColor: n.color,
        borderWidth: n.active ? 0 : 2,
        nodeOpacity: n.active ? 1 : 0.5,
        label: n.id
      },
      position: { x: n.x, y: n.y }
    });
  });

  data.edges.forEach(function(e, i){
    var isOutsideEdge = e.source === 'outside' || e.target === 'outside';
    elements.push({
      group: 'edges',
      data: {
        id: 'e' + i,
        source: e.source,
        target: e.target,
        width: e.width,
        color: e.color,
        isOutsideEdge: isOutsideEdge
      }
    });
  });

  var score = complexityScores[uid] || 0;
  statsEl.innerHTML = '<span>' + activeCount + ' nodes, ' + data.edges.length + ' edges</span><span class="complexity">Score: ' + score + '</span>';

  var instance = cytoscape({
    container: cyDiv,
    elements: elements,
    layout: { name: 'preset' },
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'text-wrap': 'wrap',
          'text-max-width': '80px',
          'font-size': '8px',
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-margin-y': 4,
          'width': 20,
          'height': 20,
          'background-color': 'data(bgColor)',
          'border-color': 'data(borderColor)',
          'border-width': 'data(borderWidth)',
          'opacity': 'data(nodeOpacity)',
          'text-opacity': 1
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 'data(width)',
          'line-color': 'data(color)',
          'curve-style': 'bezier',
          'opacity': 0.6
        }
      }
    ],
    minZoom: 0.3,
    maxZoom: 3,
    userPanningEnabled: false,
    userZoomingEnabled: false,
    boxSelectionEnabled: false,
    autoungrabify: true
  });

  cyInstances[uid] = instance;
}

function refreshAll(){
  USER_IDS.forEach(function(uid){ loadUserGraph(uid); });
}

// ---- Init ----
buildToggles();
buildGrid();
updateAllComplexityScores();
calculateAndSetTertiles();
refreshAll();
updateGroupColors();

// Setup threshold controls
document.getElementById('thresholdLow').addEventListener('change', function(){
  thresholdLow = parseFloat(this.value) || 0;
  updateGroupColors();
});

document.getElementById('thresholdMed').addEventListener('change', function(){
  thresholdMed = parseFloat(this.value) || 0;
  updateGroupColors();
});

document.getElementById('resetBtn').addEventListener('click', function(){
  calculateAndSetTertiles();
  updateGroupColors();
});

// ---- Toggle outside edges visibility ----
document.getElementById('chkShowOutsideEdgesSnapshot').addEventListener('change', function(){
  var showOutside = this.checked;
  Object.keys(cyInstances).forEach(function(uid){
    var cy = cyInstances[uid];
    if(cy){
      cy.elements('edge').forEach(function(edge){
        var isOutside = edge.data('isOutsideEdge');
        if(isOutside){
          if(showOutside){
            edge.show();
          } else {
            edge.hide();
          }
        }
      });
    }
  });
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    df, user_ids = preprocess_data(args.csv_file, args.user_filter)

    # Filter to non-empty Panel_Title for layout
    df_for_layout = df[df["Panel_Title"].notna() & (df["Panel_Title"] != "")]
    master_pos, all_panel_titles, panel_classification = create_master_layout(df_for_layout)

    print(f"\nComputing graph data (with minimal filtering for interactive dashboard)...")
    all_graphs, user_tasks, available_modalities = compute_all_graph_data(
        df, user_ids, master_pos, all_panel_titles, panel_classification,
        min_duration=0,
        min_frequency=1,
        edge_weight_method="total",
        edge_representation="time",
    )

    print(f"\nTotal graph combinations: {len(all_graphs)}")

    # Build HTML
    graph_json = json.dumps(all_graphs)
    user_tasks_json = json.dumps(user_tasks)
    user_ids_json = json.dumps(sorted(user_ids))
    available_modalities_json = json.dumps(available_modalities)

    html = HTML_TEMPLATE
    html = html.replace("{{GRAPH_JSON}}", graph_json)
    html = html.replace("{{USER_TASKS}}", user_tasks_json)
    html = html.replace("{{USER_IDS}}", user_ids_json)
    html = html.replace("{{AVAILABLE_MODALITIES}}", available_modalities_json)

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard written to: {os.path.abspath(out_path)}")

    # ---- Generate snapshot.html ----
    all_task_ids = sorted({t for tasks in user_tasks.values() for t in tasks})
    all_task_ids.insert(0, "combined")  # Add combined as first option
    all_task_ids_json = json.dumps(all_task_ids)

    snapshot = SNAPSHOT_TEMPLATE
    snapshot = snapshot.replace("{{GRAPH_JSON}}", graph_json)
    snapshot = snapshot.replace("{{USER_TASKS}}", user_tasks_json)
    snapshot = snapshot.replace("{{USER_IDS}}", user_ids_json)
    snapshot = snapshot.replace("{{ALL_TASK_IDS}}", all_task_ids_json)
    snapshot = snapshot.replace("{{AVAILABLE_MODALITIES}}", available_modalities_json)

    snapshot_path = os.path.join(os.path.dirname(out_path) or ".", "snapshot.html")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write(snapshot)

    print(f"Snapshot written to:  {os.path.abspath(snapshot_path)}")
    print("Open dashboard.html or snapshot.html in a browser.")


if __name__ == "__main__":
    main()
