# Network Graph Metrics — Design Spec
**Date:** 2026-04-13
**File:** `generate_dashboard.py`
**Status:** Approved

---

## Overview

Add structural, node-level, and edge/weight distribution metrics to the existing interactive Cytoscape.js gaze-activity dashboard. The goal is to serve both research/comparison workflows and deep per-graph exploration equally well.

---

## Architecture: Hybrid Python + JS

Heavy, filter-insensitive metrics are precomputed in Python using NetworkX and embedded in the graph JSON. Lightweight, filter-sensitive metrics are computed client-side in JavaScript and update whenever the user moves the duration/frequency sliders.

---

## Python-Side Changes

### Location
`compute_all_graph_data()` in `generate_dashboard.py`, after the node/edge lists are built for each `uid|taskId|modality` graph.

### Graph-level metrics (stored as `graph_metrics` key on each graph entry)

Computed on the active subgraph (nodes that pass `min_duration`, edges that pass `min_frequency` at generation time — i.e., the default filtered state).

| Metric | NetworkX call | Notes |
|---|---|---|
| `density` | `nx.density(G)` | Ratio of actual to possible edges |
| `num_components` | `nx.number_connected_components(G)` | Integer |
| `diameter` | `nx.diameter(G.subgraph(largest_component))` | On largest component; `-1` if graph has 0 or 1 node |
| `avg_clustering` | `nx.average_clustering(G)` | Float, 0–1 |
| `avg_degree` | mean of degree sequence | Float |

### Node-level metrics (added to each node dict in `nodes` list)

Computed on the same active subgraph. Isolated nodes (not in the subgraph) get `0.0` for all metrics.

| Metric | NetworkX call |
|---|---|
| `degree` | `G.degree(node)` |
| `betweenness_centrality` | `nx.betweenness_centrality(G)[node]` |
| `closeness_centrality` | `nx.closeness_centrality(G)[node]` |
| `clustering` | `nx.clustering(G)[node]` |

### Edge data
No new Python work. `frequency` and `time_weight` per edge are already in the JSON and sufficient for client-side distribution computation.

---

## Sidebar UI Changes

### New sidebar structure

```
Legend
Statistics          ← unchanged
Graph Metrics       ← NEW collapsible section (▶/▼ toggle)
Edge/Weight Dist.   ← NEW collapsible section (▶/▼ toggle)
Details             ← expanded with node metrics on click
```

### Graph Metrics section

Labelled **"Graph Metrics (full graph)"** to communicate that these values are not affected by the interactive filters.

```
Density          0.42
Components       1
Diameter         4
Avg Clustering   0.31
Avg Degree       3.2
```

### Edge/Weight Distribution section

Computed client-side from currently visible edges. Updates on every filter change.

```
── Frequency ────────────────
  Min  1   Max  12   Mean  4.3   Median  3
  [▁▂▄█▆▃▂▁]   ← 8-bin SVG sparkline

── Time Weight ──────────────
  Min  0.4s   Max  18s   Mean  6.1s   Median  4.8s
  [▂▄▅█▆▄▂▁]   ← 8-bin SVG sparkline
```

Sparklines are inline SVG bar charts (8–10 equal-width bins, no external library).

---

## Node Tooltip (hover)

Enhanced from current (class + active + time) to add degree and betweenness:

```
Panel Name
Class: picture  |  Active: Yes
Time: 12.4s  |  Degree: 4
Betweenness: 0.18
```

---

## Node Detail Panel (click)

Expands the existing Details section with a full node metrics table below the existing fields:

```
Panel Name
─────────────────────
Classification:    picture
Active:            Yes
Time spent:        12.4s
Connected edges:   4

── Node Metrics ─────────────
Degree:            4
Betweenness:       0.182
Closeness:         0.561
Clustering:        0.333
```

Edge click behaviour is unchanged.

---

## JavaScript Changes

### `loadGraph()`
- Read `data.graph_metrics` → populate Graph Metrics collapsible section
- Store node metrics (`degree`, `betweenness_centrality`, `closeness_centrality`, `clustering`) in Cytoscape node data at load time
- Call `updateDistributionSection(data.edges)` to render initial distribution

### `applyFilters()`
- After filtering, call `updateDistributionSection(visibleEdges)` with the currently visible edge set
- Graph Metrics section is not updated (static, full-graph values)

### Event handlers
- `mouseover node` → add `Degree` and `Betweenness` to tooltip
- `tap node` → render full node metrics table in Details panel
- `tap edge` → unchanged

### New helper functions

| Function | Purpose |
|---|---|
| `computeDistribution(values, numBins)` | Returns array of bin counts for sparkline |
| `renderSparkline(binCounts)` | Returns SVG string (inline bar chart) |
| `updateDistributionSection(edges)` | Computes stats + sparklines for frequency and time_weight channels, renders into sidebar |
| `renderGraphMetrics(graphMetrics)` | Populates Graph Metrics collapsible section |

---

## Collapsible Section Behaviour

Both new sections (Graph Metrics, Edge/Weight Distribution) start **expanded** by default. A click on the section header toggles collapsed/expanded state. State is not persisted across graph switches.

---

## What Is NOT Changing

- Snapshot page (`SNAPSHOT_TEMPLATE`) — no metrics added there
- Existing Statistics section rows
- Edge click detail panel
- Complexity score / group classification logic
- Layout, filter controls, legend

---

## Labelling Convention

To avoid confusion between full-graph and filtered metrics:

- Graph Metrics section header reads: **"Graph Metrics (precomputed)"** — communicates that these values reflect the graph at generation time (default CLI thresholds) and do not update with interactive filters
- Edge/Weight Distribution section header reads: **"Edge Distribution (filtered)"**
- Node metrics in tooltip/detail panel carry no qualifier (they are precomputed at generation time)

### Empty state handling

When `updateDistributionSection` receives an empty edge list (all edges filtered out), the section shows: `No edges visible` and hides the sparklines. Summary stat fields display `—`.
