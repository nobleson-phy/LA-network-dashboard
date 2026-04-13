# Network Graph Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structural, node-level, and edge/weight distribution metrics to the interactive Cytoscape.js gaze-activity dashboard.

**Architecture:** Hybrid Python + JS — NetworkX computes heavy metrics (centrality, clustering, diameter) at generation time and embeds them in the JSON; the JS layer computes lightweight filter-sensitive metrics (distribution stats, sparklines) client-side. Two new collapsible sidebar sections are added alongside enhanced tooltips and node detail panels.

**Tech Stack:** Python 3, NetworkX, Cytoscape.js (already present), inline SVG (no new deps)

---

## File Map

| File | Action | What changes |
|---|---|---|
| `generate_dashboard.py` | Modify | Add `compute_graph_metrics()` and `compute_node_metrics()` helper functions; wire them into `compute_all_graph_data()`; update `HTML_TEMPLATE` CSS, HTML, and JS |
| `tests/test_metrics.py` | Create | Unit tests for the two new Python helper functions |

---

### Task 1: Write failing tests for Python metric helpers

**Files:**
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_metrics.py
import sys
import os
import pytest
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_dashboard import compute_graph_metrics, compute_node_metrics


# ── Fixtures ─────────────────────────────────────────────────────────────────

def triangle():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
    return G

def path4():
    """A - B - C - D"""
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    return G

def disconnected():
    """Two components: A-B and C-D"""
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('C', 'D')])
    return G

def empty():
    return nx.Graph()


# ── compute_graph_metrics ────────────────────────────────────────────────────

def test_graph_metrics_triangle_density():
    m = compute_graph_metrics(triangle())
    assert m['density'] == 1.0

def test_graph_metrics_triangle_components():
    m = compute_graph_metrics(triangle())
    assert m['num_components'] == 1

def test_graph_metrics_triangle_diameter():
    m = compute_graph_metrics(triangle())
    assert m['diameter'] == 1

def test_graph_metrics_triangle_avg_degree():
    m = compute_graph_metrics(triangle())
    assert m['avg_degree'] == 2.0

def test_graph_metrics_path_diameter():
    m = compute_graph_metrics(path4())
    assert m['diameter'] == 3

def test_graph_metrics_disconnected_components():
    m = compute_graph_metrics(disconnected())
    assert m['num_components'] == 2

def test_graph_metrics_disconnected_diameter_uses_largest_component():
    # Largest component is A-B or C-D (both size 2, diameter 1)
    m = compute_graph_metrics(disconnected())
    assert m['diameter'] == 1

def test_graph_metrics_empty_returns_zeros():
    m = compute_graph_metrics(empty())
    assert m['density'] == 0.0
    assert m['num_components'] == 0
    assert m['diameter'] == -1
    assert m['avg_clustering'] == 0.0
    assert m['avg_degree'] == 0.0


# ── compute_node_metrics ─────────────────────────────────────────────────────

def test_node_metrics_triangle_keys():
    m = compute_node_metrics(triangle())
    assert set(m.keys()) == {'A', 'B', 'C'}

def test_node_metrics_triangle_degree():
    m = compute_node_metrics(triangle())
    assert m['A']['degree'] == 2
    assert m['B']['degree'] == 2
    assert m['C']['degree'] == 2

def test_node_metrics_triangle_betweenness_zero():
    # In a complete triangle, no node lies on shortest paths between others
    m = compute_node_metrics(triangle())
    assert m['A']['betweenness_centrality'] == 0.0

def test_node_metrics_path_betweenness_ordering():
    # Inner nodes B and C should have higher betweenness than endpoints A and D
    m = compute_node_metrics(path4())
    assert m['B']['betweenness_centrality'] > m['A']['betweenness_centrality']
    assert m['C']['betweenness_centrality'] > m['D']['betweenness_centrality']

def test_node_metrics_all_keys_present():
    m = compute_node_metrics(triangle())
    for node in m.values():
        assert 'degree' in node
        assert 'betweenness_centrality' in node
        assert 'closeness_centrality' in node
        assert 'clustering' in node

def test_node_metrics_empty_returns_empty_dict():
    m = compute_node_metrics(empty())
    assert m == {}
```

- [ ] **Step 2: Run tests — confirm they fail with ImportError**

```bash
cd /Users/kujovi/Desktop/KU/LA/gaze.dashboard
python -m pytest tests/test_metrics.py -v
```

Expected: `ImportError: cannot import name 'compute_graph_metrics' from 'generate_dashboard'`

---

### Task 2: Implement `compute_graph_metrics()` and `compute_node_metrics()`

**Files:**
- Modify: `generate_dashboard.py` — insert after `classify_panel()` (~line 235), before `create_master_layout()`

- [ ] **Step 1: Insert both helper functions into `generate_dashboard.py`**

Find this line (around line 238):
```python
def create_master_layout(df_all):
```

Insert the following block immediately before it:

```python
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

```

- [ ] **Step 2: Run tests — confirm they pass**

```bash
cd /Users/kujovi/Desktop/KU/LA/gaze.dashboard
python -m pytest tests/test_metrics.py -v
```

Expected: all 14 tests PASS

- [ ] **Step 3: Commit**

```bash
git add generate_dashboard.py tests/test_metrics.py
git commit -m "feat: add compute_graph_metrics and compute_node_metrics helpers"
```

---

### Task 3: Wire metric helpers into `compute_all_graph_data()`

**Files:**
- Modify: `generate_dashboard.py` — inside `compute_all_graph_data()`, around line 360–398

- [ ] **Step 1: Replace the node-building block and graph entry creation**

Find this block (lines ~361–398):

```python
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

                key = f"{uid}|{task_id}|{mod}"
                all_graphs[key] = {"nodes": nodes, "edges": edges}
```

Replace with:

```python
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
```

- [ ] **Step 2: Verify the script runs without errors on the sample CSV**

```bash
cd /Users/kujovi/Desktop/KU/LA/gaze.dashboard
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Expected: script completes, prints per-graph stats, produces `/tmp/test_dashboard.html`

- [ ] **Step 3: Confirm `graph_metrics` is in the JSON output**

```bash
python -c "
import json, re
html = open('/tmp/test_dashboard.html').read()
m = re.search(r'var GRAPHS = ({.*?});', html, re.DOTALL)
graphs = json.loads(m.group(1))
first_key = next(iter(graphs))
print('graph_metrics:', graphs[first_key].get('graph_metrics'))
print('first node metrics:', {k: graphs[first_key]['nodes'][0].get(k) for k in ['degree','betweenness_centrality','closeness_centrality','clustering']})
"
```

Expected: prints a dict with `density`, `num_components`, `diameter`, `avg_clustering`, `avg_degree` and node metric values

- [ ] **Step 4: Run tests to confirm nothing broke**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: all 14 PASS

- [ ] **Step 5: Commit**

```bash
git add generate_dashboard.py
git commit -m "feat: embed graph and node metrics in compute_all_graph_data output"
```

---

### Task 4: Add CSS and HTML for new sidebar sections

**Files:**
- Modify: `generate_dashboard.py` — inside `HTML_TEMPLATE`, CSS block (~line 423) and sidebar HTML (~line 499)

- [ ] **Step 1: Add CSS for collapsible sections and distribution display**

Find this line in `HTML_TEMPLATE` (end of style block, ~line 445):
```css
.complexity-high{background:#FFCDD2}
```

Insert immediately after it (before `</style>`):
```css
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
```

- [ ] **Step 2: Add the two new sidebar sections**

Find this block in the sidebar HTML (~line 521):
```html
    <div class="section" id="detailSection">
```

Insert immediately before it:
```html
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

```

- [ ] **Step 3: Verify the HTML renders without JS errors**

```bash
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Open `/tmp/test_dashboard.html` in a browser. Confirm:
- "Graph Metrics (precomputed)" section appears in sidebar with `-` placeholders
- "Edge Distribution (filtered)" section appears with "Load a graph to see distribution."
- No console errors

- [ ] **Step 4: Commit**

```bash
git add generate_dashboard.py
git commit -m "feat: add Graph Metrics and Edge Distribution sidebar sections"
```

---

### Task 5: Add JS helper functions

**Files:**
- Modify: `generate_dashboard.py` — inside `HTML_TEMPLATE`, JS section after state variables (~line 546)

- [ ] **Step 1: Add helper functions after the state variable declarations**

Find this block in the JS section (~line 541):
```javascript
// ---- Complexity calculation ----
function calculateComplexityScore(activeNodes, edgeCount){
```

Insert the following block immediately before it:

```javascript
// ---- Collapsible sections ----
function toggleCollapsible(bodyId, arrowId){
  var body = document.getElementById(bodyId);
  var arrow = document.getElementById(arrowId);
  var isHidden = body.style.display === 'none';
  body.style.display = isHidden ? '' : 'none';
  arrow.classList.toggle('collapsed', !isHidden);
}

// ---- Graph metrics display ----
function renderGraphMetrics(gm){
  if(!gm){ return; }
  document.getElementById('gmDensity').textContent    = gm.density;
  document.getElementById('gmComponents').textContent = gm.num_components;
  document.getElementById('gmDiameter').textContent   = gm.diameter === -1 ? 'N/A' : gm.diameter;
  document.getElementById('gmClustering').textContent = gm.avg_clustering;
  document.getElementById('gmAvgDegree').textContent  = gm.avg_degree;
}

function clearGraphMetrics(){
  ['gmDensity','gmComponents','gmDiameter','gmClustering','gmAvgDegree'].forEach(function(id){
    document.getElementById(id).textContent = '-';
  });
}

// ---- Edge distribution helpers ----
function computeDistribution(values, numBins){
  if(values.length === 0){ return []; }
  var mn = Math.min.apply(null, values);
  var mx = Math.max.apply(null, values);
  var bins = new Array(numBins).fill(0);
  if(mn === mx){ bins[0] = values.length; return bins; }
  values.forEach(function(v){
    var idx = Math.min(Math.floor((v - mn) / (mx - mn) * numBins), numBins - 1);
    bins[idx]++;
  });
  return bins;
}

function renderSparkline(binCounts){
  if(binCounts.length === 0){ return ''; }
  var maxCount = Math.max.apply(null, binCounts);
  var w = 100 / binCounts.length;
  var bars = binCounts.map(function(c, i){
    var h = maxCount > 0 ? (c / maxCount) * 28 : 0;
    var y = 30 - h;
    return '<rect x="' + (i * w + 0.5).toFixed(1) + '" y="' + y.toFixed(1) +
           '" width="' + (w - 1).toFixed(1) + '" height="' + h.toFixed(1) +
           '" fill="#4a90d9" rx="1"/>';
  }).join('');
  return '<svg class="dist-sparkline" viewBox="0 0 100 32" preserveAspectRatio="none">' + bars + '</svg>';
}

function computeSummaryStats(values){
  if(values.length === 0){ return null; }
  var sorted = values.slice().sort(function(a, b){ return a - b; });
  var sum = sorted.reduce(function(a, b){ return a + b; }, 0);
  var mid = Math.floor(sorted.length / 2);
  var median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { min: sorted[0], max: sorted[sorted.length - 1], mean: sum / sorted.length, median: median };
}

function renderDistChannel(label, values, unit){
  var u = unit || '';
  var stats = computeSummaryStats(values);
  if(!stats){
    return '<div class="dist-channel"><div class="dist-channel-label">' + label +
           '</div><span class="dist-empty">No data</span></div>';
  }
  var bins = computeDistribution(values, 8);
  return '<div class="dist-channel">' +
    '<div class="dist-channel-label">' + label + '</div>' +
    '<div class="dist-stats">' +
      '<span class="dist-stat">Min <span>' + stats.min.toFixed(1) + u + '</span></span>' +
      '<span class="dist-stat">Max <span>' + stats.max.toFixed(1) + u + '</span></span>' +
      '<span class="dist-stat">Mean <span>' + stats.mean.toFixed(1) + u + '</span></span>' +
      '<span class="dist-stat">Median <span>' + stats.median.toFixed(1) + u + '</span></span>' +
    '</div>' +
    renderSparkline(bins) +
  '</div>';
}

function updateDistributionSection(edges){
  var container = document.getElementById('edgeDistContent');
  if(!edges || edges.length === 0){
    container.innerHTML = '<span class="dist-empty">No edges visible.</span>';
    return;
  }
  var freqs  = edges.map(function(e){ return e.frequency || 0; });
  var times  = edges.map(function(e){
    return e.time_weight !== undefined ? e.time_weight : (e.timeWeight || 0);
  });
  container.innerHTML = renderDistChannel('Frequency', freqs, '') +
                        renderDistChannel('Time Weight', times, 's');
}

```

- [ ] **Step 2: Verify no JS syntax errors**

```bash
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Open in browser → open DevTools console. Expected: no errors on page load.

- [ ] **Step 3: Commit**

```bash
git add generate_dashboard.py
git commit -m "feat: add JS helpers for graph metrics display and edge distribution"
```

---

### Task 6: Update `loadGraph()` — wire in new data and render new sections

**Files:**
- Modify: `generate_dashboard.py` — `loadGraph()` inside `HTML_TEMPLATE`

- [ ] **Step 1: Clear new sections in the no-data branch**

Find this block inside `loadGraph()` (~line 749):
```javascript
    document.getElementById('statComplexity').textContent = '-';
    document.getElementById('statGroup').textContent = '-';
    document.getElementById('detailContent').textContent = 'No data for this combination.';
    return;
```

Replace with:
```javascript
    document.getElementById('statComplexity').textContent = '-';
    document.getElementById('statGroup').textContent = '-';
    clearGraphMetrics();
    document.getElementById('edgeDistContent').innerHTML = '<span class="dist-empty">No data for this combination.</span>';
    document.getElementById('detailContent').textContent = 'No data for this combination.';
    return;
```

- [ ] **Step 2: Add node metrics to Cytoscape node data**

Find this block inside `loadGraph()` (~line 768):
```javascript
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
```

Replace with:
```javascript
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
        label: n.id,
        node_degree: n.degree || 0,
        node_betweenness: n.betweenness_centrality || 0,
        node_closeness: n.closeness_centrality || 0,
        node_clustering: n.clustering || 0
      },
```

- [ ] **Step 3: Call `renderGraphMetrics` and `updateDistributionSection` at the end of `loadGraph()`**

Find this line near the bottom of `loadGraph()` (~line 1040):
```javascript
  document.getElementById('detailContent').innerHTML = '<span style="color:#999">Click a node or edge for details.</span>';
}
```

Insert immediately before the closing `}`:
```javascript
  renderGraphMetrics(data.graph_metrics);
  updateDistributionSection(data.edges);
```

- [ ] **Step 4: Verify graph metrics and distribution render on graph load**

```bash
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Open in browser. Select a user/task/modality. Confirm:
- "Graph Metrics (precomputed)" section shows real numbers (not `-`)
- "Edge Distribution (filtered)" section shows min/max/mean/median and sparkline bars for both Frequency and Time Weight

- [ ] **Step 5: Commit**

```bash
git add generate_dashboard.py
git commit -m "feat: wire graph metrics and edge distribution into loadGraph"
```

---

### Task 7: Update `applyFilters()` to refresh distribution on filter change

**Files:**
- Modify: `generate_dashboard.py` — `applyFilters()` inside `HTML_TEMPLATE`

- [ ] **Step 1: Add `updateDistributionSection` call at the end of `applyFilters()`**

Find this block at the end of `applyFilters()` (~line 720):
```javascript
  // Apply outside edges visibility
  var showOutside = document.getElementById('chkShowOutsideEdges').checked;
  cy.elements('edge[isOutsideEdge = true]').forEach(function(edge){
    if(!showOutside) edge.hide(); else edge.show();
  });
}
```

Replace with:
```javascript
  // Apply outside edges visibility
  var showOutside = document.getElementById('chkShowOutsideEdges').checked;
  cy.elements('edge[isOutsideEdge = true]').forEach(function(edge){
    if(!showOutside) edge.hide(); else edge.show();
  });

  // Refresh edge distribution for currently visible edges
  updateDistributionSection(filteredEdges);
}
```

- [ ] **Step 2: Verify distribution updates with filter sliders**

```bash
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Open in browser. Load a graph. Move the "Min Frequency" slider up. Confirm:
- Edge Distribution section updates (fewer edges → stats change, sparkline changes)
- Graph Metrics section does NOT change (stays static)

- [ ] **Step 3: Commit**

```bash
git add generate_dashboard.py
git commit -m "feat: refresh edge distribution in applyFilters on slider change"
```

---

### Task 8: Update tooltip and node detail panel event handlers

**Files:**
- Modify: `generate_dashboard.py` — `mouseover node` and `tap node` handlers inside `HTML_TEMPLATE`

- [ ] **Step 1: Enhance the hover tooltip**

Find this block (~line 955):
```javascript
  cy.on('mouseover', 'node', function(evt){
    var d = evt.target.data();
    tooltip.innerHTML = '<b>' + d.id + '</b><br>Class: ' + d.classification +
      '<br>Active: ' + (d.active ? 'Yes' : 'No') +
      '<br>Time: ' + d.time_spent + 's';
    tooltip.style.display = 'block';
  });
```

Replace with:
```javascript
  cy.on('mouseover', 'node', function(evt){
    var d = evt.target.data();
    tooltip.innerHTML = '<b>' + d.id + '</b><br>Class: ' + d.classification +
      '<br>Active: ' + (d.active ? 'Yes' : 'No') +
      '<br>Time: ' + d.time_spent + 's&nbsp;&nbsp;|&nbsp;&nbsp;Degree: ' + (d.node_degree || 0) +
      '<br>Betweenness: ' + (d.node_betweenness || 0).toFixed(3);
    tooltip.style.display = 'block';
  });
```

- [ ] **Step 2: Enhance the node click detail panel**

Find this block (~line 920):
```javascript
    var d = node.data();
    document.getElementById('detailContent').innerHTML =
      '<b>' + d.id + '</b><br>' +
      'Classification: ' + d.classification + '<br>' +
      'Active: ' + (d.active ? 'Yes' : 'No') + '<br>' +
      'Time spent: ' + d.time_spent + 's<br>' +
      'Connected edges: ' + connEdges.length;
```

Replace with:
```javascript
    var d = node.data();
    document.getElementById('detailContent').innerHTML =
      '<b>' + d.id + '</b><br>' +
      'Classification: ' + d.classification + '<br>' +
      'Active: ' + (d.active ? 'Yes' : 'No') + '<br>' +
      'Time spent: ' + d.time_spent + 's<br>' +
      'Connected edges: ' + connEdges.length +
      '<br><br><b>Node Metrics</b><br>' +
      'Degree: ' + (d.node_degree || 0) + '<br>' +
      'Betweenness: ' + (d.node_betweenness || 0).toFixed(3) + '<br>' +
      'Closeness: ' + (d.node_closeness || 0).toFixed(3) + '<br>' +
      'Clustering: ' + (d.node_clustering || 0).toFixed(3);
```

- [ ] **Step 3: Full end-to-end verification**

```bash
python generate_dashboard.py all.users.csv -o /tmp/test_dashboard.html
```

Open in browser. Verify all of the following:

1. **Graph Metrics section** — shows density, components, diameter, avg clustering, avg degree after selecting a graph
2. **Edge Distribution section** — shows frequency and time weight stats + sparklines; updates when sliders move
3. **Collapsible toggle** — clicking section headers collapses/expands them; arrow rotates
4. **Node tooltip** — hover over an active node shows degree and betweenness
5. **Node detail panel** — click a node shows full Node Metrics block (degree, betweenness, closeness, clustering)
6. **Inactive nodes** — hovering shows degree 0 / betweenness 0.000
7. **Empty filter state** — slide Min Frequency to maximum so no edges pass → distribution shows "No edges visible."

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: all 14 PASS

- [ ] **Step 5: Final commit**

```bash
git add generate_dashboard.py
git commit -m "feat: enhance node tooltip and detail panel with per-node metrics"
```
