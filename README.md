# LA Network Dashboard

Interactive network graph dashboard for gaze activity data. Visualises eye-tracking (eTrack) and mouse-click (mclick) transitions between dashboard panels as weighted networks, with structural, node-level, and edge distribution metrics.

## What it does

- Reads a gaze activity CSV and builds one network graph per **user × task × modality** combination
- Renders graphs interactively in the browser via [Cytoscape.js](https://js.cytoscape.org/)
- Precomputes graph and node metrics using [NetworkX](https://networkx.org/) at generation time
- Provides a snapshot overview page (all users, grid layout) and a single-user interactive dashboard

## Quickstart

```bash
pip install pandas numpy networkx
python generate_dashboard.py your_data.csv -o dashboard.html
open dashboard.html
```

The script also writes `snapshot.html` alongside the dashboard.

## Usage

```
python generate_dashboard.py <csv_file> [options]

Options:
  -u, --user-filter       Filter users by ID prefix (default: "6")
  -d, --min-duration      Minimum node duration in seconds (default: 0)
  -f, --min-frequency     Minimum edge frequency (default: 2)
  -e, --edge-weight       Edge weight method: total | average (default: total)
  -r, --edge-representation  Edge display: time | frequency (default: time)
  -o, --output            Output HTML file (default: dashboard.html)
```

## Dashboard features

### Interactive filters (top bar)
- **Min Duration** — hide nodes with less than N seconds of dwell time
- **Min Frequency** — hide edges traversed fewer than N times
- **Edge Weight** — toggle between total and average time weight
- **Edge Representation** — toggle between time-based and frequency-based edge widths
- **Show edges to "outside"** — toggle edges to the off-panel region

### Sidebar panels

| Section | Updates on filter change? | Source |
|---|---|---|
| Statistics | Yes | Client-side |
| Graph Metrics (precomputed) | No — fixed at generation time | Python / NetworkX |
| Edge Distribution (filtered) | Yes — reflects visible edges | Client-side |
| Details | On click | Client-side |

### Graph Metrics
Computed by NetworkX on the active subgraph at generation time:

| Metric | Description |
|---|---|
| Density | Ratio of actual to possible edges |
| Components | Number of connected components |
| Diameter | Longest shortest path (largest component) |
| Avg Clustering | Mean local clustering coefficient |
| Avg Degree | Mean node degree |

### Node metrics
Shown on hover (degree + betweenness) and on click (full panel):

| Metric | Description |
|---|---|
| Degree | Number of connected edges |
| Betweenness | Fraction of shortest paths through node |
| Closeness | Inverse mean distance to all other nodes |
| Clustering | Local clustering coefficient |

### Edge Distribution
Frequency and time-weight distributions for currently visible edges — min, max, mean, median plus an inline SVG sparkline.

## Node colours

| Colour | Type |
|---|---|
| Green | Picture / chart panel |
| Purple | Text / report panel |
| Orange | Instruction panel |
| Grey | Outside (off-panel) |

Hollow nodes with a coloured border = panel visited but below the current duration threshold.

## Project structure

```
generate_dashboard.py       Main script
tests/
  test_metrics.py           Unit tests for graph/node metric helpers
docs/superpowers/
  specs/                    Design specification
  plans/                    Implementation plan
```

## Running tests

```bash
pip install pytest networkx
pytest tests/ -v
```

## Data format

The script expects a CSV with a skipped header row and these columns (in order):

`User_ID, Timestamp, X, Y, Panel_Title, Task_ID, Activity_ID, Screen_ID, Activity, Task_Type, Element_ID, Element_Type, Verb, CourseID, Duration, Attempted`

Modality is inferred: rows with no X coordinate are `mclick`; rows with an X coordinate are `eTrack`.
