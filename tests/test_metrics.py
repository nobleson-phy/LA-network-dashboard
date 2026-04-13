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
