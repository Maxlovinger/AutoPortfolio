"""
network_model.py — (C) Network / graph model over the stock universe.

We build a graph where nodes are stocks and edges connect stocks whose returns
are strongly correlated. Then we measure each stock's EIGENVECTOR CENTRALITY:

  * HIGH centrality  = moves with the crowd (redundant, offers little diversification)
  * LOW  centrality  = marches to its own drum (a diversifier / under-followed)

For portfolio construction we REWARD low centrality, because those names improve
diversification and are less arbitraged. This is rarely done at retail level.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

from utils import zscore


def build_correlation_graph(prices: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    rets = np.log(prices / prices.shift(1)).dropna()
    corr = rets.corr()
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                # weight = correlation strength
                G.add_edge(cols[i], cols[j], weight=abs(c))
    return G


def network_scores(prices: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Return a z-scored 'diversifier' signal: LOW centrality -> HIGH score.
    """
    G = build_correlation_graph(prices, threshold)
    # A graph with no edges has no meaningful centrality — eigenvector
    # centrality on a zero-adjacency matrix returns arbitrary values, so we
    # must short-circuit to a neutral (all-zero) signal here.
    if G.number_of_edges() == 0:
        return pd.Series(0.0, index=prices.columns)
    try:
        cent = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        cent = nx.degree_centrality(G)
    cent = pd.Series(cent).reindex(prices.columns).fillna(0.0)
    # reward LOW centrality -> negate
    return zscore(-cent)


def community_labels(prices: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Bonus: greedy-modularity communities. Useful to avoid concentrating the
    final portfolio inside one correlated cluster.
    """
    G = build_correlation_graph(prices, threshold)
    try:
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        return {node: i for i, c in enumerate(comms) for node in c}
    except Exception:
        return {n: 0 for n in G.nodes}
