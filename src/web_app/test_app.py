from app import create_plot, clustering_wordclouds_graphs, clustering_trigrams_graphs
import pandas as pd

def test_graph1():
    df = pd.read_parquet("src/data/No_Man's_Sky_clean.parquet")
    assert create_plot(df, "Month", "2023-01-01", "2023-09-12")

def test_graph2():
    df = pd.read_parquet("src/data/No_Man's_Sky_clean.parquet")
    assert clustering_wordclouds_graphs(df)

def test_graph3():
    df = pd.read_parquet("src/data/No_Man's_Sky_clean.parquet")
    assert clustering_trigrams_graphs(df)