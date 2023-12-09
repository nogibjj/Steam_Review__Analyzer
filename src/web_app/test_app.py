from azure_sql_connect import get_dataframe
from app import create_plot, clustering_wordclouds_graphs



def test_query():
    the_query = "SELECT * FROM default.final_steam_table WHERE review LIKE '%fun%' LIMIT 1"
    assert len(get_dataframe(the_query)) > 0

def test_graph1():
    the_query = "SELECT * FROM default.final_steam_table WHERE game_name = 'Rust'"
    df = get_dataframe(the_query)
    assert create_plot(df, "Month", "2016-01-01", "2023-09-12")

def test_graph2():
    the_query = "SELECT * FROM default.final_steam_table WHERE review LIKE '%fun%' LIMIT 10"
    df = get_dataframe(the_query)
    assert clustering_wordclouds_graphs(df)