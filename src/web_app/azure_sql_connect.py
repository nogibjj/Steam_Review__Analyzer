import requests
from databricks import sql
import pandas as pd

def start_cluster(cluster_id, databricks_token):
    url = f"https://{databricks_hostname}/api/2.0/clusters/start"
    headers = {"Authorization": f"Bearer {databricks_token}"}
    payload = {"cluster_id": cluster_id}
    response = requests.post(url, headers=headers, json=payload)
    return response.ok

def check_cluster_state(cluster_id, databricks_token):
    url = f"https://{databricks_hostname}/api/2.0/clusters/get"
    headers = {"Authorization": f"Bearer {databricks_token}"}
    payload = {"cluster_id": cluster_id}
    response = requests.get(url, headers=headers, json=payload)
    if response.ok:
        return response.json()["state"]
    return "STOPPED"

def execute_query(query):
    with sql.connect(
        server_hostname=databricks_hostname,
        http_path=databricks_http,
        access_token=databricks_token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
            return df
        

def get_dataframe(query):
    if check_cluster_state(cluster_id, databricks_token) == "RUNNING":
        return execute_query(query)
    else:
        if start_cluster(cluster_id, databricks_token):
            # Polling to check if the cluster is up
            return execute_query(query)