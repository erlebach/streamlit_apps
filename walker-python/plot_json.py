import pandas as pd

df = pd.read_json("out.json", orient="index", typ='frame')

nodes = df.loc['nodes']
print(nodes)
nodes_df = pd.DataFrame.from_records(nodes)

edges = df.loc['links']
print(edges)
edges_df = pd.DataFrame.from_records(edges)

#print("nodes_df: ", nodes_df)
#print("edges_df: ", edges_df)
