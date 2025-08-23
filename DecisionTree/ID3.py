import json
from entropy import gain
from pandas import DataFrame
import matplotlib.pyplot as plt
import networkx as nx
import random

def findBestFeature(df, target_label):
    labels = [col for col in df.columns if col != target_label]
    targets = df[target_label].values
    id3_dict = {}
    for label in labels:
        gain_val = gain(df, label, target_label, targets)
        id3_dict[label] = gain_val
    max_gain_key, max_gain_value = max(id3_dict.items(), key=lambda item: item[1])
    return max_gain_key, max_gain_value

def plot_tree(tree, parent=None, graph=None, edge_label=''):
    if graph is None:
        graph = nx.DiGraph()
    if isinstance(tree, dict):
        for node, branches in tree.items():
            if parent is not None:
                graph.add_edge(parent, node, label=edge_label)
            for branch, subtree in branches.items():
                plot_tree(subtree, node, graph, str(branch))
    else:
        graph.add_edge(parent, tree, label=edge_label)
        
    if parent is None:
        G = graph
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")  
        except Exception:
            pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
    return graph

def getAllGain(data: DataFrame, cate: str):
    labels=[]
    for idx in data.keys():
        labels.append(idx)
    cates = data.get(cate)
    gain_dict = {}
    for label in labels:
        gain_val = gain(data, label, cate, cates)
        gain_dict.update({label: gain_val})
    return gain_dict

def classify(tree: dict, labels: dict, default=None):
    node = next(iter(tree.keys()))
    branch = labels[node]
    if branch not in tree[node]:
        return default
    sub_node = tree[node][branch]
    if isinstance(sub_node, dict):
        return classify(sub_node, labels)
    else:
        return sub_node

def createDecisionTree(data: DataFrame, target_label):
    if len(set(data[target_label])) == 1:
        return data[target_label].iloc[0]
    if len(data.columns) == 1:
        return data[target_label].mode()[0]
    bestFeature, _ = findBestFeature(data, target_label)
    tree = {bestFeature: {}}
    for cate in set(data[bestFeature]):
        sub_data = data[data[bestFeature] == cate].drop(columns=[bestFeature])
        tree[bestFeature][cate] = createDecisionTree(sub_data, target_label)
    return tree

if __name__ == "__main__":
    data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal"],
    "Windy": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak"],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes"]
}
    df = DataFrame(data)
    print(df)
    print(getAllGain(df, "Play"))
    print(findBestFeature(df, "Play"))
    tree = createDecisionTree(df, "Play")
    print(json.dumps(tree, indent=4, ensure_ascii=False))
    plot_tree(tree)
    labels = {"Outlook": "Sunny", "Temperature": "Cool"}
    isPlay = classify(tree, labels)
    print(isPlay)