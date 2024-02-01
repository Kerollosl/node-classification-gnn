# -*- coding: utf-8 -*-
"""Node_Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1edafYJxRn_3Viw2Xe4ckWROdgtgb9QAA

#Import Packages
"""

# Commented out IPython magic to ensure Python compatibility.
# Install required packages.
import os
import pandas as pd
import networkx as nx
import numpy as np

from gensim.models import Word2Vec
from node2vec import Node2Vec

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from sklearn.feature_extraction.text import CountVectorizer

# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Visualization Function
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

"""# Create Graph from DataFrames"""

path = "./"

#Parse txt files into dataframes
def create_df(file):
  data = []
  with open(path + file, "r") as file:
      for line in file:
          parts = line.strip().split(' ', 1)
          data.append(parts)
  return pd.DataFrame(data, columns=['Column1', 'Column2'])



titles_df = create_df('titles.txt')#categories_df = create_df('categories.txt')
network_df = create_df('network.txt').astype(int)
test_nodes = pd.read_csv(path+'test.txt', header=None, squeeze=True).tolist()
categories_df = create_df('categories.txt')
train_df = create_df('train.txt')
val_df = create_df('val.txt')

G = nx.from_pandas_edgelist(network_df, source='Column1', target='Column2')

#Dictionaries to translate txt files
titles_dict = dict(zip(titles_df['Column1'].astype(int), titles_df['Column2']))
category_train_dict = dict(zip(train_df['Column1'].astype(int), train_df['Column2'].astype(int)))
category_convert_dict = dict(zip(categories_df['Column1'].astype(int), categories_df['Column2']))

category_name_node_dict = {node_id: category_convert_dict[category_num] for node_id, category_num in category_train_dict.items()}

#Set node attributes in graph, G
nx.set_node_attributes(G, titles_dict, name='title')
nx.set_node_attributes(G, category_train_dict, name='category_number')
nx.set_node_attributes(G, category_name_node_dict, name='category') #Assign Category names based on specific node


nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').sort_index()

"""#Word2Vec Embeddings"""

def word2vec_embeddings():
  titles = list(nodes_df['title'])
  node_index = list(nodes_df.index)

  # All titles into one string
  titles_as_strings = [" ".join(title.split()) for title in titles]

  # Tokenize titles string
  tokenized_titles = [title.split() for title in titles_as_strings]

  model = Word2Vec(sentences=tokenized_titles, vector_size=100, window=10, min_count=1, sg=1)

  sentence_embeddings = []
  for title in tokenized_titles:
    word_embeddings = []
    for word in title:
      embedding = model.wv[word][0]
      word_embeddings.append(embedding)
    sentence_embeddings.append([np.mean(word_embeddings)])
  return sentence_embeddings

"""#Node2Vec Embeddings and Embeddings as a Tensor"""

def node2vec_embeddings():
  node_index = list(nodes_df.index)
  # Use the Node2Vec algorithm to generate embeddings
  node2vec = Node2Vec(G, dimensions=128, walk_length=25, num_walks=100, workers=4)

  # Embed nodes
  model = node2vec.fit(window=10, min_count=1)

  # Access the embeddings for nodes
  node_embeddings = [model.wv[node] for node in node_index]
  return node_embeddings

def make_tensor(embedding_list):
  list_of_tensors = [tensor(embedding) for embedding in embedding_list]
  tensor_of_tensors = torch.stack(list_of_tensors, dim=0)
  return tensor_of_tensors

def make_double_tensor(embedding_list_1, embedding_list_2):
    tensor_1 = tensor(embedding_list_1)
    tensor_2 = tensor(embedding_list_2)
    combined_tensor = torch.cat((tensor_1, tensor_2), dim=1)
    return combined_tensor


#Choose Desired Embeddings and Make a tensor out of them
double_tensor = make_double_tensor(node2vec_embeddings(), word2vec_embeddings())
#node2vec_tensor = make_tensor(node2vec_embeddings())
#word2vec_tensor = make_tensor(word2vec_embeddings())

"""# GraphSAGE Model"""

#GraphSAGE Model
class GraphSAGENet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers):
        super(GraphSAGENet, self).__init__()
        self.layers = nn.ModuleList([
            dglnn.SAGEConv(in_feats, hidden_feats, 'mean')
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.35)
        self.prediction = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph, x):
        for layer in self.layers:
            x = F.relu(layer(graph, x))
            x = self.dropout(x)
        x = self.prediction(x)
        return x

#Hyper Parameters
node_features = double_tensor
in_feats = node_features.shape[1]
hidden_feats = in_feats
num_layers = 4
out_feats = 26

#Create Model
model = GraphSAGENet(in_feats,hidden_feats, out_feats, num_layers)
model.eval()
dgl_graph = dgl.from_networkx(G)
out = model(dgl_graph,node_features)

#Visualize Embeddings and Classes
labels = tensor(nodes_df['category_number'].dropna().tolist(), dtype=torch.long)
labeled_nodes = nodes_df['category_number'].dropna().index.tolist()
mask = tensor(~nodes_df['category_number'].isna())

visualize(out[mask], color=labels)
model = GraphSAGENet(in_feats,hidden_feats, out_feats, num_layers)

"""
# Train Model"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()


    out = model(dgl_graph, node_features)
    loss = criterion(out[mask], labels)
    loss.backward()
    optimizer.step()
    return loss


for epoch in range(1, 600):
    loss = train()
    if epoch %10 ==0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

def validation():
      val_nodes = val_df['Column1'].astype(int).to_list()
      val_labels = val_df['Column2'].astype(int).to_list()

      model.eval()
      out = model(dgl_graph,node_features)
      pred = out.argmax(dim=1)  # Class with highest probability.

      val_correct = pred[val_nodes].numpy() == val_labels  # True if Pred Label is Correct
      val_acc = val_correct.sum() / len(val_nodes)
      #for prediction, actual in zip(pred[val_nodes].numpy(), val_labels):
       # print(f'Val Prediction: {prediction}, Actual: {actual} ')
      return val_acc
val_acc = validation()
print(f'Test Accuracy: {val_acc}')

out = model(dgl_graph,node_features)
node_ids = [node for node in G.nodes]
predicted_classes = out.argmax(dim=1)

#visualize(out[mask], color=labels)

for node_id in sorted(test_nodes):
    print(f"{node_id} {predicted_classes[node_id].item()}")
    #print(f"Node: {node_id}, Predicted Class #: {predicted_classes[node_id].item()}, Class Name: {category_convert_dict[predicted_classes[node_id].item()]}")