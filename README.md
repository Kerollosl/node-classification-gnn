# Node Classification with GraphSAGE 🔗

## 📋 Overview

A GraphSAGE graph neural network that classifies nodes in a graph by combining **Node2Vec** structural embeddings with **Word2Vec** textual embeddings. Fusing both feature types lets the model exploit *how* a node is connected (graph topology) and *what* it's about (text content) simultaneously.

> **Note:** The specific details of the original university assignment (dataset domain, target categories) have been removed to prevent plagiarism. The project stands on its general relevance as a node-classification reference implementation.

## 🧠 Approach

- **Architecture:** Multiple stacked `SAGEConv` layers from DGL
- **Node features:** Concatenation of Node2Vec embeddings (graph structure) and Word2Vec embeddings (node titles / text)
- **Regularization:** Large embedding dimensions paired with a high dropout rate — trades capacity for generalization so the model can surface nuanced feature interactions without overfitting
- **Training choice:** Multiple SAGEConv layers prevent train loss from collapsing too quickly relative to validation accuracy
- **Visualization:** Built-in function plots embedding trajectories and classification progress across epochs

## 📊 Dataset

- `network.txt` — edge list defining the graph
- `titles.txt` — textual feature for each node (fed through Word2Vec)
- `categories.txt` — class labels
- `train.txt` / `val.txt` / `test.txt` — node ID splits

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas networkx numpy gensim node2vec torch dgl scikit-learn matplotlib
```

### Run
Place `network.txt`, `categories.txt`, `titles.txt`, `train.txt`, `val.txt`, and `test.txt` alongside `classifier.py`, then:

```bash
python classifier.py
# or
./classifier.py
```

## 📁 Repository Contents

- `classifier.py` — main training + evaluation pipeline (GraphSAGE model, embedding generation, training loop, visualization)
- `Node Classification.ipynb` — notebook version of the full pipeline
- `Facebook_Graph.ipynb` — companion notebook applying the same approach to a Facebook graph dataset
- `network.txt` — graph edge list
- `titles.txt` — per-node textual features
- `categories.txt` — class labels
- `train.txt` / `val.txt` / `test.txt` — train/val/test node splits
- `test_nodes_simple.txt` / `test_nodes_detailed.txt` — prediction outputs
