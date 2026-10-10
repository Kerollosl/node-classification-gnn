import numpy as np

def node_rows(ids):
    ids=list(ids)
    if len(set(ids))!=len(ids):raise ValueError("Duplicate node IDs")
    return {node:i for i,node in enumerate(ids)}

def binary_labels(scores,threshold=.5):
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or not 0<=threshold<=1:raise ValueError("Invalid probability or threshold")
    return (scores>=threshold).astype(int)

def mean_embedding(words,keyed_vectors,dimensions=100):
    vectors=[keyed_vectors[word] for word in words if word in keyed_vectors]
    return np.mean(vectors,axis=0) if vectors else np.zeros(dimensions,dtype=np.float32)
