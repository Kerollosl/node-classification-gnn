### Note the specific details of this university assignment have been removed to prevent against threats of plagiarism
### The practical relevance of the project stand as node classification GraphSAGE graph neural network using Node2Vec and Word2Vec embedding features
Step 1: Make sure all necessary packages have been installed:
	os
	pandas
	networkx
	numpy
	gensim
	node2vec
	torch
	dgl
	sklearn
	matplotlib

Step 2: Ensure the following files are in the same directory as classifier.py:
	network.txt
	categories.txt
	test.txt
	titles.txt
	train.txt
	val.txt

Step 3: Navigate to the directory containing classifier.py in the terminal 
	and run "$ python classifier.py" or "./classifier.py"
 
============================================================================================

Methodology: node2Vec and word2vec embeddings were used to benefit from both types of vectors. Data was organized based on pandas dataframe indices to avoid mislabeling. Large 
number of embeddings used with large dropout rate for optimal opportunity to discover 
trends in features while also avoiding overfitting. Multiple SAGEConv layers used to 
avoid train loss from converging too quickly without good test accuracy. Visualization function used to see progress of embeddings and classification
