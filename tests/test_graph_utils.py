import unittest,numpy as np
from graph_utils import node_rows,binary_labels,mean_embedding
class GraphTests(unittest.TestCase):
    def test_ids(self):self.assertEqual(node_rows([99,2,40]),{99:0,2:1,40:2})
    def test_threshold(self):self.assertEqual(binary_labels([.1,.5,.8]).tolist(),[0,1,1])
    def test_full_vector(self):np.testing.assert_equal(mean_embedding(['a','b'],{'a':np.array([1,2]),'b':np.array([3,4])}),[2,3])
