import pytest
import numpy as np
import os
import graphzero as gz

@pytest.fixture
def graph1():
    if not os.path.exists("tests/graph1.gl"):
        gz.convert_csv_to_gl("tests/graph1.csv", "tests/graph1.gl")
    return gz.Graph("tests/graph1.gl")

@pytest.fixture
def graph2():
    if not os.path.exists("tests/graph2.gl"):
        gz.convert_csv_to_gl("tests/graph2.csv", "tests/graph2.gl")
    return gz.Graph("tests/graph2.gl")

@pytest.fixture
def graph3():
    if not os.path.exists("tests/graph3.gl"):
        gz.convert_csv_to_gl("tests/graph3.csv", "tests/graph3.gl", directed=True)
    return gz.Graph("tests/graph3.gl")

def test_properties(graph1, graph2):
    assert graph1.num_nodes == 6
    assert graph1.num_edges == 22
    assert graph1.has_weights == False
    assert graph2.has_weights == True

def test_neighbors(graph1):
    g1_neighbors = graph1.get_neighbours(1)
    # 1 -> 0, 2, 5, 3, 4
    assert graph1.get_degree(1) == len(g1_neighbors)
    assert g1_neighbors[0] == 0
    assert g1_neighbors[1] == 2
    assert g1_neighbors[2] == 5
    assert g1_neighbors[3] == 3
    assert g1_neighbors[4] == 4

def test_weights(graph2):
    g2_weights = graph2.get_weights(1)
    # 1 -> 0, 2, 5, 3, 4 ; weights are 0.0001 1.0 1.25 2.1 1.2
    assert g2_weights[0] == 0.0001
    assert g2_weights[1] == 1.0
    assert g2_weights[2] == 1.25
    assert g2_weights[3] == 2.1
    assert g2_weights[4] == 1.2

def test_sampling(graph1):
    assert len(graph1.sample_neighbours(1, K=2)) == 2
    assert len(graph1.sample_neighbours(1, K=1)) == 1
    assert len(graph1.sample_neighbours(5, K=3)) == 3
    assert len(graph1.sample_neighbours(1, K=9)) == 5 # only 5 neighbors, so should return all of them

def test_batch_methods(graph1,graph2):
    startingNodes = [1,3]
    batch_neighbors1 = graph1.batch_random_walk_uniform(startingNodes,3)
    batch_neighbors2 = graph2.batch_random_walk(startingNodes,3)
    batch_neighbors3 = graph2.batch_random_fanout(startingNodes,2) # doesnt include the starting node, so should return 2 neighbors per starting node
    assert batch_neighbors1.shape == (2,4)
    assert batch_neighbors2.shape == (2,4)
    assert batch_neighbors3.shape == (2,2)

def test_weighted_bias_sampling(graph3):
    startingNodes = np.array([1]*10000)
    batch_neighbors = graph3.batch_random_walk(startingNodes, 1)
    assert batch_neighbors.shape == (10000,2)
    batch_neighbors = batch_neighbors.flatten()
    # With the given weights, we expect to see more 2's than 3's in the output
    assert np.sum(batch_neighbors == 2) > np.sum(batch_neighbors == 3)
    assert np.sum(batch_neighbors == 1) == 10000

def test_memory_stability(graph2):
    # If there is a memory leak, this will either crash or take forever.
    nodes = [1] * 1000
    for _ in range(1000):
        _ = graph2.batch_random_walk(nodes, 10)
    assert True 