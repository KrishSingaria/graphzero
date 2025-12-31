from typing import List, Tuple, Any, Union
import numpy as np
import numpy.typing as npt

class Graph:
    """
    Graph Class contains the graphfile and its relevant functions and methods.
    It holds the mmap / zero-copy memory.
    """
    
    num_nodes: int
    num_edges: int

    def __init__(self, filename: str) -> None:
        """
        Args:
            filename (str): either absolute path or relative path (depends on the current working directory).
        Returns:
            Graph class instance.
        """
        ...

    def get_degree(self, node_id: int) -> int:
        """
        Get the degree of a node.
        
        Args:
            node_id (int)
        Returns:
            degree (int)
        """
        ...

    def get_neighbours(self, node_id: int) -> npt.NDArray[np.uint64]:
        """
        Returns the neighbours of a node.
        
        Args:
            node_id (int)
        Returns:
            1-D numpy ndarray of neighbour node IDs (dtype: platform-size integer).
        """
        ...

    def batch_random_walk(self, start_nodes: List[int], walk_length: int, p: float = 1.0, q: float = 1.0) -> npt.NDArray[np.int64]:
        """
        Performs 2nd-order random walks (Node2Vec style).

        Args:
            start_nodes (list): list of starting nodes.
            walk_length (int): how long walks should be (e.g. 10).
            p (float): Return parameter; Low = keeps walk local (BFS-like).
            q (float): In-out parameter; Low = explores far away (DFS-like).

        Returns:
            ndarray of shape (len(start_nodes), walk_length) with dtype np.int64.
        """
        ...

    def batch_random_walk_uniform(self, start_nodes: List[int], walk_length: int) -> npt.NDArray[np.int64]:
        """
        Performs uniform random walks.

        Args:
            start_nodes (list): list of starting nodes.
            walk_length (int): how long walks should be (e.g. 10).

        Returns:
            ndarray of shape (len(start_nodes), walk_length) with dtype np.int64.
        """
        ...

    def batch_random_fanout(self, start_nodes: List[int], K: int) -> npt.NDArray[np.int64]:
        """
        Performs uniform random fanout sampling.

        Args:
            start_nodes (list): list of starting nodes.
            K (int): how many neighbours to sample.

        Returns:
            ndarray of shape (len(start_nodes), K) with dtype np.int64.
        """
        ...

    def sample_neighbours(self, start_node: int, K: int) -> npt.NDArray[np.int64]:
        """
        Performs uniform random neighbour sampling for a node.

        Args:
            start_node (int): node id to sample from.
            K (int): how many neighbours to sample.

        Returns:
            1-D ndarray with up to K neighbour ids (dtype np.int64).
        """
        ...

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...


def convert_csv_to_gl(csv_path: str, out_path: str, directed: bool = False) -> None:
    """
    Convert a CSV edge list to GraphZero binary format (.gl)

    Args:
        csv_path (str): Path to input CSV.
        out_path (str): Path to output .gl file.
        directed (bool): Whether the graph is directed. Defaults to False.
    """
    ...