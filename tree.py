# from functools import cached_property

from datetime import datetime
from queue import SimpleQueue
import pickle

import numpy as np
import cupy as cp
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from solver import ( WordleSolver )

class WordleSolutionTree(object):
    # Augmentation of WordleSolver that constructs a solution tree for a Wordle problem.
    # Our guesses aren't necessarily unique across all branches, so the networkx nodes
    # are keyed to integer ids to make sure we actually get a tree.

    def __init__(self, solver: WordleSolver, verbose: bool=False):
        self.solver: WordleSolver = solver
        self.verbose = verbose
        # Cache
        self._responses_cache: np.ndarray = None

        # Global state
        self.tree: nx.DiGraph = None
        self.queue: SimpleQueue = None
        self.time_start: datetime = None
        # Iterative state
        self._current_node_id: int = None

        # Initialize / reset state
        self.init()

    @property
    def candidates(self):
        return self.solver.candidates
    @property
    def pool(self):
        return self.solver.pool
    @property
    def responses(self):
        if self._responses_cache is None:
            self._responses_cache = self.solver.compute_responses(
                                            self.solver.pool, self.solver.candidates)
        return self._responses_cache
    @property
    def xp(self):
        return cp if self.solver.has_gpu else np
    @property
    def elapsed(self):
        if self.time_start is None:
            return 0
        else:
            return datetime.now() - self.time_start

    def init(self):
        # Global state
        self.tree: nx.DiGraph = nx.DiGraph()
        self.queue = SimpleQueue() # for BFT mode
        self.time_start = datetime.now()
        # Iterative state
        self._current_node_id = 0


    def serialize(self, pickle_path: str):
        pickle.dump(self.tree, open(pickle_path, 'wb'))

    def deserialize(self, pickle_path: str):
        self.tree = pickle.load(open(pickle_path, 'rb'))


    def _build_tree_inner(self, parent_id: int, key: int, live_cidxs: np.ndarray, recursive_mode=False) -> None:

        xp = self.xp

        G = self.tree
        pool = self.solver.pool
        # Partition candidates and responses.
        candidates = self.solver.candidates[live_cidxs]
        R = self.responses[:,live_cidxs] # (p, c)

       # Make guess
        guess_idx = self.solver.get_best_guess(candidates, pool, R, live_cidxs) # scalar
        R_row = R[guess_idx] # (c, )
        guess_raw = self.solver.idx2codewords(guess_idx)[0]

        # Create node
        node_id = self._current_node_id
        G.add_node(node_id,
            guess_raw = guess_raw,
            # guess_idx = guess_idx,
            # live_cidxs = live_cidxs
        )
        self._current_node_id += 1

        # Join to parent
        if parent_id is not None:
            G.add_edge(parent_id, node_id,
                key = key,
                vec = self.solver.keys[key]
            )

        # Print
        if self.verbose and recursive_mode:
            print("Guess {}: {}. Elapsed: {}".format(
                node_id, guess_raw, self.elapsed
            ))

        # Create children
        n_keys = xp.shape(self.solver.keys)[0]
        c, _ = xp.shape(candidates) # Recompute c
        for key in range(n_keys):
            # The key is an integer encoding its corresponding response vector.
            idxs = xp.arange(c)[R_row == key]
            new_live_cidxs = live_cidxs[idxs] # This is an array copy from fancy indexing
            size = xp.size(new_live_cidxs)
            if size == 0 or key == n_keys-1:
                # if size == 0, empty partition --> ignore
                # if key == n_keys-1, it means full match, so it's a leaf node.
                continue
            # print(f"guess {guess_idx} {guess_raw}")
            # print(f"Response: {self.solver.keys[key]}")
            # print(self.solver.idx2codewords(new_live_cidxs))
            if recursive_mode:
                self._build_tree_inner(node_id, key, new_live_cidxs, recursive_mode=True)
            else:
                self.queue.put((node_id, key, new_live_cidxs))
        
        # Print
        if self.verbose and not recursive_mode:
            print("Guess {}: {}. n_unexplored_nodes: {}. Elapsed: {}".format(
                node_id, guess_raw, self.queue.qsize(), self.elapsed
            ))


    def build_tree(self, mode: str='bft') -> None:
        # mode: 'bft' or 'dft'. Default 'bft'. Depth-first traversal or breadth-first.

        if self.verbose:
            print("Building tree. Wordlist: {}. Solver: {}".format(
                self.solver._wordlist,
                self.solver.__class__.__name__
            ))

        xp = self.xp
        self.init() # Reset state

        c, _ = xp.shape(self.solver.candidates)
        live_cidxs = xp.arange(c)

        if mode == 'bft':
            self.queue.put((None, None, live_cidxs))
            while not self.queue.empty():
                parent_id, key, live_cidxs = self.queue.get()
                self._build_tree_inner(parent_id, key, live_cidxs, recursive_mode=False)

        elif mode == 'dft':
            self._build_tree_inner(None, None, live_cidxs, recursive_mode=True)

        else:
            raise ValueError(mode)

        if self.verbose:
            print(f"COMPLETE. Total elapsed: {self.elapsed}")
