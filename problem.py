import cupy as cp
import numpy as np

from wordlist import ( WordleWordlist )

class WordleProblem(object):
    # Class encapsulating a Wordle problem that returns a response vector when given a guess.
    # Its functionality is a strict subset of solver so it's unused at the moment.
    
    def __init__(self, wordlist: WordleWordlist, candidates: np.ndarray, pool: np.ndarray, truth: str=None):
        self._wordlist = wordlist
        # Initial candidate and pool.
        # When running in conjunction with a solver, the arrays need to match those in the solver.
        self._candidates: np.ndarray = candidates
        self._pool: np.ndarray = pool

        self.truth: np.ndarray = None
        self.set_truth(truth) if truth is not None else self.set_random_truth()


    def compute_response(self, guess: np.ndarray) -> np.ndarray:
        # Copy of solver's method, but raveled and without encoding.

        xp = cp.get_array_module(guess)
        guess = xp.atleast_2d(guess)
        truth = xp.atleast_2d(self.truth)

        g, k = xp.shape(guess)
        t, _ = xp.shape(truth)

        exact_matches = (guess == truth[:,None,:]).transpose(1,0,2)           # (g, t, k)
        inexact_matches = (guess.reshape(-1) == truth[:,:,None]).any(axis=1)  # ((g*k) == (t,k,1)) --> (g, k, t*k).any(axis=1) --> (g, t*k)
        inexact_matches = inexact_matches.reshape((t,g,k)).transpose((1,0,2)) # (g, t, k)
        inexact_matches = inexact_matches & ~exact_matches

        result = xp.add(
            xp.multiply(exact_matches,   2, dtype=xp.ubyte),
            xp.multiply(inexact_matches, 1, dtype=xp.ubyte)
        ) # (g, t, k)
        return result.ravel() # (g, t)


    def set_truth(self, string: str) -> None:
        xp = cp.get_array_module(self._candidates)
        arr = self._wordlist.codewords2array(string)
        arr = xp.asarray(arr)
        self.truth = arr.ravel()
        

    def set_random_truth(self) -> None:
        xp = cp.get_array_module(self._candidates)
        c, _ = xp.shape(self._candidates)
        idx = xp.random.randint(0, c, dtype=xp.int32)
        codeword = self._candidates[idx]
        self.truth = codeword


    

