import cupy as cp
import numpy as np
from wordlist import ( WordleWordlist )

class WordleSolver(object):
    
    def __init__(self, wordlist: WordleWordlist):
        self._wordlist: WordleWordlist = wordlist
        self._valid_gpu = None
        self._candidates_gpu = None
        self._response_keys_gpu = None
        self.has_gpu: bool = False

        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices >= 1:
            self.has_gpu = True
            cp.cuda.runtime.setDevice(0) # Just use the first GPU.
            # Copy arrays to GPU memory.
            self._valid_gpu = cp.asarray(wordlist.valid)
            self._candidates_gpu = cp.asarray(wordlist.candidates)
            self._keys_cpu = self.get_response_keys()
            self._keys_gpu = cp.asarray(self._keys_cpu)
        else:
            print("WARN: No CUDA device available.")

    @property
    def codeword_length(self):
        return self._wordlist.codeword_length
    @property
    def symbols(self):
        return self._wordlist.symbols

    @property
    def valid(self):
        return self._valid_gpu if self.has_gpu else self._wordlist.valid
    @property
    def candidates(self):
        return self._candidates_gpu if self.has_gpu else self._wordlist.candidates
    @property
    def keys(self):
        return self._keys_gpu if self.has_gpu else self._keys_cpu

    
    def __repr__(self):
        xp = cp if self.has_gpu else np
        return "{}(n_candidates={}, n_valid={})".format(
            self.__class__.__name__,
            xp.shape(self.candidates)[0],
            xp.shape(self.valid)[0]
        )


    def use_gpu(self, use_gpu: bool=True) -> None:
        # Manual toggle for whether to use GPU.
        self.has_gpu = use_gpu


    def get_response_keys(self) -> np.ndarray:
        # CPU routine.
        # Return all possible response vectors as a 2d matrix of shape (3**k, k),
        # where k is the length of codewords.
        k = self.codeword_length
        n = 3 ** k
        mat = np.zeros((k, n), dtype=int)
        pattern = np.array([2,1,0], dtype=int)
        for i in range(k):
            mat[i,:] = np.repeat(np.tile(pattern, 3**i), 3**(k-i-1))
        return np.ascontiguousarray(mat.T)


    def get_response(self, guess: np.ndarray, truth: np.ndarray) -> np.ndarray:
        # For the response vector, we encode as follows:
        # 0 == no match
        # 1 == inexact match
        # 2 == exact match

        xp = cp.get_array_module(guess)

        # Increase dimensionality if we're getting flat arrays
        if len(xp.shape(guess)) == 1: guess = guess[None,:]
        if len(xp.shape(truth)) == 1: truth = truth[:,None]

        g, k = xp.shape(guess) # k == self.codeword_length
        t, _ = xp.shape(truth)

        result = xp.zeros((t, g, k), dtype=int)

        exact_matches = (guess == truth[:,None,:])        # (t, g, k)
        inexact_matches = xp.zeros((t, g, k), dtype=bool) # (t, g, k)
        for i in range(t):
            inexact_matches[i,:,:] = xp.isin(guess, truth[i], assume_unique=False)
        inexact_matches = inexact_matches & ~exact_matches # (t, g, k)

        result += exact_matches * 2 + inexact_matches * 1
        return result.transpose((1,0,2)) # (g, t, k)
    

    def compute_information(self, candidates: np.ndarray, valid: np.ndarray, eliminated: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Note: method doesn't return information directly. It returns partition
        # sizes, which could be used to compute information.
        # @param candidates: array of remaining candidate words.
        # @param valid: array of valid words (constant every iteration).
        # @param eliminated: array of all candidate words eliminated from previous cycles.

        xp = cp.get_array_module(candidates)
        
        keys = self.keys
        n_keys = xp.shape(keys)[0]
        key_idx = xp.arange(n_keys)

        c, k = xp.shape(candidates) # k == codeword_length
        v, _ = xp.shape(valid)
        p = c + v

        if eliminated is None:
            pool = xp.vstack((candidates, valid)) # (p, k)
        else:
            pool = xp.vstack((candidates, eliminated, valid))

        # For every possible solution, get its response vector for every possible guess.
        r = self.get_response(pool, candidates) # (p, c, k)

        # (p, c) matrix. Each row (for each guess) contains
        # the response key encoding the appropriate response vector for each possible solution.
        responses = xp.zeros((p, c), dtype=int)

        # (p, n_keys) matrix. Each row (for each guess) contains
        # the sizes of resulting partitions w.r.t. each possible response key.
        V = xp.zeros((p, n_keys), dtype=int)

        for i in range(p):
            mask = (r[None,i,:] == keys[:,None,:]).T # (k, p, n_keys)
            mask = xp.all(mask, axis=0) # (p, n_keys)
            response_indices = xp.sum(mask * key_idx, axis=1)
            responses[i,:] = response_indices
            uniques, counts = xp.unique(response_indices, return_counts=True)
            V[i, uniques] = counts

        return V, responses, pool


    def get_best_guess(self, candidates: np.ndarray, valid: np.ndarray, eliminated: np.ndarray=None):
        raise NotImplementedError()


    def solve(self, truth_raw: str, max_iters: int=10):
        # Generate solution path given the truth string.
        # This obviously cannot be used in an interactive setting.
        raise NotImplementedError()


class WordleMinimaxSolver(WordleSolver):

    def __init__(self, wordlist):
        super().__init__(wordlist)


    def get_best_guess(self, candidates: np.ndarray, valid: np.ndarray, eliminated: np.ndarray=None) -> tuple[np.ndarray, int, str]:

        xp = cp.get_array_module(candidates)

        V, responses, pool = self.compute_information(candidates, valid, eliminated)
        p, n_keys = xp.shape(V)

        # Knuth's algorithm requires calling np.unique, which creates jagged arrays.
        # These jaggies are put into matrix M, which is pre-filled with maxint.
        M = xp.full((p, n_keys), xp.iinfo(int).max)
        for i in range(p):
            entries = xp.unique(V[i])[::-1]
            length = xp.size(entries)
            M[i][0:length] = entries

        p, k = xp.shape(pool)
        n_keys = xp.shape(self.keys)[0]

        # We want to find the guess with the smallest maximum partition size. The
        # partition sizes for each guess w.r.t. each possible candidate was in matrix V,
        # now reduced to only unique entries in matrix M.
        # Starting with every possible guess as valid, our goal is to reduce that number
        # to a single one by tie-breaking, comparing their next-smallest partitions until we run out.
        guess_idx = xp.arange(p)
        for i in range(n_keys):
            col = M[:,i]
            maximin_idx = xp.argwhere(col == xp.amin(col)).flatten()
            # Reduce the set of possible guesses
            mask = xp.isin(guess_idx, maximin_idx, assume_unique=True)
            test_idx = guess_idx[mask]
            if xp.size(test_idx) == 0:
                break # If set of guesses becomes empty, stop and use last valid set.
            guess_idx = test_idx

        # Despite tiebreaking, by symmetry, we may have more than one minimax guess.
        best_guess_idx = guess_idx[0] # In that case, just pick the first such one.
        best_guess_raw = self._wordlist.decode(pool[best_guess_idx].get())

        # (p, n_candidates), int, str
        return responses, best_guess_idx, best_guess_raw


    def solve(self, truth_raw: str, max_iters:int=10, logging: bool=True, verbose: bool=True):

        assert(truth_raw in self._wordlist.candidates_raw)
        xp = cp if self.has_gpu else np

        n_keys = xp.shape(self.keys)[0]
        truth = xp.asarray(self._wordlist.encode(truth_raw))
        candidates = self.candidates
        valid = self.valid
        eliminated = None

        # For logging only
        guesses = []
        messages = []
        remaining = []

        nsteps = 1
        for i in range(max_iters):

            responses, best_guess_idx, best_guess_raw = self.get_best_guess(candidates, valid, eliminated)

            # Generate valid response from truth value. In an interactive setting, this needs
            # to be read from another source.
            mask = (truth[None,:] == candidates)
            truth_idx = xp.argwhere(xp.all(mask, axis=1))[0]
            response = responses[best_guess_idx, truth_idx][0] # the 0 is to unpack the unit length array

            # Logging
            response_raw = self._keys_gpu[response]
            if logging:
                guesses.append(best_guess_raw)
                messages.append(response_raw)
            if verbose:
                print("Guess: '{}', response: {}".format(best_guess_raw, response_raw))

            c, k = xp.shape(candidates)
            if c <= 1:
                break

            mask = (responses[best_guess_idx] == response)
            candidates = candidates[xp.arange(c)[mask]]
            eliminated = candidates[xp.arange(c)[~mask]]

            # Logging
            candidates_cpu = candidates.get()
            candidates_raw = []
            if logging:
                for c in candidates_cpu:
                    candidates_raw.append(self._wordlist.decode(c))
                remaining.append(candidates_raw)
            if verbose:
                print("Partition: {}".format(candidates_raw))

            if i >= max_iters:
                break

            nsteps += 1

        return candidates.flatten(), nsteps, guesses, messages
        