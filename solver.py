import cupy as cp
import numpy as np
import networkx

from wordlist import ( WordleWordlist )

class WordleSolver(object):
    # Base implementation of a solver for generic Wordle-like puzzles.
    # Supports arbitrary symbol sets (up to 255 individual symbols) and
    # codeword lengths up to 20 different symbols.

    def __init__(self, wordlist: WordleWordlist, logging: bool=True, verbose: bool=True):

        # Codewords longer than 20 symbols generate more response vectors than
        # could be enumerated using uint32. For practical purposes, most words
        # aren't even this long to begin with.
        assert(wordlist.codeword_length <= 20)

        self._wordlist: WordleWordlist = wordlist
        self._valid_gpu:         cp.ndarray = None # dtype ubyte
        self._candidates_gpu:    cp.ndarray = None # dtype ubyte
        self._response_keys_gpu: cp.ndarray = None # dtype ubyte
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

        self.logging: bool = logging
        self.verbose: bool = verbose


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

    @property
    def response_enc_dtype(self):
        xp = cp if self.has_gpu else np
        if self._wordlist.codeword_length <= 5: # wordle original
            return xp.ubyte  # 3^5 == 243 < 255
        elif self._wordlist.codeword_length <= 10:
            return xp.uint16 # 3^10 == 59049 < 65535
        else:
            return xp.uint32 # 3^20 == 3.4 * 10^9 < 2^32
        

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
        # where k is the length of codewords. Valid interval is [0-2] so ubyte.
        k = self.codeword_length
        n = 3 ** k
        mat = np.zeros((k, n), dtype=np.ubyte)
        pattern = np.array([2,1,0], dtype=np.ubyte)
        for i in range(k):
            mat[i,:] = np.repeat(np.tile(pattern, 3**i), 3**(k-i-1))
        mat = mat.T
        # Make sure matrix indices agree with self.decode_response
        mat = np.flipud(mat)
        mat = np.fliplr(mat)
        return np.ascontiguousarray(mat) # (3**k, k)


    def get_response(self, guess: np.ndarray, truth: np.ndarray, encode: bool=True) -> np.ndarray:

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

        exact_matches = (guess == truth[:,None,:]).transpose(1,0,2)           # (g, t, k)
        inexact_matches = (guess.reshape(-1) == truth[:,:,None]).any(axis=1)  # ((g*k) == (t,k,1)) --> (g, k, t*k).any(axis=1) --> (g, t*k)
        inexact_matches = inexact_matches.reshape((t,g,k)).transpose((1,0,2)) # (g, t, k)
        inexact_matches = inexact_matches & ~exact_matches

        ## result = (exact_matches * 2 + inexact_matches * 1).astype(xp.ubyte)   # (g, t, k)
        # The line above will create copies of int32 arrays from the array multiplies so we avoid that.
        result = xp.add(
            xp.multiply(exact_matches,   2, dtype=xp.ubyte),
            xp.multiply(inexact_matches, 1, dtype=xp.ubyte)
        ) # (g, t, k)
        if encode:
            result = self.encode_response(result) # (g, t) Encode to response keys i.e. single integers in place of the vectors.
        return result # (g, t)

    
    def encode_response(self, expanded_response: np.ndarray) -> np.ndarray:
        xp = cp.get_array_module(expanded_response)
        k = self.codeword_length
        r = expanded_response # (g, t, k)
        encoder = 3**xp.arange(k)
        ## r = xp.sum(r * encoder, axis=2)
        r = xp.sum(xp.multiply(r, encoder, dtype=self.response_enc_dtype), axis=2)
        return r # (g, t)


    def decode_response(self, compact_response: np.ndarray) -> np.ndarray:
        # Decoding by radix decomposition. 3 bits per symbol position.
        # It's slower than encode but we only use this for printing and logging.
        xp = cp.get_array_module(compact_response)
        k = self.codeword_length
        c = xp.copy(compact_response) # (g, t)
        g, t = xp.shape(c)
        r = xp.zeros((g, t, k), dtype=xp.ubyte)
        for i in range(k):
            r[:,:,i] = c % 3
            c = c // 3
        return r # (g, t, k)
    

    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray, responses: np.ndarray, live_cidxs) -> int:
        raise NotImplementedError()


    def solve(self, truth_raw: str, max_iters: int=10):
        # Generate solution path given the truth string.
        # This obviously cannot be used in an interactive setting.
        raise NotImplementedError()



class WordleMinimaxSolver(WordleSolver):
    # Wordle solver implementation using Knuth's minimax algorithm, which he used
    # to solve MM(4,6) Mastermind in 1976. Minimax selects the guess which produces
    # a partitioning which has the smallest largest resulting partition of the 
    # solution space. Hence minimax.

    def __init__(self, wordlist, logging: bool=True, verbose: bool=True):
        super().__init__(wordlist, logging, verbose)


    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray,
                        responses: np.ndarray, live_cidxs: np.ndarray) -> int:
        # Let C, V, P=V+C be the sizes of the *initial* candidate, valid, and pool arrays. k is codeword length.
        # candidates: (c, k)   Array of current candidates. c <= C. Changes every time step.
        # pool:       (P, k)   This is the same every time step.
        # responses:  (P, c)   Array of responses vs current candidates. Changes every time step.
        # live_cidxs: (c, )    This array is used to track the indices of current candidates in pool / responses.
        #                      e.g. if used as fancy index 'pool[live_cidxs]' is called, it returns
        #                      a sliced copy of pool that is equal to 'candidates'.

        xp = cp.get_array_module(candidates)

        # Short-circuit when there's only one valid guess left. That's our answer.
        if xp.shape(live_cidxs)[0] == 1:
            # Logging
            if self.verbose:
                print("  Single candidate left. Short-circuit.")

            return live_cidxs[0]

        p, k = xp.shape(pool)
        n_keys = xp.shape(self.keys)[0]

        # (p, n_keys) dtype int64 matrix. Each row (for each guess) contains
        # the sizes of resulting partitions w.r.t. each possible response key.
        psizes = xp.apply_along_axis(xp.bincount, 1, responses, minlength=n_keys)

        # Skip the unique call for better performance. Minimax doesn't clarify what the
        # 'second largest partition' is, as in whether it's second largest after a unique filter or not.
        # psorted = xp.fliplr(xp.apply_along_axis(utils.justified_unique1d, 1, psizes, filler=0)) # (p, n_keys)
        # Just sort instead.
        psorted = xp.fliplr(xp.sort(psizes)) # row sort descending

        # Minimax algorithm: Find the guess that has the minimal largest partition.
        # Tiebreaking lvl 1: Compare second-largest partitions and so on.
        guess_idxs = None
        for i in range(n_keys):
            col = psorted[:,i]
            minimax_idxs = xp.argwhere(col == xp.amin(col)).reshape(-1)
            if guess_idxs is None:
                guess_idxs = minimax_idxs
            else:
                mask = xp.isin(guess_idxs, minimax_idxs, assume_unique=True)
                test_idxs = guess_idxs[mask]
                if xp.size(test_idxs) == 0:
                    break # If set of guesses becomes empty, stop and use last valid set.
                guess_idxs = test_idxs
        # Logging
        if self.verbose:
            guesses_raw = []
            guesses_cpu = pool[guess_idxs].get()
            for g in guesses_cpu:
                guesses_raw.append(self._wordlist.decode(g))
            print("  Minimax guesses ({}): {}".format(guess_idxs.shape[0], guesses_raw))

        # Tiebreaking lvl 2: Prioritize guesses which are in the candidate set.
        test_idxs = guess_idxs[xp.isin(guess_idxs, live_cidxs)]
        if xp.size(test_idxs) == 0:
            # Logging
            if self.verbose:
                print("  Minimax guesses (candidates) (0)")
        else:
            guess_idxs = test_idxs
            # Logging
            if self.verbose:
                guesses_raw = []
                guesses_cpu = pool[guess_idxs].get()
                for g in guesses_cpu:
                    guesses_raw.append(self._wordlist.decode(g))
                print("  Minimax guesses (candidates) ({}): {}".format(guess_idxs.shape[0], guesses_raw))

        # Despite tiebreaking, by symmetry, we may still have more than one minimax guess.
        best_guess_idx = guess_idxs[0] # In that case, just pick the first such one.
        return best_guess_idx


    def solve(self, truth_raw: str, max_iters: int=10):
        # Simulate a solving session with a given truth value (string) to check convergence.

        assert(truth_raw in self._wordlist.candidates_raw)
        xp = cp if self.has_gpu else np

        # For logging only
        guesses = []
        messages = []
        remaining = []

        n_keys = xp.shape(self.keys)[0]
        truth = xp.asarray(self._wordlist.encode(truth_raw))
        candidates, valid = self.candidates, self.valid
        c, k = xp.shape(candidates)

        pool = xp.vstack((candidates, valid)) # (p, k)

        # Compute all possible responses for all pairs of (pool, candidate).
        # (p, c) dtype uint8 matrix. Each row (for each guess) contains
        # the response key encoding the appropriate response vector for each possible solution.
        R = self.get_response(pool, candidates) # (p, c)

        # This keeps track of which candidate indices are still live. Used for tracking live
        # indices when selecting from the pool.
        live_cidxs = xp.arange(c)

        nsteps = 1
        for _ in range(max_iters):
            
            # Logging
            if self.verbose:
                print(f"Time step: {nsteps}")

            c, k = xp.shape(candidates)

            # Get best guess.
            guess_idx = self.get_best_guess(candidates, pool, R, live_cidxs) # scalar

            # Logging
            if self.logging:
                guess_raw = self._wordlist.decode(pool[guess_idx].get())
                guesses.append(guess_raw)

            # Generate response.
            R_row = R[guess_idx] # (c, )
            mask = (truth[None,:] == candidates)
            truth_idx = xp.argwhere(xp.all(mask, axis=1))[0] # scalar
            response = R_row[truth_idx][0] # scalar

            # Logging
            if self.logging:
                response_raw = self._keys_gpu[response]
                messages.append(response_raw)
                if self.verbose:
                    print("  Best guess {}: '{}', response: {}".format(nsteps, guess_raw, response_raw))

            # Check for winning condition.
            if response == n_keys-1:
                # Logging
                if self.verbose:
                    print(f"SOLVED in {nsteps} steps.")

                break

            # Partition remaining candidates.
            # 1. Generate the mask,
            # 2. Eliminate the columns of R that correspond to eliminated candidates.
            # 3. Eliminate the rows in candidates that correspond to eliminated candidates.
            mask = (R_row == response)
            idx = xp.arange(c)[mask]
            R = R[:,idx]
            candidates = candidates[idx,:]
            live_cidxs = live_cidxs[idx]

            # Logging
            if self.logging:
                candidates_cpu = candidates.get()
                candidates_raw = []
                for c in candidates_cpu:
                    candidates_raw.append(self._wordlist.decode(c))
                remaining.append(candidates_raw)
                if self.verbose:
                    print("    Partition ({}): {}".format(len(candidates_raw), candidates_raw))

            nsteps += 1

        return candidates.ravel(), nsteps, guesses, messages
        