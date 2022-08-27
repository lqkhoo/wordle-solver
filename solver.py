import cupy as cp
import numpy as np

# GPU profiling tools
import nvtx

from wordlist import ( WordleWordlist )

class WordleSolver(object):
    # Base implementation of a solver for generic Wordle-like puzzles.
    # Supports arbitrary symbol sets (up to 255 individual symbols) and
    # codeword lengths up to 20 different symbols.

    def __init__(self, wordlist: WordleWordlist, logging: bool=False, verbose: bool=False):

        # Codewords longer than 20 symbols generate more response vectors than
        # could be enumerated using uint32. For practical purposes, most words
        # aren't even this long to begin with.
        assert(wordlist.codeword_length <= 20)

        self._wordlist: WordleWordlist = wordlist
        self._candidates_gpu:    np.ndarray = None # dtype ubyte
        self._valid_gpu:         np.ndarray = None # dtype ubyte
        self._pool_gpu:          np.ndarray = None # dtype ubyte

        self.has_gpu: bool = True if cp.cuda.runtime.getDeviceCount() >= 1 else False
        if self.has_gpu:
            cp.cuda.runtime.setDevice(0) # Just use the first GPU.
            # Copy arrays to GPU memory.
            self._candidates_gpu = cp.asarray(wordlist.candidates)
            self._valid_gpu = cp.asarray(wordlist.valid)
            self._pool_gpu = cp.vstack((self._candidates_gpu, self._valid_gpu))
            self._keys_cpu = self._get_response_keys()
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
    def candidates(self):
        return self._candidates_gpu if self.has_gpu else self._wordlist.candidates
    @property
    def valid(self):
        return self._valid_gpu if self.has_gpu else self._wordlist.valid
    @property
    def pool(self):
        return self._pool_gpu if self.has_gpu else self._wordlist.pool
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


    def idx2codewords(self, pool_idxs: np.ndarray) -> list[str]:
        # Synchronizing.
        # Given an array of indices pointing into rows of self.pool, return
        # the codewords as a list of raw strings.
        pool_idxs = cp.asnumpy(pool_idxs) # Sync
        return self._wordlist.idx2codewords(pool_idxs)

    def codewords2idx(self, strings: list[str]) -> np.ndarray:
        # Synchronizing.
        # Given a list of strings, return an array of indices pointing
        # into codewords in self.pool that correspond to those strings.
        pool_idxs = self._wordlist.codewords2idx(strings) # cpu
        if self.has_gpu:
            pool_idxs = cp.asarray(pool_idxs) # Sync
        return pool_idxs

    def array2codewords(self, arr: np.ndarray) -> list[str]:
        # Synchronizing.
        # Given a 2d array of symbol-encoded codewords, return
        # the codewords as a list of strings.
        arr = cp.asnumpy(arr)
        return self._wordlist.array2codewords(arr)

    def codewords2array(self, strings: list[str]) -> np.ndarray:
        # Synchronizing.
        # Given a list of strings, return a 2d array of their
        # symbol-encoded representation.
        arr = self._wordlist.codewords2array(strings)
        if self.has_gpu:
            arr = cp.asarray(arr) # Sync
        return arr.ravel()


    def _get_response_keys(self) -> np.ndarray:
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

    @nvtx.annotate(color='yellow')
    def compute_responses(self, guess: np.ndarray, truth: np.ndarray) -> np.ndarray:
        # Where 'g' is the number of guesses and 't' is the number of (potential) truths to test against,
        # return a matrix of shape (g, t) containing integers encoding the response vectors.

        # For the response vector, we encode each symbol position as follows:
        # 0 == no match
        # 1 == inexact match
        # 2 == exact match

        xp = cp.get_array_module(guess)
        # Increase dimensionality if we're getting flat arrays
        guess = xp.atleast_2d(guess)
        truth = xp.atleast_2d(truth)

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
        result = self.encode_response(result) # (g, t) Encode to response keys i.e. single integers in place of the vectors.
        return result # (g, t)

    
    def encode_response(self, expanded_response: np.ndarray) -> np.ndarray:
        # Encodes an array of response vectors into their integer encoding.
        xp = cp.get_array_module(expanded_response)
        k = self.codeword_length
        r = expanded_response # (g, t, k)
        encoder = 3**xp.arange(k)
        ## r = xp.sum(r * encoder, axis=-1)
        r = xp.sum(xp.multiply(r, encoder, dtype=self.response_enc_dtype), axis=-1)
        return r # (g, t)


    def decode_response(self, compact_response: np.ndarray) -> np.ndarray:
        # Decoding by radix decomposition. 3 bits per symbol position.
        # It's slower than encode but we only use this for printing and logging.
        xp = cp.get_array_module(compact_response)
        k = self.codeword_length
        c = xp.copy(compact_response) # (g, t)
        g, t = xp.shape(c)
        r = xp.empty((g, t, k), dtype=xp.ubyte)
        for i in range(k):
            r[:,:,i] = c % 3 # (g, t)
            c = c // 3
        return r # (g, t, k)
    

    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray, responses: np.ndarray, live_cidxs) -> int:
        raise NotImplementedError()

    @nvtx.annotate(color='red')
    def solve(self, truth_raw: str, max_iters: int=10):
        # Simulate a solving session with a given truth value (string) to check convergence.

        assert(truth_raw in self._wordlist.candidates_raw)
        xp = cp if self.has_gpu else np

        # For logging only
        guesses = []
        messages = []
        remaining = []

        truth = self.codewords2array(truth_raw) # (k, )
        candidates, pool = self.candidates, self.pool
        keys = self.keys

        n_keys = xp.shape(keys)[0]
        c, _ = xp.shape(candidates)

        # Compute all possible responses for all pairs of (pool, candidate).
        # (p, c) dtype uint8 matrix. Each row (for each guess) contains
        # the response key encoding the appropriate response vector for each possible solution.
        R = self.compute_responses(pool, candidates) # (p, c)

        # This keeps track of which candidate indices are still live. Used for tracking live
        # indices when selecting from the pool.
        live_cidxs = xp.arange(c)

        if self.verbose:
            p, _ = xp.shape(pool)
            print(f"Computed responses for {p} x {c} pairs. Solving:")

        nsteps = 1
        for _ in range(max_iters):

            # Logging
            if self.verbose:
                print(f"Time step: {nsteps}")

            # Get best guess.
            guess_idx = self.get_best_guess(candidates, pool, R, live_cidxs) # scalar

            # Generate response.
            R_row = R[guess_idx] # (c, )
            mask = (truth[None,:] == candidates)
            truth_idx = xp.argwhere(xp.all(mask, axis=1))[0] # scalar
            response = R_row[truth_idx][0] # scalar

            # Logging
            if self.logging:
                guess_raw = self.idx2codewords(guess_idx)[0]
                response_raw = keys[response]
                guesses.append(guess_raw)
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
            c, k = xp.shape(candidates) # Recompute c
            idxs = xp.arange(c)[R_row == response]
            R = R[:,idxs]
            candidates = candidates[idxs,:]
            live_cidxs = live_cidxs[idxs]

            # Logging
            if self.logging:
                candidates_raw = self.idx2codewords(live_cidxs)
                remaining.append(candidates_raw)
                if self.verbose:
                    print("    Partition ({}): {}".format(len(candidates_raw), candidates_raw))

            nsteps += 1

        return candidates.ravel(), nsteps, guesses, messages



class WordleSimpleSolver(WordleSolver):
    # This solver simply picks the first available solution.

    def __init__(self, wordlist, logging: bool=False, verbose: bool=False):
        super().__init__(wordlist, logging, verbose)

    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray,
                        responses: np.ndarray, live_cidxs: np.ndarray) -> int:
        return live_cidxs[0]
        


class WordleMinimaxSolver(WordleSolver):
    # Wordle solver implementation using Knuth's minimax algorithm, which he used
    # to solve MM(4,6) Mastermind in 1976. Minimax selects the guess which produces
    # a partitioning which has the smallest largest resulting partition of the 
    # solution space. Note: This is not minimax search used for 2-player games.

    def __init__(self, wordlist, logging: bool=False, verbose: bool=False):
        super().__init__(wordlist, logging, verbose)

    @nvtx.annotate(color='green')
    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray,
                        responses: np.ndarray, live_cidxs: np.ndarray) -> int:
        # Let C, V, P=V+C be the sizes of the *initial* candidate, valid, and pool arrays. k is codeword length.
        # candidates: (c, k)   Array of current candidates. c <= C. Changes every time step.
        # pool:       (P, k)   This is the same every time step.
        # responses:  (P, c)   Array of responses vs current candidates. Changes every time step.
        # live_cidxs: (c, )    This array is used to track the indices of current candidates in pool / responses.
        #                      e.g. if used as fancy index 'pool[live_cidxs]', it returns
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
            print("  Minimax guesses ({}): {}".format(guess_idxs.shape[0], self.idx2codewords(guess_idxs)))

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
                print("  Minimax guesses (candidates) ({}): {}".format(guess_idxs.shape[0], self.idx2codewords(guess_idxs)))

        # Despite tiebreaking, by symmetry, we may still have more than one minimax guess.
        best_guess_idx = guess_idxs[0] # In that case, just pick the first such one.

        return best_guess_idx



class WordleMaxEntropySolver(WordleSolver):

    def __init__(self, wordlist, logging: bool=False, verbose: bool=False):
        super().__init__(wordlist, logging, verbose)

    def get_best_guess(self, candidates: np.ndarray, pool: np.ndarray,
                        responses: np.ndarray, live_cidxs: np.ndarray) -> int:
        
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