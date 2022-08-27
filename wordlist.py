import numpy as np


# Symbol sets.
# Implementation assumes fewer than 255 symbols used to allow representing codewords with ubyte arrays.
SYM_ENGLISH = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


class WordleWordlist(object):
    # CPU-only class.

    def __init__(self, codeword_length: int, symbols_raw: list[str], candidates_path: str, valid_words_path: str=None):
        # codeword_length: The fixed length of codewords for this wordle. Original is 5.
        # symbols_raw: An exhaustive list of valid glyphs for this wordle. Original is [a-z].
        # candidates_path: Filepath to txt file of codewords in the solution set.
        # valid_words_path: Filepath to codewords not in the solution set, but are nontheless accepted as valid guesses.

        assert(len(symbols_raw) < 256) # Enforce limitation for encoding codewords in ubytes.

        self.symbols_raw: list[str]     = symbols_raw
        self.symbols:     list[int]     = [x[0] for x in enumerate(symbols_raw)]
        self.codeword_length: int       = codeword_length
        self.candidates_raw: list[str]  = None
        self.valid_raw:      list[str]  = None
        self.pool_raw:       list[str]  = None
        self.candidates: np.ndarray     = None # dtype ubyte
        self.valid:      np.ndarray     = None # dtype ubyte
        self.pool:       np.ndarray     = None # dtype ubyte

        self.sym_encode: dict[str, int] = dict(zip(self.symbols_raw, self.symbols))
        self.sym_decode: dict[int, str] = dict(zip(self.symbols, self.symbols_raw))

        # Dictionary mapping a codeword string to its index in self.pool
        self.codeword2poolidx: dict[str, int] = None


        self.candidates_raw = set(self._read_wordlist(candidates_path))
        if valid_words_path is not None:
            self.valid_raw = set(self._read_wordlist(valid_words_path))
        else:
            self.valid_raw = set()

        self.candidates_raw = sorted(list(self.candidates_raw))
        self.valid_raw = sorted(list(self.valid_raw))
        # This one's not completely sorted to line up with the arrays in solver
        # which is stacked as follows: cp.vstack(candidates, valid)
        self.pool_raw = self.candidates_raw + self.valid_raw

        self.candidates = self._encode_pool(self.candidates_raw)
        self.valid = self._encode_pool(self.valid_raw)
        self.pool = self._encode_pool(self.pool_raw)

        self.dic_codeword2idx = dict(zip(self.pool_raw, [x for x,_ in enumerate(self.pool)]))


    def __repr__(self) -> str:
        repr =  "{}(n_syms={}, length={}, n_candidates={}, n_valid={})".format(
            self.__class__.__name__,
            len(self.symbols),
            self.codeword_length,
            len(self.candidates_raw),
            len(self.valid_raw)
        )
        return repr


    def idx2codewords(self, pool_idxs: np.ndarray) -> list[str]:
        # Given an array of indices pointing into rows of self.pool, return
        # the codewords as a list of raw strings.
        pool_idxs = np.atleast_1d(pool_idxs)
        return [self.pool_raw[idx] for idx in pool_idxs]

    def codewords2idx(self, strings: list[str]) -> np.ndarray:
        # Given a list of strings, return an array of indices pointing
        # into codewords in self.pool that correspond to those strings.
        return np.array([self.dic_codeword2idx[x] for x in strings], dtype=int)

    def array2codewords(self, arr: np.ndarray) -> list[str]:
        # Given a 2d array of symbol-encoded codewords, return
        # the codewords as a list of strings.
        arr = np.atleast_2d(arr)
        m, _ = np.shape(arr)
        codewords = []
        for i in range(m):
            codewords.append(''.join([self.sym_decode[x] for x in arr[i]]))
        return codewords

    def codewords2array(self, strings: list[str]) -> np.ndarray:
        # Given a list of strings, return a 2d array of their
        # symbol-encoded representation.
        if type(strings) == str: strings = [strings]
        arr = np.empty((len(strings), self.codeword_length), dtype=np.ubyte)
        for i in range(len(strings)):
            s = strings[i]
            arr[i] = np.array([self.sym_encode[x] for x in s])
        return arr


    def _encode_pool(self, pool: list[str]) -> np.ndarray:
        n = len(pool)
        arr = np.empty((n, self.codeword_length), dtype=np.ubyte)
        for i in range(n):
            arr[i] = np.array([self.sym_encode[x] for x in pool[i]])
        return arr


    def _read_wordlist(self, path: str) -> list[str]:
        try:
            wordlist = open(path, 'r').read().splitlines()
            return wordlist
        except IOError:
            print("IOError")


    def get_symbol_counts_global(self, array: np.ndarray) -> np.ndarray:
        # Returns a 1d matrix of global symbol counts in the specified array.
        # For example, if self.candidates is passed in, this returns the individual counts of all symbols,
        # counting repeats, regardless of their position in the word.
        mat = self.get_symbol_counts_by_position(array)
        mat = np.sum(mat, axis=0)
        return mat

    
    def get_symbol_counts_by_position(self, array: np.ndarray) -> np.ndarray:
        # Returns a 2d matrix of symbol counts at each position.
        m, n = self.codeword_length, len(self.symbols)
        mat = np.zeros((m, n), dtype=int) # This has to be np.zeros.
        array_T = array.T
        for i in range(m):
            uniques, counts = np.unique(array_T[i], return_counts=True)
            mat[i, uniques] = counts
        return mat


class WordleWordlistOriginal(WordleWordlist):
    def __init__(self):
        super().__init__(
            5, SYM_ENGLISH,
            "data/original_candidates.txt",
            "data/original_valid.txt"
        )
