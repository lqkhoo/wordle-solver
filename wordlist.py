import numpy as np


# Symbol sets.
# Implementation assumes fewer than 255 symbols used to allow representing codewords with ubyte arrays.
SYM_ENGLISH = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]



class WordleWordlist(object):

    def __init__(self, codeword_length: int, symbols: list[str], candidates_path: str, valid_words_path: str=None):
        # @param codeword_length: The fixed length of codewords for this wordle. Original is 5.
        # @param symbols: An exhaustive list of valid glyphs for this wordle. Original is [a-z].
        # @param candidates_path: Filepath to txt file of codewords in the solution set.
        # @param valid_words_path: Filepath to codewords not in the solution set, but are nontheless accepted as valid guesses.

        assert(len(symbols < 256)) # Enforce limitation for encoding codewords in ubytes.

        self.symbols: list[str]         = symbols
        self.codeword_length: int       = codeword_length
        self.pool_raw:       list[str]  = None
        self.valid_raw:      list[str]  = None
        self.candidates_raw: list[str]  = None
        self.pool:       np.ndarray     = None # dtype ubyte
        self.valid:      np.ndarray     = None # dtype ubyte
        self.candidates: np.ndarray     = None # dtype ubyte

        self.dic_encode: dict[str, int] = None
        self.dic_decode: dict[int, str] = None

        self.candidates_raw = set(self._read_wordlist(candidates_path))
        if valid_words_path is not None:
            self.valid_raw = set(self._read_wordlist(valid_words_path))
        else:
            self.valid_raw = set()
        self.pool_raw = self.candidates_raw.union(self.valid_raw)

        self.candidates_raw = sorted(list(self.candidates_raw))
        self.valid_raw = sorted(list(self.valid_raw))
        self.pool_raw = sorted(list(self.pool_raw))

        self.dic_encode, self.dic_decode = self._build_dicts()
        self.candidates = self._encode_pool(self.candidates_raw)
        self.valid = self._encode_pool(self.valid_raw)
        self.pool = self._encode_pool(self.pool_raw)


    def __repr__(self) -> str:
        repr =  "{}(n_syms={}, length={}, n_candidates={}, n_valid={})".format(
            self.__class__.__name__,
            len(self.symbols),
            self.codeword_length,
            len(self.candidates_raw),
            len(self.valid_raw)
        )
        return repr


    def decode(self, codeword: np.ndarray) -> str:
        return "".join([self.dic_decode[x] for x in codeword])


    def encode(self, codeword: str) -> np.ndarray:
        return np.array([self.dic_encode[x] for x in list(codeword)], dtype=np.ubyte)


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
        mat = np.zeros((m, n), dtype=int)
        array_T = array.T
        for i in range(m):
            uniques, counts = np.unique(array_T[i], return_counts=True)
            mat[i, uniques] = counts
        return mat


    def _build_dicts(self) -> tuple[dict[str, int], dict[int, str]]:
        dict_encode = dict()
        dict_decode = dict()
        for i in range(len(self.symbols)):
            sym = self.symbols[i]
            dict_encode[sym] = i
            dict_decode[i] = sym
        return (dict_encode, dict_decode)


    def _encode_pool(self, pool: list[str]) -> np.ndarray:
        n = len(pool)
        arr = np.zeros((n, self.codeword_length), dtype=np.ubyte)
        for i in range(n):
            arr[i] = self.encode(pool[i])
        return arr


    def _read_wordlist(self, path: str) -> list[str]:
        try:
            wordlist = open(path, 'r').read().splitlines()
            return wordlist
        except IOError:
            print("IOError")


class WordleWordlistOriginal(WordleWordlist):
    def __init__(self):
        super().__init__(
            5, SYM_ENGLISH,
            "data/original_candidates.txt",
            "data/original_valid.txt"
        )
