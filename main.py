if __name__ == '__main__':

    import cupy as cp
    import numpy as np
    # from cupyx.profiler import benchmark


    from wordlist import ( WordleWordlistOriginal )
    from solver import ( WordleMinimaxSolver )

    wordlist = WordleWordlistOriginal()
    solver = WordleMinimaxSolver(wordlist)
    foo = solver.solve("haute")
    print(foo)
