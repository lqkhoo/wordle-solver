if __name__ == '__main__':

    import cupy as cp
    import numpy as np

    from cupyx.profiler import benchmark

    from wordlist import ( WordleWordlistOriginal )
    from solver import ( WordleMinimaxSolver )

    # cp.cuda.profiler.start()

    wordlist = WordleWordlistOriginal()
    solver = WordleMinimaxSolver(wordlist)
    
    print(benchmark(solver.get_response, (solver.pool, solver.candidates), n_repeat=10))
