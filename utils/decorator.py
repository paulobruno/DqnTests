from time import time


def timeit(fn):
    def timed(*args, **kw):
        ts = time()
        result = fn(*args, **kw)
        te = time()
        print("{} Elapsed time {}".format(fn.__name__.upper(), (te - ts) * 1000))
        return result

    return timed
