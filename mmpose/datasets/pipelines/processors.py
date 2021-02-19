class HandTypeFlipProcessor:

    def __init__(self, key):
        self.key = key

    def __call__(self, results, *args, **kwargs):
        results[self.key].reverse()
        return results
