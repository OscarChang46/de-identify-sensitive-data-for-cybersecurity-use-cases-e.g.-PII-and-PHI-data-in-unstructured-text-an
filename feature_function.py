import numpy as np

class StausFF():
    def __init__(self, c):
        self.c = c
    
    def __call__(self, y_, y, X, i):
        return 1 if y == self.c else 0

class ObsFF():
    pass

class TransFF():
    pass