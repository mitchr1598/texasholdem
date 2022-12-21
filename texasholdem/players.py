import numpy as np


class TexasRange:
    def __init__(self, range_: str):
        self.range_ = np.array([float(f) for f in range_.split()])


