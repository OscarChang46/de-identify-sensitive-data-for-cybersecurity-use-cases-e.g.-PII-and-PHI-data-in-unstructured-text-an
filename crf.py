import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import time 
import json
import datetime

# input: feature
# output: probability


# generic form
def probability_function():
    pass


def _generate_potential_table(params, num_labels, feature_set, X, inference=True):
    '''
    Generate a potential table using given obeservations.
    * potential_table[t][pre_y, y]
        := exp(inner_product(params, feature_vector(pre_y, y, X, t)))
    '''
    pass




def _forward_backward():
    pass