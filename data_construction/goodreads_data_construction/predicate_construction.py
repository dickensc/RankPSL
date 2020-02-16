"""
This script constructs the necessary data files needed for psl-ltr-recommender
"""
import os
import pandas as pd
import numpy as np

import sys
sys.path.append('../')

from predicate_construction_helpers import query_relevance_cosine_similarity
from predicate_construction_helpers import query_item_preferences
from predicate_construction_helpers import hac_cluster_from_distance
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels

n_folds = 5

"""
Import raw data
"""


"""
Create data directory to write output to
"""
if not os.path.exists('./goodreads'):
    os.makedirs('./goodreads')

"""
Dev. subset to only 100 users and 200 movies
"""


for fold in range(n_folds):
    """
    Create data directory to write output to
    """
    if not os.path.exists('./goodreads/' + str(fold)):
        os.makedirs('./goodreads/' + str(fold))

    if not os.path.exists('./goodreads/' + str(fold) + '/learn/'):
        os.makedirs('./goodreads/' + str(fold) + '/learn/')

    if not os.path.exists('./goodreads/' + str(fold) + '/eval/'):
        os.makedirs('./goodreads/' + str(fold) + '/eval/')

    """
    Partition into learn and eval sets
    """

    for setting in ['learn', 'eval']:
        """
        Partition into target and observed movie item and set ratings
        """
