import numpy as np
from typing import Iterable
##################################################################
#                                                                #
#                                                                #
#                INPUT FUNCTIONS for mergeEmbeddings             #
#                                                                #
#                                                                #
##################################################################

def avgMerge(embeddings: Iterable[Iterable[float]]=[],
             weights: Iterable[float]=None):
    """
    INPUT FUNCTION for mergeEmbeddings function
    Please refer to the docstring of mergeEmbeddings for more details

    this method is really just a wrapper around the call
        np.mean(a, axis=1, weights=weights)

    :param embeddings: the embeddings. for instance, an array of word embeddings
    :param weights: weights to do a weighted average.
                    Weights are on the words (eg. rows)

    :return: a single array of the merged embeddings
    :rtype: np.array[float]
    """
    if weights is not None:
        return np.mean(embeddings * weights[:, np.newaxis], axis=0)
    return np.mean(embeddings, axis=0)


def sumMerge(embeddings: Iterable[Iterable[float]]=[],
             weights: Iterable[float]=None):
    """
    INPUT FUNCTION for mergeEmbeddings function
    Please refer to the docstring of mergeEmbeddings for more details

    this method is really just a wrapper around the call
        np.sum(a, axis=1, weights=weights)

    :param embeddings: the embeddings. for instance, an array of word embeddings
    :param weights: weights to do a weighted average.
                    Weights are on the words (eg. rows)

    :return: a single array of the merged embeddings
    :rtype: np.array[float]
    """
    if weights is not None:
        return np.sum(embeddings * weights[:, np.newaxis], axis=0)
    return np.sum(embeddings, axis=0)

#
#
#
# TODO:
# SVD merging
# PCA merging
# Merging based on Simple baseline (Arora 2017)
# python 2.7 code to port here:
# https://github.com/PrincetonML/SIF
# https://github.com/PrincetonML/SIF_mini_demo
# other dimensionality reduction merging
#
#
#
#

