from copy import deepcopy
import gensim
import numpy as np
import pandas as pd
import random
import unittest
from unittest import TestCase

try:
    import embedding
    from embedding import mergeEmbeddings
    from stringprocessing import tokenizeCol, tokenize
    import stringprocessing
    import mergingmethods
except ModuleNotFoundError:
    # Because the module is in a parent folder and not necessarily in our path
    # this will get the path to this file and add its parent to path
    # Lets the tests be run from any environment
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
    import embedding
    from embedding import mergeEmbeddings
    from stringprocessing import tokenizeCol, tokenize
    import stringprocessing
    import mergingmethods


class TestIsolatedStringProcessing(TestCase):
    """
    basic testing of vectorized document cleaning -> splitting
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = [
            ["157681 Miles. Only $12,849", 
             "Used 2006 Ford F-150"],
            ["\"city, state\" format (quotes)", 
             "different 'quotes'"],
        ]
        for i in range(len(tdata)):
            tdata[i] = pd.Series(tdata[i])
        removeCharDict = deepcopy(stringprocessing.REMOVE_CHAR)
        for i in range(0, 10):
            removeCharDict[str(i)] = "#"
        identityOp = lambda x: tokenizeCol(
            x, split=False, nanValue=np.nan, lower=False, replaceDict={})
        lowerSplit = lambda x: tokenizeCol(
            x, split=True, nanValue='', lower=True, replaceDict={})
        replaceNumerics = lambda x: tokenizeCol(
            x, lower=True, nanValue='', split=True, 
            replaceDict=removeCharDict)

        for i in tdata:
            # The function with everything off returns the same data
            self.assertTrue(identityOp(i).equals(i))
        
        # whitespace splitting and lowercasing works as expected
        self.assertTrue(lowerSplit(tdata[0]).equals(
            pd.Series([["157681", "miles.", "only", "$12,849"], 
                        ["used", "2006", "ford", "f-150"]])
        ), msg=lowerSplit(tdata[0]))
        self.assertTrue(lowerSplit(tdata[1]).equals(
            pd.Series([["\"city,", "state\"", "format", "(quotes)"], 
                        ["different", "'quotes'"]])
        ), msg=lowerSplit(tdata[1]))
        # replacing punctuation & numerics -> #
        self.assertTrue(replaceNumerics(tdata[0]).equals(
            pd.Series([["######", "miles", "only", "#####"], 
                        ["used", "####", "ford", "f", "###"]]),
        ), msg=replaceNumerics(tdata[0]))
        self.assertTrue(replaceNumerics(tdata[1]).equals(
            pd.Series([["city", "state", "format", "quotes"], 
                        ["different", "quotes"]]),
        ), msg=replaceNumerics(tdata[1]))
    
    def test_edge_cases(self):
        tdata = [
            "157681 Miles. Only $12,849", 
            "Used 2006 Ford F-150",
            "\"city, state\" format (quotes)", 
            "different 'quotes'",
            [],
            np.nan,
            ""]

        # passing as list or as pd.Series should be the same
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenizeCol(pd.Series(tdata))))

        resultLen = list(tokenizeCol(tdata).str.len())
        self.assertEqual(resultLen, [4, 5, 4, 2, 0, 0, 0])

    def test_tokenizing(self):
        """Test Inputs in tokenize method"""
        tdata = [
            "157681 Miles. Only $12,849", 
            "Used 2006 Ford F-150",
            "\"city, state\" format (quotes)", 
            "different 'quotes'",
            [],
            np.nan,
            ""]
        # passing a single column to tokenize should do the same as tokenizeCol
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenize(pd.Series(tdata))))
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenize(tdata)))
        # result is equal to concatenating several tokenizeCol results
        cat_res = tokenizeCol(tdata) + tokenizeCol(tdata)
        self.assertTrue(cat_res.equals(
                        tokenize([tdata, tdata])))
        # DataFrame input works the same as above
        self.assertTrue(cat_res.equals(
                        tokenize(pd.DataFrame([tdata, tdata]).T)))
# Input data for embedding methods below
def given_test_embeddings():
    return np.array([[1, -2, 3.5, 1], 
                    [0.4, 5e12, 0, 1], 
                    [0.0, 8, 0.000001, 1]])
def given_test_weights():
    return [
        [1.,1.,1.],
        [0.,0.,0.],
        [0.1, 0.2, 0.3],
        [2., 2., 2.]
    ]

class testEmbeddingMerging(TestCase):
    """
    basic testing of the mergeEmbeddings function
    """
    def test_given_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mergeEmbeddings([])

    def test_given_normal_input_MergesEmbeddings_normally(self):
        tembeddings = given_test_embeddings()
        tweights = given_test_weights()

        # default weight behavior is multiplication on np.sum
        # and multiplying by weights of 1 is same as not having weights
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings),
            mergeEmbeddings(tembeddings, tweights[0]))

        # passing weights in default behavior works
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(2 * tembeddings),
            mergeEmbeddings(tembeddings, tweights[3]))

        # Assure that passing weights on default behavior doesn't delete them
        # as the weights = None line
        self.assertIsNotNone(tweights[0])

        # passing empty weight vector raises error
        with self.assertRaises(ValueError):
            mergeEmbeddings(tembeddings, [])
        
        # tautological check on numpy mean function
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, method=np.mean, axis=0),
            np.mean(tembeddings, axis=0))

        # Same with different func signature (for kwargs messiness)
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, None, np.mean, axis=0),
            np.mean(tembeddings, axis=0))

    def test_given_normal_input_predefined_funcs_merge_correctly(self):
        tembeddings = given_test_embeddings()
        tweights = given_test_weights()
        # avgMerge function works as expected
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, None, mergingmethods.avgMerge),
            np.mean(tembeddings, axis=0))
        
        # weighed average
        # note you need to multiply each row (word) by the weights
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, tweights[2], 
                            method=mergingmethods.avgMerge),
            np.mean(tembeddings * np.array(tweights[2])[:, np.newaxis], axis=0))


def given_tdocs():
    """test docs for groupedEmbedding"""
    return [
        ["157681", "miles.", "only", "$12,849"], 
        ["used", "2006", "ford", "f-150"],
        ["\"city,", "state\"", "format", "(quotes)"], 
        ["different", "'quotes'"],
        ["######", "miles", "only", "#####"], 
        ["used", "####", "ford", "f-###"],
        ["city", "state", "format", "quotes"], 
        ["different", "quotes"]]

def given_repeated_words():
    """doc with repeated words"""
    return [["test", "test", "test"],
           ["unique", "words", "except", "here"],
           ["test", "again", "here"]]

def given_tgroups():
    """test columns keys, indexed with tdocs"""
    return [1, 1, 1, 1, 2, 2, 3, 3]

def given_normalized_tweights():
    "test weights coindexed to tdocs"
    return [
        [0.5, 0.2, 0.2, 0.1], 
        [0.3, 0.3, 0.3, 0.1],
        [0.25, 0.25, 0.25, 0.25], 
        [1,  0],
        [0, 0.000001, 0.9, 0.999999], 
        [0.8, 0.05, 0.05, 0.1],
        [0.1, 0.7, 0.01, 0.19], 
        [0.5, 0.5]]

def given_tweights():
    "test weights coindexed to tdocs"
    return [
        [5, 2, 20, 10000], 
        [3, 3, 3, 1],
        [0.0000025, 0.25, 2500, 250], 
        [1,  0],
        [0, 0.000001, 9, 999999], 
        [8, 0.05, 0.05, 1],
        [1, 7, 0.000001, 19], 
        [5, 5.]]

class mockW2V(object):
    """Mock gensim.models.keyedvectors class"""
    def __init__(self, embeddingDict):
        """init based on a word embedding dict[str -> np.array]"""
        self.embeddings = embeddingDict
        self.words = list(embeddingDict.keys())
        self.vector_size = len(self.embeddings[self.words[0]])
        for word in self.words:
            assert len(self.embeddings[word]) == self.vector_size, (
                "vector length not the same fro all mock word2vec model input")

    def __getitem__(self, key):
        return self.embeddings[key]

    def remove(self, key):
        self.embeddings.pop(key)
        self.words = list(self.embeddings.keys())

    def keys(self):
        return self.words

def given_mock_keyedVectors():
    """
    mock a gensim keyedVector object 
    (base class of word2vec)
    """
    # concat list of lists to unique words
    all_words = list(set(sum(given_tdocs(), [])))
    mymock = {}
    for word in range(len(all_words)):
        mymock[all_words[word]] = np.array([0.1, 0.2, 0.3]) * word
    return mockW2V(mymock)

def given_incomplete_keyedVectors():
    """
    mock gensim keyedVector object with missing words
    """
    kv = given_mock_keyedVectors()
    for _ in range(4):
        kv.remove(random.choice(list(kv.keys())))
    return kv


class TestWordWeights(TestCase):
    """
    tests for the getWordWeights function

    Note: can't test tf-idf method by composing tf and idf methods
    since there is internal normalizing and SMART value differences
    https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
    """
    def test_given_unique_words_per_doc_tf_and_idf_normalize_equal(self):
        tdocs = [["157681", "miles.", "only", "$12,849"], 
                ["used", "2006", "ford", "f-150"],
                ["\"city,", "state\"", "format", "(quotes)"], 
                ["different", "'quotes'"],
                ["######", "miles", "only", "#####"], 
                ["used", "####", "ford", "f-###"],
                ["city", "state", "format", "quotes"], 
                ["different", "quotes"]]
        normalizedtf = embedding.getWordWeights(tdocs, "tf", True)
        normalizeddf = embedding.getWordWeights(tdocs, "df", True)
        for i in range(len(normalizedtf)):
            np.testing.assert_array_almost_equal(
                normalizedtf[i], normalizeddf[i])

    def test_given_wordlist_tf_idf_count_normally(self):
        tdocs= given_repeated_words()
        tf = embedding.getWordWeights(tdocs, "tf", False)
        np.testing.assert_array_almost_equal(tf[0], [4, 4, 4])
        np.testing.assert_array_almost_equal(tf[1], [1, 1, 1, 2])
        np.testing.assert_array_almost_equal(tf[2], [4, 1, 2])
        df = embedding.getWordWeights(tdocs, "df", False)
        np.testing.assert_array_almost_equal(df[0], [2, 2, 2])
        np.testing.assert_array_almost_equal(df[1], [1, 1, 1, 2])
        np.testing.assert_array_almost_equal(df[2], [2, 1, 2])

    def test_given_normalized_sums_1(self):
        normalizedtf = embedding.getWordWeights(given_tdocs(), "tf", True)
        normalizeddf = embedding.getWordWeights(given_tdocs(), "df", True)
        normalizeddf = embedding.getWordWeights(given_tdocs(), "idf", True)
        normalizedtfidf = embedding.getWordWeights(given_tdocs(), "tf-idf", True)
        for i in range(len(given_tdocs())):
            for res in [normalizedtf[i], normalizeddf[i], 
                        normalizeddf[i], normalizedtfidf[i]]:
                self.assertAlmostEqual(res.sum(), 1)

    def test_given_repeated_words_output_dim_ok(self):
        """repeated words break gensim pipeline by default
           check weights are OK here"""
        tdocs = given_repeated_words()
        for method in ["tf", "idf", "df", "tf-idf"]:
            ww = embedding.getWordWeights(tdocs, method)
            for line in range(len(tdocs)):
                self.assertEqual(len(ww[line]), len(tdocs[line]),
                    msg="weights output dim should be equal on repeated words")

class TestGroupedEmbedding(TestCase):
    """
    basic checks for groupedEmbedding function
    """    
    def test_given_list_input_input_untouched(self):
        tdocs = given_tdocs()
        tgroups = given_tgroups()
        w2vec = given_mock_keyedVectors()
        gembeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec)
        self.assertTrue(type(tdocs) == list and type(tgroups) == list,
                        msg="input should be unchanged")
        self.assertEqual(len(gembeddings), len(set(tgroups)),
                        msg="groupe dict should be # of categories")

    def test_give_input_functions_applies_correctly(self):
        """
        test function from mergingmethods.py works with tautological equivalent
        """
        tdocs = given_tdocs()
        tgroups = given_tgroups()
        w2vec = given_mock_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=np.mean,
            word2SentenceKwargs={"axis": 0},
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0})
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=mergingmethods.avgMerge,
            sentence2GroupMerge=mergingmethods.avgMerge)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key], 
                meanEmbeddings[key])

    def test_given_oov_applies_correctly(self):
        """
        Test that OOV words apply equally and don't break pipeline
        """
        tdocs = given_tdocs()
        tgroups = given_tgroups()
        w2vec = given_incomplete_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=np.mean,
            word2SentenceKwargs={"axis": 0},
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0})
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=mergingmethods.avgMerge,
            sentence2GroupMerge=mergingmethods.avgMerge)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key],
                meanEmbeddings[key])

    def test_given_weights_tautological_works(self):
        """ 
        test that weights work normally in a tautological check
        """
        meanEmbeddings = embedding.groupedEmbedding(
            given_tdocs(), given_tgroups(), 
            model=given_incomplete_keyedVectors(), 
            weights=given_normalized_tweights(),
            word2SentenceMerge=np.sum,
            sentence2GroupMerge=np.sum)
        w2vec = given_incomplete_keyedVectors()
        for key in meanEmbeddings:
            self.assertEqual(len(meanEmbeddings[key]),
                             w2vec.vector_size)

    def test_given_weight_embedding_mismatch_raises(self):
        """weight dimension != embedding dimension should raise"""
        bad_weights = given_normalized_tweights()
        bad_weights[0] = bad_weights[0][:-1]
        bad_weights[1] = bad_weights[0][:-2]
        with self.assertRaises(ValueError):
            try:
                meanEmbeddings = embedding.groupedEmbedding(
                    given_tdocs(), given_tgroups(), 
                    model=given_incomplete_keyedVectors(), 
                    weights=bad_weights,
                    word2SentenceMerge=np.sum,
                    sentence2GroupMerge=np.sum)
                del meanEmbeddings # avoid pylint warning
            except IndexError: # can happen on edgecase of dropped word
                raise ValueError

if __name__ == '__main__':
    unittest.main()