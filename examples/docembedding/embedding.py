import collections
import gensim
import inspect
import numpy as np
import pandas as pd
from types import FunctionType
from typing import Iterable


def getWordWeights(wordListVec: Iterable[Iterable[str]],
                   method="tf-idf",
                   normalize=True
                   ) -> Iterable[Iterable[float]]:
    """
    TODO (mranger): refactor!! remove gensim dependencies
                    they complicate most of the logic for little gain
                    Make sure new scaling is good, too
    TODO (mranger): add support for external corpus
                    eg. get stat on original corpus first
                    then add stats of current corpus on top of it
                    to correct for w2vec training biases with weights

    Produce weights for words from a list of lists of words.
    The resulting shape is the same as wordListVec (eg. 1 weight per word)
    The weights are created only taking the wordListVec input into account
    (for now)

    :param wordListVec:
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
    :param method:
        the weighing scheme. Should be in:
            'tf': the word count in the corpus
            'df': the number of documents the word appears in
            'idf': IDF part of tf-idf. Equal to log(num_documents/df)
            'tf-idf': Standard tf-idf weighing scheme
        NOTE: tf-idf != (tf / idf) from the independent methods
            there is internal normalizing and SMART value differences
            https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
    :param normalize:
        whether to normalize the output in each document
        normalizing by weights = weights / weights.sum()

    :return: an vector of vectors of weights matching each word in each document
    :raises: ValueError
    """
    valid_methods = ['tf-idf', 'df', 'idf', 'tf']
    if method not in valid_methods:
        raise ValueError(
            "Invalid method '{0}' in getWordWeights."
             "valid methods are {1}".format(
                 method, ', '.join(valid_methods)))
    if method == 'tf':
        wordCount = collections.Counter(sum(wordListVec, []))
        res = []
        if normalize:
            for sentence in wordListVec:
                w = np.array([wordCount[word] for word in sentence])
                res.append(w / w.sum())
        else:
            for sentence in wordListVec:
                res.append(np.array([wordCount[word] for word in sentence]))
        return res
    # Cases with gensim dependencies
    # TODO: refactor this
    # gensim harms more than it helps to get these stats
    dct = gensim.corpora.Dictionary(wordListVec)
    weightTupleList = None
    if method == 'tf-idf':
        corpus = [dct.doc2bow(line) for line in wordListVec]
        tfidfTuples = gensim.models.TfidfModel(corpus)
        weightTupleList = []
        for line in wordListVec:
            wordFreq = collections.Counter(line)
            wordids = [wordId for wordId in dct.doc2idx(line) if wordId > -1]
            wordFreqs = [wordFreq[word] for word in line]
            bagOfWords = list(zip(wordids, wordFreqs))
            weightTupleList.append(tfidfTuples[bagOfWords])
    elif method in ['df', 'idf']:
        weightTupleList = []
        for line in wordListVec:
            wordIDs = [wordId
                       for wordId in dct.doc2idx(line) 
                       if wordId > -1]
            if method =='df':
                weightTupleList.append([
                    (wordId, dct.dfs[wordId])
                    for wordId in wordIDs])
            elif method =='idf':
                weightTupleList.append([
                    (wordId, 
                    np.log(dct.num_docs / dct.dfs[wordId]))
                    for wordId in wordIDs])
    res = []
    if normalize:
        for wordlist in weightTupleList:
            w = np.array([weightTuple[1] for weightTuple in wordlist])
            res.append(w / w.sum())
    else:
        for wordlist in weightTupleList:
            res.append(np.array([weightTuple[1]
                        for weightTuple in wordlist]))
    return res 


def OOVWords(model: gensim.models.keyedvectors,
             wordListVec: Iterable[Iterable[str]]) -> dict:
    """
    TODO (mranger): integration test me
    Gets words out of Word2Vec model vocabulary

    :param model: a trained Gensim keyedVector model (usually word2vec)
    :param wordListVec: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
        If they aren't, use stringprocessing.tokenizeCol beforehand
    :type model: gensim.models.keyedvectors
    :type wordListVec: Iterable[str]

    :return: a dictionary mapping OOV words to their number of occurences
            and a list of the rows where they appear
    :rtype: dict(word -> [number occurences, [row locations]])
    """
    # return value of missing words dict
    missingWords = collections.defaultdict(lambda: [0, []])
    for row_num in range(len(wordListVec)):
        row = wordListVec[row_num]
        for word in row:
            try:
                model[word]
            except KeyError:
                missingWords[word][0] += 1
                missingWords[word][1].append(row_num)
    return dict(missingWords)


def mergeEmbeddings(embeddings: Iterable[Iterable[float]],
                    weights: Iterable[float]=None,
                    method: FunctionType=np.sum,
                    **kwargs) -> np.array:
    """
    Takes a list of embeddings and merges them into a single embeddings.
    Typical usecase is to create a sentence embedding from word embeddings.

    :param embeddings:
        the embeddings. for instance, an array of word embeddings
        note that weird numbers (np.NaN, np.inf, etc.) are undefined behavior
    :param weights: 
        weights on the embeddings, to be used in the method called
        weird numbers are undefined behavior here too
    :param method:
        A method is passed that takes either the format 
            Iterable[Iterable[float]], **kwargs 
        eg. the embeddings and of keyword arguments
        or the format 
            Iterable[Iterable[float]], Iterable[float], **kwargs
        which is (embeddings, weights, {keyword arguments})
    
        Methods include:
            "avg": method=np.sum, axis=0
                elementwise mean of word vectors. 
                This can be quality-degrading due to semantically meaningless 
                components having large values (see ICLR (2017), p.2, par.4)
            "sum": method=np.mean, axis=0
                elementwise sum of word vectors. 
                Also called "word centroid" in literature

    :return: a single array merged by the procedure
    :rtype: np.array[float]
    :raises: ValueError
    ------------------------------------------------------------------
    references:
        "Simple but tough to beat baseline", Arora et al. (ICLR 2017)
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings input empty!")
    if weights is not None:
        if type(weights) != np.array:
            weights = np.array(weights)
        if len(weights) != len(embeddings):
            raise ValueError(
                "Incorrect # of weights! {0} weights, {1} embeddings".format(
                    len(weights), len(embeddings)))
    if method is None:
        method = np.sum
    # sane default behavior is "sum"
    if not kwargs and method == np.sum:
        if weights is not None:
            embeddings = embeddings * weights[:, np.newaxis]
            # This should not delete the actual value, 
            # only the local name (unit tested)
            weights = None
        kwargs['axis'] = 0
    try:
        if weights is None:
            return method(embeddings, **kwargs)
        return method(embeddings, weights, **kwargs)
    except TypeError as te:
        print(("\n\nError calling defined method.\n "
               + "method called: {0}\n").format(method),
            "\n\nNOTE: This can happen if you are passing weights "
            "in a function that doesn't take them as the second argument!\n"
            "Function signature was:\n\t {0}".format(inspect.signature(method)),
            ("\nArgs passed were:"
              + "\n\tembeddings: {0}"
              + "\n\tweights: {1}"
              + "\n\tkwargs: {2}").format(
                  embeddings, weights, kwargs))
        raise(te)


def sentenceEmbedding(documentCol: Iterable[Iterable[str]],
                      model: gensim.models.keyedvectors,
                        weights: Iterable[Iterable[float]]=None,
                        mergingFunction: FunctionType=np.sum,
                        mergingFunctionKwargs: dict={}
                        )-> np.array:
    """
    Merge a vector of sentences (which are already split into words)
        into a sentence embedding. 

    Methods for each embedding merging methods are input as params. 
    See mergingFunction param for reference

    NOTE: This method expects the document column to be already split 
        and preprocessed (so in series of ['word', 'another', 'word'] format)

    :param documentCol: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
        gets converted to a pd.series before treating
    :param model:
        a trained Gensim keyedVector model (usually word2vec)
    :param weights:
        weights on each words, to be used in the embedding merging method called
    :param mergingFunction:
        a function to merge word embeddings into sentence embeddings
        passed to mergeEmbeddings.
        function is usually from mergingmethods.py
        Should respect the require function signature
            (embeddings, weights, {keyword arguments}) if weights are passed
            (embeddings, {keyword arguments}) if no weights are passed
    :param mergingFunctionKwargs:
        keyword arguments for above function

    :return: a matrix where each row is the sentence embedding of the corresponding
             row in documentCol.
    """
    if type(documentCol) != pd.Series:
        documentCol = pd.Series(documentCol)
    if weights is not None and type(weights) != pd.Series:
        weights = pd.Series(weights)
    SentenceEmbeddings = np.zeros((len(documentCol), model.vector_size))
    for row in range(len(documentCol)):
        document = documentCol.iloc[row]
        wordWeights = np.array(weights.iloc[row]) if weights is not None else None
        if wordWeights is not None and len(wordWeights) != len(document):
            raise ValueError(
                ("Incorrect # of weights on row {0} Weights:\n{1}\nWords:\n{2}"
                ).format(row, wordWeights, document))
        # pre allocate word embedding matrix for sentence 
        sentenceWords = np.zeros((len(document), model.vector_size))
        oovIndeces = []
        for word in range(len(document)):
            try:
                sentenceWords[word] = model[document[word]]
            except KeyError:
                oovIndeces.append(word)
        if oovIndeces: # if oov words, filter result
            allIndices = np.indices((len(document),))
            notOOVindices = np.setxor1d(allIndices, oovIndeces)
            sentenceWords = sentenceWords[[notOOVindices]]
            if weights is not None:
                try:
                    wordWeights = wordWeights[[notOOVindices]]
                except IndexError:
                    print(("Index error on: \n\t{0}\n with weights: \n\t {1}"
                        "\nDropped Indices: {2}"
                        ).format(document, weights[row], oovIndeces))
                    raise IndexError
        # edge cases (0 or 1 word sentence)
        if len(sentenceWords) == 0:
            continue
        elif len(sentenceWords) == 1:
            SentenceEmbeddings[row] = sentenceWords[0]
        else:
            SentenceEmbeddings[row] = mergeEmbeddings(
                    sentenceWords,
                    weights=wordWeights,
                    method=mergingFunction,
                    **mergingFunctionKwargs)
    return SentenceEmbeddings


def groupedEmbedding(documentCol: Iterable[Iterable[str]],
                     groupKeyCol: Iterable[int],
                     model: gensim.models.keyedvectors,
                     weights: Iterable[Iterable[float]]=None,
                     word2SentenceMerge: FunctionType=np.sum,
                     word2SentenceKwargs: dict={},
                     sentence2GroupMerge: FunctionType=np.sum,
                     sentence2GroupKwargs: dict={},
                     verbose=True
                    ) -> dict:
    """
    Creates embeddings for Groups of documents
    Does this by first embedding the documents 
        then embedding each document embedding into a single embedding per group

    For instance, you can create paragraph embeddings by passing a list of split 
    sentences with groupkeycol being the paragraph number on each sentence.

    Methods for each embedding merging methods are input as params

    NOTE: This method expects the document column to be already split 
        and preprocessed (so in series of ['word', 'another', 'word'] format)

    returns a dictionary of each groupkey -> group embedding

    :param documentCol: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
        gets converted to a pd.series before treating
    :param groupKeyCol:
        a vector of group keys co-indexed with each document in documentCol.
        This could be each paragraph a document belongs to, or author, etc.
    :param model:
        a trained Gensim keyedVector model (usually word2vec)
    :param weights:
        weights on each words, to be used in the embedding merging method called
    :param word2SentenceMerge:
        a function to merge word embeddings into sentence embeddings
        passed to mergeEmbeddings.
        function is usually from mergingmethods.py
        Should respect the require function signature
            (embeddings, weights, {keyword arguments}) if weights are passed
            (embeddings, {keyword arguments}) if no weights are passed
    :param word2SentenceKwargs:
        keyword arguments for above function
    :param sentence2GroupMerge:
        function to merge sentence embeddings into grouped embeddings
        same requirements as word2SentenceMerge functions
    :param sentence2GroupKwargs:
        kwargs for above
    :param verbose:
        whether to print info output

    :return: a dict[group] -> group_embedding 
             for all groups in groupKeyCol
             where embeddings are merged from words to sentence
             then from sentence to group
    """
    if type(documentCol) != pd.Series:
        documentCol = pd.Series(documentCol)
    if type(groupKeyCol) != pd.Series:
        groupKeyCol = pd.Series(groupKeyCol)
    if weights is not None and type(weights) != pd.Series:
        weights = pd.Series(weights)
    if len(documentCol) != len(groupKeyCol):
        raise ValueError("documentCol and groupKeycol should be co-indexed!"
                         "groupKeyCol is the group for each coindexed document")
    elif weights is not None and len(documentCol) != len(weights):
        raise ValueError("documentCol, weights should be co-indexed!"
                         "weights are weights for each word in coindexed document")
    groupEmbeddings = {}
    for group in groupKeyCol.unique():
        # find documents for the group
        indices = documentCol.index[groupKeyCol == group].tolist()
        SentenceEmbeddings = sentenceEmbedding(
            documentCol.iloc[indices],
            model,
            weights.iloc[indices] if weights is not None else None,
            mergingFunction=word2SentenceMerge,
            mergingFunctionKwargs=word2SentenceKwargs)
        try:
            groupEmbeddings[group] = mergeEmbeddings(
                SentenceEmbeddings,
                weights=None, # TODO (mranger): add 2nd level weights fn??
                method=sentence2GroupMerge,
                **sentence2GroupKwargs)
        except ValueError as ve:
            if verbose:
                print("Bad sentence Embeddings: {1}".format(SentenceEmbeddings))
                print(ve)
            pass
    return groupEmbeddings