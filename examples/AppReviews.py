import nltk
import numpy as np
import pandas as pd
import requests

def single_country_reviews(
    appID, 
    country='us',
    # Internal parameters
    page=1, 
    df=pd.DataFrame()):
    """
    Gets pandas DataFrame of app store reviews based on an ID
    Recursively descends pages on itself to get all reviews for a country

    ARGUMENTS
    ----------
    appID: int
        apple ID of the app. Generally 9 digit number
        Can be found in the url of a store page after "/id="
    Country: string of length 2
        apple country ID
        the list of country IDs can be found in a dict
    RETURNS:
    ---------
    Pandas DataFrame formatted: 
        [app_name, title, game version, rating, review, review votes]

    INTERNAL ARGUMENTS:
    -------------------
    page: int
        Page number to start the request at
        Looks recursively until it cannot find a page

    df: pandas DataFrame
        dataframe from previous page. final df is appended from each page        
    """
    url = 'https://itunes.apple.com/' + country \
          + '/rss/customerreviews/id=%s/page=%d/sortby=mostrecent/json' \
          % (appID, page)

    req = requests.get(url)

    try:
        data = req.json().get('feed')
        page_error = False
    except ValueError:
        return df.reset_index(drop=True)

    try:
        df_index = np.arange(len(data.get('entry')))
    except:
        return df.reset_index(drop=True)

    csvTitles = ['app_name', 'title', 'version', 'rating', 'review', 'vote_count']
    page_df = pd.DataFrame(index=df_index, columns=csvTitles)

    entry_index = -1  # DataFrame Index
    try:
        for entry in data.get('entry'):
            # There's app info @ top of JSON
            if entry.get('im:name'):
                page_df.app_name = entry.get('im:name').get('label')
                continue
            entry_index += 1
            page_df.title.loc[entry_index] = entry.get('title').get('label')
            page_df.version.loc[entry_index] = entry.get('im:version').get('label')
            page_df.rating.loc[entry_index] = entry.get('im:rating').get('label')
            page_df.review.loc[entry_index] = entry.get('content').get('label')
            page_df.vote_count.loc[entry_index] = entry.get('im:voteCount').get('label')
    except AttributeError as ae:
        print(data, "\n\n")
        raise(ae)


    # Clean up returned values
    page_df.dropna(axis=0, how='all', inplace=True)
    page_df.fillna({'title' : '', 'review' : '', 'version': ''})
    page_df.rating = pd.to_numeric(page_df.rating, 
                                   downcast='unsigned', 
                                   errors='coerce')
    page_df.vote_count = pd.to_numeric(page_df.vote_count, 
                                       downcast='unsigned', 
                                       errors='coerce')
    
    
    if not page_error:
        return single_country_reviews(
            appID,
            country=country,
            page=page + 1,
            df=df.append(page_df).reset_index(drop=True))


def get_reviews(
    appID, 
    list_countries=['us', 'gb', 'ca', 'au', 'ie', 'nz']):
    """
    Gets single pandas dataframe from multiple countries

    ARGUMENTS
    ----------
    appID: int
        apple ID of the app. Generally 9 digit number
        Can be found in the url of a store page after "/id="
    Country: list of apple country ID strings
        the list of country IDs can be found in a dict
    RETURNS:
    ---------
    Pandas DataFrame formatted: 
        [title, game version, rating, review, review votes]   
    """
    if type(list_countries) == str:
        list_countries = [list_countries]

    df = pd.DataFrame()
    
    for country in list_countries:
        df=df.append(
            single_country_reviews(appID, country=country)
            ).reset_index(drop=True)
    return df



def break_sentence(review):
    """
    Given a text, returns adjectives and related noun
    Text is possibly multiple sentences

    ARGUMENTS
    -----------
    review:
        single string. Body of text to tokenize

    RETURNS
    ---------
    list of strings. 
    Each string is a n-gram with a single nouns and multiple qualifiers
    n is variable (multiple adjectives can modify the same noun)
    """
    returned_token_list = list()
    sentences = nltk.sent_tokenize(review)
    for sent in sentences:       
        sentence_structure = dict()     
        words = nltk.pos_tag(word_tokenize(sent))
        # note: adjectives is ordered by sentence
        adjectives = [words.index(adj) 
                      for adj in words 
                      if adj[1] in ['JJ', 'JJR', 'JJS', 'VBP']]
        nouns = [words.index(adj) 
                 for adj in words 
                 if adj[1] in ['NN', 'NNP', 'NNPS', 'NNS']]
        closest_nouns = [(adj - np.array(nouns)).argmin() for adj in adjectives]
        for pos in range(len(adjectives)):
            sentence_structure.setdefault(closest_nouns[pos], [])
            sentence_structure[closest_nouns[pos]].append(adjectives[pos])
        for noun in sentence_structure:
            returned_token_list.append(
                str(' '.join([words[adj][0] for adj in sentence_structure[noun]])
                    + ' ' 
                    + words[nouns[noun]][0])
            )
    return returned_token_list


def outputTopics(reviews, n_topics=20, excluded_words=[], n_top_words=13):
    """
    Print topic top words
    Uses LDA to get topic clusters 
    outputs top words from each
    """
    word_count_model = CountVectorizer(stop_words='english')
    word_bag2 = word_count_model.fit_transform(reviews)

    vocab = word_count_model.get_feature_names()

    model = sklearn.decomposition.LatentDirichletAllocation(
                n_topics=n_topics, learning_method ='batch', 
                random_state=1)
    model.fit(word_bag2)
    topic_word = model.components_
    
    # Output top topic words
    for i, topic_dist in enumerate(topic_word):
        topic_words = list(np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words):-1])
        for drop_word in excluded_words:
            try: topic_words.remove(drop_word)
            except: pass
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
