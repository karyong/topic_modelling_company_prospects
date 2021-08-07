# Virtual env name: gensim_lda_env
# Run in anaconda prompt:
# cd "E:/PI/2021/Investment_paper/"
# type e: to change directory to external drive

# -- Load packages --
# Run in python console (only need to do once)
# import nltk; nltk.download('stopwords')

# Run in anaconda/ command prompt
# python3 -m spacy download en
# conda install --file requirements.txt


# -- Tokenize and clean up text --
# function to tokenize and clean up text
def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc = True)) # deacc removes punctuations
        
# -- Remove stopwords, make bigrams and lemmatize --
## define functions
# stop_words
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# make bigrams
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# lemmatize
def lemmatization(texts, allowed_postags = ['NOUN', 'ADJ', 'ADV', 'VERB']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# -- Calculate weighted word frequency --
# this is different from tf-idf. In fact, it is the opposite: (word_count * doc_freq/num_docs). Words that appear in more docs have higher values
def weighted_freq(processed_text):
    # count word freq
    allwords = []
    for doc in data_lemmatize:
        allwords.extend(doc)
    wordcounts = collections.Counter(allwords)
    # print(wordcounts.most_common(30))

    # calculate document frequency - num of docs in which a word appears
    DF = {}
    for i in range(len(data_lemmatize)):
        tokens = data_lemmatize[i]
        for word in tokens:
            # get dict of each word and a list of docs in which the word appear
            try:
                DF[word].add(i)
            except:
                DF[word] = {i}
    # but we want the number of docs, instead of a list of docs
    for w in DF:
        DF[w] = len(DF[w])/len(processed_text)

    # combine and convert wordcounts and DF into a dataframe
    docfreq_df = pd.DataFrame.from_dict(DF, orient='index', columns = ['doc_freq'])
    wordcount_df = pd.DataFrame.from_dict(wordcounts, orient='index', columns = ['word_count'])
    df_merged = docfreq_df.join(wordcount_df) 
    df_merged['weighted_wordfreq'] = df_merged['doc_freq'] * df_merged['word_count']
    df_merged = df_merged.sort_values('weighted_wordfreq', ascending=False)

    return df_merged

# -- Choose optimal number of topics --
def compute_coherence_values(dictionary, corpus, k, a, b):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    k: Num of topics
    a: Dirichlet hyperparameter alpha (Document-Topic Density)
    b: Dirichlet hyperparameter beta (Word-Topic Density)

    Returns:
    -------
    coherence_values: Coherence values corresponding to the LDA model with respective num of topics
    """

    model = gensim.models.LdaMulticore(corpus=corpus,
    id2word = dictionary, num_topics = k, random_state=100,
    alpha=a, eta=b, passes=10, per_word_topics=True)

    coherencemodel = CoherenceModel(model=model, texts=data_lemmatize,dictionary=id2word, coherence='c_v')
    
    return coherencemodel.get_coherence()

# tune parameters
def tune_params(min_topics, max_topics, step_topics, min_alpha=0.01, max_alpha=1, step_alpha=0.3, min_beta=0.01, max_beta=1, step_beta=0.3):
    # tune the model
    import tqdm

    grid = {}
    grid['Validation_Set'] = {}

    # Topic range
    min_topics = min_topics
    max_topics = max_topics
    step_size = step_topics
    topics_range = range(min_topics, max_topics+1, step_size)

    # Alpha parameter
    alpha = list(np.arange(min_alpha, max_alpha, step_alpha))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(min_beta, max_beta, step_beta))
    beta.append('symmetric')

    # Validation sets
    # num_of_docs = len(corpus)
    # corpus_sets = [gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75),
    #                 corpus]

    # corpus_title = ['75% Corpus', '100% Corpus']

    model_results = {#'Validation_Set': [],
                    'Topics': [],
                    'Alpha': [],
                    'Beta': [],
                    'Coherence': []
                    }
                    
    # can take long time to run
    if 1 == 1:
        bar_len = len(alpha) * len(beta) * (np.floor((max_topics-min_topics)/step_topics) + 1)
        pbar = tqdm.tqdm(total = bar_len)

        # iterate through validation corpuses
        # for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterate through beta values
                for b in beta:
                    # get the coherence score for the given parameter
                    cv = compute_coherence_values(corpus=corpus, #corpus=corpus_sets[i],
                    dictionary=id2word, k=k, a=a, b=b
                    )

                    # save the model results
                    # model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)

    pbar.close()

    return model_results

# -- Find the dominant keywords for each doc --
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# put all my logics in the main block
if __name__ == '__main__':

    # -- Load packages --
    # import re
    import numpy as np
    import pandas as pd
    import collections

    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # spacy for lemmatization
    import spacy

    # Plotting tools
    # import pyLDAvis
    # import pyLDAvis.gensim  # don't skip this
    # import matplotlib.pyplot as plt
    # %matplotlib inline

    from pathlib import PureWindowsPath

    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    # add custom stop words to the list
    stop_words.extend(['business', 'businesses', 'group', 'groups'])

    # -- Read file --
    # to change to your path
    source_path = PureWindowsPath(r'E:\PI\2021\Investment_paper').as_posix()
    filepath = source_path + '/company_prospects.csv'
    df = pd.read_csv(filepath, encoding='cp1252')
    # df.head()
    # print(len(df))

    # -- Tokenize and clean up text --
    data = df['prospect'].values.tolist()
    # data

    data_words = list(sent_to_words(data))
    # print(data_words[:1])

    # -- Creating bigrams --
    # Build bigrams
    bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 100) # higher threshold fewer phrases
    # list(bigram[data_words])

    # Faster way to get a sentence clubbed as bigram/trigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # print(bigram_mod)

    # see bigram
    # print(bigram_mod[data_words[0]])

    # -- Remove stopwords, make bigrams and lemmatize --
    # remove stopwords
    data_words_nostops = remove_stopwords(data_words)
    
    # form bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en (run this in anaconda/command prompt)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # lemmatize
    data_lemmatize = lemmatization(data_words_bigrams)
        
    # -- Calculate weighted word frequency --
    weighted_freq(data_lemmatize).to_csv('weighted_word_freq.csv')

    # print("First doc lemmatize:\n")
    # print(data_lemmatize[0])

    # -- Create Dictionary and Corpus needed for topic modelling --
    # Create dictionary
    id2word = corpora.Dictionary(data_lemmatize)

    # create corpus
    texts = data_lemmatize

    # term document frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # print(corpus[:1]) # this prints the (word_id, freq) pairs
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]) # print human-readable format

    # -- not run (just for exploration) --
    # -- Build Topic Model --
    # # Build LDA model
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    # id2word = id2word, num_topics = 5, random_state=100, update_every = 1,
    # alpha='auto', passes=10, per_word_topics=True)

    # # view topics in LDA model
    # # print the keyword in the 5 topics
    # print(lda_model.print_topics())

    # # -- Model Perplexity and Coherence Score --
    # # to estimate how good a given topic model is

    # # Compute Perplexity
    # print('Perplexity: ', lda_model.log_perplexity(corpus)) # the lower the better

    # # Compute Coherence score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatize, dictionary=id2word, coherence='c_v')
    # print('\nCoherence Score: ', coherence_model_lda.get_coherence())

    # -- run this --
    # -- Choose optimal num of topics --
    model_results = tune_params(min_topics=2, max_topics=10, step_topics=1)
    pd.DataFrame(model_results).to_csv(source_path + '/lda_tuning_results.csv', index = False)

    # -- run this after choosing the optimal num of topics --
    # -- Final Model --
    # # num_topics = 4, alpha = 0.61, beta = 'symmetric', highest coherence = 0.5208 (second optimal num of topics is selected as optimal 6 is too many) 
    # optimal_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, 
    # num_topics=4, random_state=100, passes = 10, alpha=0.61, eta='symmetric')
    # print(optimal_model.print_topics())

    # -- not run --
    # -- Find the dominant keywords for each doc --
    # df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

    # # Format
    # df_dominant_topic = df_topic_sents_keywords.reset_index()
    # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # # Show
    # # df_dominant_topic.head(10)
    # df_dominant_topic.to_csv(source_path + '/dominant_topics.csv', index = False)
    
