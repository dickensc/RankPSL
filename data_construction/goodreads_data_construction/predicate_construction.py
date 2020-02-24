"""
This script constructs the necessary data files needed for psl model
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


def construct_predicates(books_df, interactions_df, reviews_df,
                         obs_interactions, obs_reviews,
                         target_interactions, target_reviews,
                         truth_interactions, truth_reviews,
                         fold, setting):
    """

    :param books_df:
    :param interactions_df:
    :param reviews_df:
    :param obs_interactions:
    :param obs_reviews:
    :param target_interactions:
    :param target_reviews:
    :param truth_interactions:
    :param truth_reviews:
    :param fold:
    :param setting:
    :return:
    """

    """
    Create data directory to write output to
    """
    if not os.path.exists('./goodreads/' + str(fold)):
        os.makedirs('./goodreads/' + str(fold))

    if not os.path.exists('./goodreads/' + str(fold) + '/learn/'):
        os.makedirs('./goodreads/' + str(fold) + '/learn/')

    if not os.path.exists('./goodreads/' + str(fold) + '/eval/'):
        os.makedirs('./goodreads/' + str(fold) + '/eval/')

    # scoping predicates
    targeted_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    authored_predicate(books_df, fold, setting)
    series_predicate(books_df, fold, setting)
    genre_predicate(books_df, fold, setting)

    # feedback Predicates
    shelved_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    read_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    rating_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    review_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    preference_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)

    # similarity blocking predicates
    item_collab_filter_cosine_predicate(interactions_df, obs_interactions, target_interactions,
                                        truth_interactions, fold, setting)
    user_collab_filter_cosine_predicate(interactions_df, obs_interactions, target_interactions,
                                        truth_interactions, fold, setting)


def shelved_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/shelved_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    shelved_series = pd.Series(data=1, index=obs_interactions, name='shelved')
    write(shelved_series, partition)

    # truth predicates
    partition = 'truth'
    shelved_series = pd.Series(data=1, index=truth_interactions, name='shelved')
    write(shelved_series, partition)

    # target predicates
    partition = 'targets'
    shelved_df = pd.DataFrame(index=target_interactions, columns=['shelved'])
    write(shelved_df, partition)


def read_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/read_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    read_interactions = observed_interactions_df[observed_interactions_df.is_read]
    read_series = pd.Series(data=1, index=read_interactions.index, name='read')
    write(read_series, partition)

    # truth predicates
    partition = 'truth'
    truth_interactions_df = interactions_df.loc[truth_interactions, :]
    read_interactions = truth_interactions_df[truth_interactions_df.is_read]
    read_series = pd.Series(data=1, index=read_interactions.index, name='read')
    write(read_series, partition)

    # target predicates
    partition = 'targets'
    read_df = pd.DataFrame(index=target_interactions, columns=['read'])
    write(read_df, partition)


def rating_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/rating_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    rating_series = observed_interactions_df[observed_interactions_df.rating > 0]['rating']
    rating_series = rating_series - rating_series.min()
    rating_series = rating_series / rating_series.max()
    write(rating_series, partition)

    # truth predicates
    partition = 'truth'
    truth_interactions_df = interactions_df.loc[truth_interactions, :]
    rating_series = truth_interactions_df[truth_interactions_df.rating > 0]['rating']
    rating_series = rating_series - rating_series.min()
    rating_series = rating_series / rating_series.max()
    write(rating_series, partition)

    # target predicates
    partition = 'targets'
    rating_df = pd.DataFrame(index=target_interactions, columns=['read'])
    write(rating_df, partition)


def review_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/review_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    review_interactions = observed_interactions_df[observed_interactions_df.review_text_incomplete == '']
    review_series = pd.Series(data=1, index=review_interactions.index, name='reviewed')
    write(review_series, partition)

    # truth predicates
    partition = 'truth'
    truth_interactions_df = interactions_df.loc[truth_interactions, :]
    review_interactions = truth_interactions_df[truth_interactions_df.review_text_incomplete == '']
    review_series = pd.Series(data=1, index=review_interactions.index, name='reviewed')
    write(review_series, partition)

    # target predicates
    partition = 'targets'
    review_df = pd.DataFrame(index=target_interactions, columns=['reviewed'])
    write(review_df, partition)


def preference_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/preference_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # target predicates
    partition = 'targets'
    preference_df = pd.DataFrame(index=target_interactions, columns=['reviewed'])
    write(preference_df, partition)


def item_collab_filter_cosine_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """

    sim_threshold = 0.25

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/item_collab_filter_cosine_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    # similarity based on ratings
    user_book_ratings = observed_interactions_df['rating'].unstack().fillna(0)
    similarity_matrix = cosine_similarity(user_book_ratings.transpose())
    item_item_sim = pd.DataFrame(similarity_matrix, index=user_book_ratings.columns, columns=user_book_ratings.columns)
    item_item_sim = item_item_sim[item_item_sim > sim_threshold].stack()
    item_item_series = pd.Series(data=1, index=item_item_sim.index)
    write(item_item_series, partition)


def user_collab_filter_cosine_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """
    sim_threshold = 0.25

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/user_collab_filter_cosine_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    # similarity based on ratings
    user_book_ratings = observed_interactions_df['rating'].unstack().fillna(0)
    similarity_matrix = cosine_similarity(user_book_ratings)
    user_user_sim = pd.DataFrame(similarity_matrix, index=user_book_ratings.index, columns=user_book_ratings.index)
    user_user_sim = user_user_sim[user_user_sim > sim_threshold].stack()
    user_user_series = pd.Series(data=1, index=user_user_sim.index)
    write(user_user_series, partition)


def targeted_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/targeted_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    targeted_df = pd.DataFrame(data=1, index=target_interactions, columns=['targeted'])
    write(targeted_df, partition)


def authored_predicate(books_df, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/authored_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    # create author to book data frame
    author_df = pd.DataFrame(columns=books_df.index)
    author_df.index.name = 'author_id'
    for book_id, book in books_df.iterrows():
        for author in book.authors:
            author_df.loc[author['author_id'], book.name] = 1

    authored_series = author_df.stack().swaplevel()
    write(authored_series, partition)


def series_predicate(books_df, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/series_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    # create series to book data frame
    series_df = pd.DataFrame(columns=books_df.index)
    series_df.index.name = 'series'
    for book_id, book in books_df.iterrows():
        for series in book.series:
            series_df.loc[series, book.name] = 1

    series_series = series_df.stack().swaplevel()
    write(series_series, partition)


def genre_predicate(books_df, fold, setting):
    """

    :param interactions_df:
    :param partition:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/genre_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    # create genre to book data frame, genre is from shelf names by users
    max_genres = 3
    genre_df = pd.DataFrame(columns=books_df.index)
    genre_df.index.name = 'genre'
    for book_id, book in books_df.iterrows():
        n_genres = 0
        for genre in book.popular_shelves:
            if n_genres >= 3:
                break
            else:
                if genre['name'] != 'to-read':
                    genre_df.loc[genre['name'], book.name] = 1
                    n_genres = n_genres + 1

    genre_series = genre_df.stack().swaplevel()
    write(genre_series, partition)
