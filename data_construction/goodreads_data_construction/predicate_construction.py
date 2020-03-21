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
from sklearn.neighbors import NearestNeighbors


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

    # blocking predicates
    targeted_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    authored_predicate(books_df, fold, setting)
    series_predicate(books_df, fold, setting)
    genre_predicate(books_df, fold, setting)
    book_publisher_predicate(books_df, fold, setting)
    book_average_rating_predicate(books_df, fold, setting)
    
    # similarity calculation based blocking predicates
    item_collab_filter_cosine_predicate(interactions_df, books_df, obs_interactions, target_interactions,
                                        truth_interactions, fold, setting)
#     user_collab_filter_jaccard_predicate(interactions_df, books_df, obs_interactions, target_interactions,
#                                         truth_interactions, fold, setting)
    
    # local predictor predicates
    # TODO: 

    # feedback Predicates
    shelved_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    read_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    rating_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    review_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)
    preference_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting)


def shelved_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    A book is shelved if it is present in the interactions frame.
    :param interactions_df: Dataframe
    :param obs_interactions: index object 
    :param target_interactions: index object 
    :param truth_interactions: index object 
    :param fold: integer
    :param setting: string, either learn or eval
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
    shelved_df = pd.DataFrame(index=target_interactions)
    write(shelved_df, partition)


def read_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    A book is read if it is the is_read value is True in the interactions frame.
    :param interactions_df: Dataframe
    :param obs_interactions: index object 
    :param target_interactions: index object 
    :param truth_interactions: index object 
    :param fold: integer
    :param setting: string, either learn or eval
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
    missing_reads = observed_interactions_df[~observed_interactions_df.is_read].index
    augmented_targets = target_interactions.union(missing_reads)
    read_df = pd.DataFrame(index=augmented_targets)
    write(read_df, partition)


def rating_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    A user's book rating is given by the rating value in the interactions frame.
    We normalize the rating between 0 and 1 across all users.
    TODO: standardization should be on a per user level so every user rating is on the same scale.
          We would need to handle users who have only given ratings with a single value since the 
          variance would be 0 in that case, hence we would be dividiing by 0 and the world would 
          implode.
    :param interactions_df: Dataframe
    :param obs_interactions: index object 
    :param target_interactions: index object 
    :param truth_interactions: index object 
    :param fold: integer
    :param setting: string, either learn or eval
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
    missing_ratings = observed_interactions_df[observed_interactions_df.rating == 0].index
    augmented_targets = target_interactions.union(missing_ratings)
    rating_df = pd.DataFrame(index=augmented_targets)
    write(rating_df, partition)


def review_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    At this point the review_predicate is simply whether or not the user has given a review for a book.
    This is captured by the review_text_incomplete column in the interactions frames.
    TODO: incorporate sentiment analysis into these predicates.
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
    review_interactions = observed_interactions_df[observed_interactions_df.review_text_incomplete != '']
    review_series = pd.Series(data=1, index=review_interactions.index, name='reviewed')
    write(review_series, partition)

    # truth predicates
    partition = 'truth'
    truth_interactions_df = interactions_df.loc[truth_interactions, :]
    review_interactions = truth_interactions_df[truth_interactions_df.review_text_incomplete != '']
    review_series = pd.Series(data=1, index=review_interactions.index, name='reviewed')
    write(review_series, partition)

    # target predicates
    partition = 'targets'
    missing_reviews = observed_interactions_df[observed_interactions_df.review_text_incomplete == ''].index
    augmented_targets = target_interactions.union(missing_reviews)
    review_df = pd.DataFrame(index=augmented_targets)
    write(review_df, partition)


def preference_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    The predicate representing the latent preference of a user.
    At this point we only are inferring the preference of target user book pairs.
    TODO: Add method for growing the target set for this predicate beyond the target user book pairs
    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/preference_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # target predicates
    partition = 'targets'
    preference_df = pd.DataFrame(index=obs_interactions.union(target_interactions))
    write(preference_df, partition)


def item_collab_filter_cosine_predicate(interactions_df, books_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
    """
    For Item item collaborative filtering.
    Calculates the top k most similar items to another item based on user shelves
    To aviod having to calculate the n x n cross pruduct similarity, we instead block by language and genre
    :param interactions_df:
    :param obs_interactions:
    :param target_interactions:
    :param truth_interactions:
    :param fold:
    :param setting:
    :return:
    """

    k = 10

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/item_collab_filter_cosine_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]
    observed_books_df = books_df.loc[obs_interactions.get_level_values(1).unique()]

    # similarity based on shelves (<- chosen since it the most dense value)
    # TODO: incorporate sparser and less noising signals from user

    # index: user_ids columns: book_ids values: 1 if shelved by user 0 o.w.
    user_book_shelves = pd.Series(data=1, index=obs_interactions).unstack().fillna(0)
    # index: book_ids columns: book_ids
    item_item_sim = pd.DataFrame(data=0, index=user_book_shelves.columns, columns=user_book_shelves.columns)

    # block similarity calculations by language and by genre heuristics
    # use unique values of language codes to block similarity calculations: generalize by adding 
    #             support for clusters of languages, i.e. eng and can-en can be grouped together
    # use top-1 informative shelf name: generalize by using top-n informative shelf names

    # create genre to book data frame, genre is from shelf names by users
    # index: book_id, columns: genre
    genre_series = pd.Series(index=observed_books_df.index)
    for book_id, book in observed_books_df.iterrows():
        for i in range(len(book.popular_shelves)):
            # note that popular_shelves is sorted
            if book.popular_shelves[i]['name'] != 'to-read':
                genre_series.loc[book.name] = book.popular_shelves[i]['name']
                break

    # add genre series as column of observed_books_df
    observed_books_df.loc[:, 'genre'] = genre_series

    # block by both 'genre' and 'language_code' and calculate similarity for books within block
    for _, book_group in observed_books_df.groupby(['genre', 'language_code']):
        similarity_matrix = cosine_similarity(user_book_shelves.loc[:, book_group.index].transpose())
        item_item_sim.loc[book_group.index, book_group.index] = similarity_matrix
        
    # Very sparse, thus very small cosine sim values
    # approach: for each book use k most similar books to it
    book_top_k_sim = item_item_sim.apply(pd.Series.nlargest, n=5).stack()
    item_item_series = pd.Series(data=1, index=book_top_k_sim.index)
    write(item_item_series, partition)


# TODO: need structure across users to block
# def user_collab_filter_cosine_predicate(interactions_df, books_df, obs_interactions, target_interactions, truth_interactions, fold, setting):
#     """
#     For User User collaborative filtering.
#     Calculates the top k most similar users to another user based on user shelves
    
#     To aviod having to calculate the n x n cross pruduct similarity, we instead block by clustering books 
#     based on content (measured through labeled shelves) and then block users by the cluster that represents 
#     the majority on their shelf
        
#     :param interactions_df:
#     :param partition:
#     :return:
#     """
#     k = 10

#     def write(s, p):
#         s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/item_collab_filter_cosine_' + p + '.txt',
#                  sep='\t', header=False, index=True)

#     # observed predicates
#     partition = 'obs'
#     obs_interactions = train_interactions_df.index
#     target_interactions = train_interactions_df.index
#     truth_interactions = test_interactions_df.index

#     # observed predicates
#     partition = 'obs'
#     observed_interactions_df = interactions_df.loc[obs_interactions, :]
#     observed_books_df = books_df.loc[obs_interactions.get_level_values(1).unique()]

#     # similarity based on shelves (<- chosen since it the most dense value)
#     # TODO: incorporate sparser and less noising signals from user

#     # index: user_ids columns: book_ids values: 1 if shelved by user 0 o.w.
#     user_book_shelves = pd.Series(data=1, index=obs_interactions).unstack().fillna(0)
#     # index: book_ids columns: book_ids
#     item_item_sim = pd.DataFrame(data=0, index=user_book_shelves.columns, columns=user_book_shelves.columns)

#     # block similarity calculations by language and by genre heuristics
#     # use unique values of language codes to block similarity calculations: generalize by adding 
#     #             support for clusters of languages, i.e. eng and can-en can be grouped together
#     # use top-1 informative shelf name: generalize by using top-n informative shelf names

#     # create genre to book data frame, genre is from shelf names by users
#     # index: book_id, columns: genre
#     genre_series = pd.Series(index=observed_books_df.index)
#     for book_id, book in observed_books_df.iterrows():
#         for i in range(len(book.popular_shelves)):
#             # note that popular_shelves is sorted
#             if book.popular_shelves[i]['name'] != 'to-read':
#                 genre_series.loc[book.name] = book.popular_shelves[i]['name']
#                 break

#     # add genre series as column of observed_books_df
#     observed_books_df.loc[:, 'genre'] = genre_series

#     # block by both 'genre' and 'language_code' and calculate similarity for books within block
#     for _, book_group in observed_books_df.groupby(['genre', 'language_code']):
#         similarity_matrix = cosine_similarity(user_book_shelves.loc[:, book_group.index].transpose())
#         item_item_sim.loc[book_group.index, book_group.index] = similarity_matrix
        
#     # Very sparse, thus very small cosine sim values
#     # approach: for each book use k most similar books to it
#     book_top_k_sim = item_item_sim.apply(pd.Series.nlargest, n=5).stack()
#     item_item_series = pd.Series(data=1, index=book_top_k_sim.index)
#     write(item_item_series, partition)


# def user_collab_filter_jaccard_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting, n_neighbors=25):
#     """

#     :param interactions_df:
#     :param partition:
#     :return:
#     """

#     def write(s, p):
#         s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/user_collab_filter_jaccard_' + p + '.txt',
#                  sep='\t', header=False, index=True)

#     # observed predicates
#     partition = 'obs'
#     observed_interactions_df = interactions_df.loc[obs_interactions, :]

#     # build user book ratings matrix
#     # Note that this could perhaps be improved by incorporating user means as threshold
#     user_book_ratings = observed_interactions_df['rating'].unstack().fillna(0)
#     user_book_ratings[user_book_ratings < 2.5] = 0
#     user_book_ratings[user_book_ratings > 2.5] = 1

#     # find top n_neighbors most similar users based on this similarity metric
#     NN = NearestNeighbors(n_neighbors=n_neighbors, metric="jaccard", algorithm='ball_tree')
#     NN.fit(user_book_ratings)
#     NearestNeighbors_df = pd.DataFrame(NN.kneighbors(return_distance=False), index=user_book_ratings.index)

#     # dereference neighbors, stack, set index, and write
#     NearestNeighbors_df = NearestNeighbors_df.apply(lambda x: NearestNeighbors_df.index[x], axis=0)
#     NearestNeighbors_df = NearestNeighbors_df.stack().reset_index().set_index(['user_id', 0])
#     NearestNeighbors_df.level_1 = 1

#     write(NearestNeighbors_df, partition)


def item_collab_filter_jaccard_predicate(interactions_df, obs_interactions, target_interactions, truth_interactions, fold, setting, n_neighbors=25):
    """

    :param interactions_df:
    :param partition:
    :return:
    """

    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/item_collab_filter_jaccard_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    observed_interactions_df = interactions_df.loc[obs_interactions, :]

    # build user book ratings matrix
    # Note that this could perhaps be improved by incorporating user means as threshold
    user_book_ratings = observed_interactions_df['rating'].unstack().fillna(0)
    user_book_ratings[user_book_ratings < 2.5] = 0
    user_book_ratings[user_book_ratings > 2.5] = 1

    # find top n_neighbors most similar users based on this similarity metric
    NN = NearestNeighbors(n_neighbors=n_neighbors, metric="jaccard", algorithm='ball_tree')
    NN.fit(user_book_ratings.transpose())
    NearestNeighbors_df = pd.DataFrame(NN.kneighbors(return_distance=False), index=user_book_ratings.columns)

    # dereference neighbors, stack, set index, and write
    NearestNeighbors_df = NearestNeighbors_df.apply(lambda x: NearestNeighbors_df.index[x], axis=0)
    NearestNeighbors_df = NearestNeighbors_df.stack().reset_index().set_index(['book_id', 0])
    NearestNeighbors_df.level_1 = 1

    write(NearestNeighbors_df, partition)


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
            if n_genres >= max_genres:
                break
            else:
                if genre['name'] != 'to-read':
                    genre_df.loc[genre['name'], book.name] = 1
                    n_genres = n_genres + 1

    genre_series = genre_df.stack().swaplevel()
    write(genre_series, partition)


def book_average_rating_predicate(books_df, fold, setting):
    """

    :param books_df:
    :param fold:
    :param setting:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/book_average_rating_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    # create book to average_rating_frame
    average_rating_frame = books_df.average_rating
    average_rating_frame = average_rating_frame - 1
    average_rating_frame = average_rating_frame / 5

    write(average_rating_frame, partition)


def book_publisher_predicate(books_df, fold, setting):
    """

    :param books_df:
    :param fold:
    :param setting:
    :return:
    """
    def write(s, p):
        s.to_csv('./goodreads/' + str(fold) + '/' + setting + '/publisher_' + p + '.txt',
                 sep='\t', header=False, index=True)

    # observed predicates
    partition = 'obs'
    # create book to publisher
    book_publisher = books_df.loc[:, ["publisher"]]
    book_publisher.loc[:, 'pred'] = 1

    write(book_publisher, partition)
