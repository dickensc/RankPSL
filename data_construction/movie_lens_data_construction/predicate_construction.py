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
from predicate_construction_helpers import target_preferences
from predicate_construction_helpers import filter_and_write_targets
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels

n_folds = 5

"""
Import raw data
"""
movies_df = pd.read_csv('./ml-100k/u.item', sep='|', header=None, encoding="ISO-8859-1")
movies_df.columns = ["movieId", "movie title", "release date", "video release date", "IMDb URL ", "unknown", "Action",
                     "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                     "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movies_df = movies_df.set_index('movieId')

ratings_df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None)
ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_df = ratings_df.astype({'userId': str, 'movieId': str})
ratings_df.rating = ratings_df.rating / ratings_df.rating.max()

users = ratings_df.userId.unique()
movies = ratings_df.movieId.unique()

print("Num Users: {}".format(users.shape[0]))
print("Num Movies: {}".format(movies.shape[0]))

"""
Create data directory to write output to
"""
if not os.path.exists('./movie_lens'):
    os.makedirs('./movie_lens')

"""
Dev. subset to only 100 users and 200 movies
"""
n_users = 100
n_movies = 200
np.random.seed(0)

users = ratings_df.userId.unique()[:n_users]
ratings_df = ratings_df[ratings_df.userId.isin(users)]

movies = ratings_df.movieId.unique()
movies = np.random.choice(movies, n_movies, replace=False)
movies_df = movies_df[movies_df.index.isin(movies)]
ratings_df = ratings_df[ratings_df.movieId.isin(movies)]

for fold in range(n_folds):
    """
    Create data directory to write output to
    """
    if not os.path.exists('./movie_lens/' + str(fold)):
        os.makedirs('./movie_lens/' + str(fold))

    """
    Partition into target and observed movie ratings
    """
    ratings_permutation = np.random.permutation(ratings_df.index)

    observed_ratings = ratings_permutation[: int(1 * len(ratings_permutation) / 2)]
    truth_ratings = ratings_permutation[int(1 * len(ratings_permutation) / 2):]

    observed_ratings_frame = ratings_df.loc[observed_ratings]
    truth_ratings_frame = ratings_df.loc[truth_ratings]

    """
    User scoping predicate
    """
    user_scope_series = pd.Series(data=1, index=users)
    user_scope_series.to_csv('./movie_lens/' + str(fold) + '/user_obs.txt',
                             sep='\t', header=False, index=True)

    """
    Item scoping predicate
    """
    item_scope_series = pd.Series(data=1, index=movies)
    item_scope_series.to_csv('./movie_lens/' + str(fold) + '/item_obs.txt',
                             sep='\t', header=False, index=True)

    """
    Ratings Predicates
    """
    # obs
    observed_ratings_series = observed_ratings_frame.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
    observed_ratings_series.to_csv('./movie_lens/' + str(fold) + '/rating_obs.txt',
                                   sep='\t', header=False, index=True)

    # truth
    truth_ratings_series = truth_ratings_frame.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
    truth_ratings_series.to_csv('./movie_lens/' + str(fold) + '/rating_truth.txt',
                                sep='\t', header=False, index=True)

    # target
    truth_ratings_series.to_csv('./movie_lens/' + str(fold) + '/rating_targets.txt',
                                sep='\t', header=False, index=True)

    """
    Rated Blocking Predicate
    """
    rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
    rated_series.to_csv('./movie_lens/' + str(fold) + '/rated_obs.txt',
                        sep='\t', header=False, index=True)


    """
    Average user rating predicates
    """
    avg_user_rating_series = observed_ratings_frame.loc[:, ['userId', 'rating']].groupby('userId').mean()
    avg_user_rating_series.to_csv('./movie_lens/' + str(fold) + '/avg_user_rating_obs.txt',
                                  sep='\t', header=False, index=True)

    """
    Average user rating predicates
    """
    avg_item_rating_series = observed_ratings_frame.loc[:, ['movieId', 'rating']].groupby('movieId').mean()
    avg_item_rating_series.to_csv('./movie_lens/' + str(fold) + '/avg_item_rating_obs.txt',
                                  sep='\t', header=False, index=True)

    """
    User Similarity Predicate: sim_cosine_users, built only from observed ratings
    """
    user_cosine_similarity_series = query_relevance_cosine_similarity(
        observed_ratings_frame.loc[:, ['userId', 'movieId', 'rating']],
        'userId', 'movieId')
    # take top 50 for each user to define pairwise blocks
    user_cosine_similarity_block_frame = pd.DataFrame(index=users, columns=range(50))
    for u in observed_ratings_frame.userId.unique():
        user_cosine_similarity_block_frame.loc[u, :] = user_cosine_similarity_series.loc[u].nlargest(50).index

    flattened_frame = user_cosine_similarity_block_frame.values.flatten()
    user_index = np.array([[i] * 50 for i in user_cosine_similarity_block_frame.index]).flatten()
    user_cosine_similarity_block_index = pd.MultiIndex.from_arrays([user_index, flattened_frame])
    user_cosine_similarity_block_series = pd.Series(data=1, index=user_cosine_similarity_block_index)

    user_cosine_similarity_block_series.to_csv('./movie_lens/' + str(fold) + '/sim_cosine_users_obs.txt',
                                               sep='\t', header=False, index=True)

    """
    Item Similarity Predicate: sim_cosine_items, built only from observed ratings
    """
    item_cosine_similarity_series = query_relevance_cosine_similarity(
        observed_ratings_frame.loc[:, ['userId', 'movieId', 'rating']],
        'movieId', 'userId')

    # take top 50 for each movie to define pairwise blocks
    item_cosine_similarity_block_frame = pd.DataFrame(index=movies, columns=range(50))
    for m in observed_ratings_frame.movieId.unique():
        item_cosine_similarity_block_frame.loc[m, :] = item_cosine_similarity_series.loc[m].nlargest(50).index

    flattened_frame = item_cosine_similarity_block_frame.values.flatten()
    item_index = np.array([[i] * 50 for i in item_cosine_similarity_block_frame.index]).flatten()
    item_cosine_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
    item_cosine_similarity_block_series = pd.Series(data=1, index=item_cosine_similarity_block_index)

    item_cosine_similarity_block_series.to_csv('./movie_lens/' + str(fold) + '/sim_cosine_items_obs.txt',
                                               sep='\t', header=False, index=True)

    """
    Item Content Similarity Predicate: sim_content_items_jaccard_obs:  built from genres
    """
    movie_genres_df = movies_df.loc[:, ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                                        "Romance", "Sci-Fi", "Thriller", "War", "Western"]]

    movie_jaccard_similarity_series = pd.DataFrame(
        data=pairwise_kernels(movie_genres_df, metric=jaccard_score, n_jobs=4),
        index=movie_genres_df.index,
        columns=movie_genres_df.index).stack()

    # take top 50 for each movie to define pairwise blocks
    movie_content_similarity_block_frame = pd.DataFrame(index=movies_df.index, columns=range(50))
    for m in movies_df.index:
        movie_content_similarity_block_frame.loc[m, :] = movie_jaccard_similarity_series.loc[m].nlargest(50).index

    flattened_frame = movie_content_similarity_block_frame.values.flatten()
    item_index = np.array([[i] * 50 for i in movie_content_similarity_block_frame.index]).flatten()
    item_content_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
    item_content_similarity_block_series = pd.Series(data=1, index=item_content_similarity_block_index)

    item_content_similarity_block_series.to_csv('./movie_lens/' + str(fold) + '/sim_content_items_jaccard.txt',
                                               sep='\t', header=False, index=True)

    """
    Item cluster predicate: item_cluster: based on only content
    """
    movie_jaccard_distance_series = (1 - movie_jaccard_similarity_series)
    movie_cluster_assignments = hac_cluster_from_distance(movie_jaccard_distance_series,
                                                          distance_threshold=0.5)

    item_cluster_predicates_index = pd.MultiIndex.from_arrays([movie_cluster_assignments.index, movie_cluster_assignments.values])
    item_cluster_predicates = pd.Series(data=1, index=item_cluster_predicates_index)
    item_cluster_predicates.to_csv('./movie_lens/' + str(fold) + '/item_cluster_obs.txt',
                                   sep='\t', header=False, index=True)

    """
    Item cluster similarity: sim_cosine_item_clusters: based on user ratings
    """
    cluster_groups = item_cluster_predicates.groupby(level=1)

    movie_user_ratings_frame = observed_ratings_frame[['movieId', 'userId', 'rating']].set_index(['movieId', 'userId']).rating.unstack()

    clustered_ratings = pd.DataFrame(data=0, index=np.unique(movie_cluster_assignments.values), columns=users)

    # Find average movie rating for each user over the cluster
    for cluster in cluster_groups.groups:
        cluster_item_ids = [str(x) for x in cluster_groups.groups[cluster].get_level_values(0).values]
        clustered_ratings.loc[cluster, movie_user_ratings_frame.columns] = \
            movie_user_ratings_frame.reindex(cluster_item_ids, copy=False).mean(skipna=True)

    filled_clustered_ratings = clustered_ratings.fillna(0)

    # use the users average rating to find the cosine similarity of the clusters
    item_cluster_cosine_similarity_series = pd.DataFrame(
        data=cosine_similarity(filled_clustered_ratings),
        index=filled_clustered_ratings.index,
        columns=filled_clustered_ratings.index).stack()

    # take top 5 for each movie to define pairwise blocks
    item_cluster_cosine_similarity_block_frame = pd.DataFrame(index=filled_clustered_ratings.index, columns=range(5))
    for cluster in cluster_groups.groups:
        item_cluster_cosine_similarity_block_frame.loc[cluster, :] = item_cluster_cosine_similarity_series.loc[cluster].nlargest(5).index

    flattened_frame = item_cluster_cosine_similarity_block_frame.values.flatten()
    item_cluster_index = np.array([[i] * 5 for i in item_cluster_cosine_similarity_block_frame.index]).flatten()
    item_cluster_cosine_similarity_block_index = pd.MultiIndex.from_arrays([item_cluster_index, flattened_frame])
    item_cluster_cosine_similarity_block_series = pd.Series(data=1, index=item_cluster_cosine_similarity_block_index)

    item_cluster_cosine_similarity_block_series.to_csv('./movie_lens/' + str(fold) + '/sim_cosine_item_clusters.txt',
                                                       sep='\t', header=False, index=True)

    """
    Item cluster rank: item_cluster_rank: latent variable only need targets
    """
    user_item_cluster_index = pd.MultiIndex.from_product([users, list(cluster_groups.groups.keys())])
    pd.DataFrame(index=user_item_cluster_index).to_csv('./movie_lens/' + str(fold) + '/item_cluster_rank_targets.txt',
                                                       sep='\t', header=False, index=True)

    """
    Cluster Preference Predicate: item_cluster_preference: only observed those clusters with a rating
    """
    # observed clustered ratings
    observed_cluster_ratings = clustered_ratings.stack()
    observed_cluster_ratings.index = observed_cluster_ratings.index.set_names(['clusterId', 'userId'])
    observed_cluster_ratings.name = 'rating'
    observed_cluster_ratings_frame = observed_cluster_ratings.reset_index()

    observed_cluster_preferences = list(
        map(query_item_preferences(observed_cluster_ratings_frame, 'userId', 'clusterId', 'rating'),
            observed_cluster_ratings_frame.userId.unique()
            )
    )

    observed_cluster_preferences_df = pd.concat(observed_cluster_preferences, keys=[df.name for df in
                                                                                    observed_cluster_preferences])

    observed_cluster_preferences_df.to_csv('./movie_lens/' + str(fold) + '/item_cluster_preference_obs.txt',
                                           sep='\t', header=False, index=True)

    """
    Preference Predicate: 
    """
    # # observed relative ranks
    # observed_user_movie_preferences = list(
    #     map(query_item_preferences(observed_ratings_frame, 'userId', 'movieId', 'rating'),
    #         observed_ratings_frame.userId.unique()
    #         )
    # )
    # observed_relative_rank_df = pd.concat(observed_user_movie_preferences, keys=[df.name for df in
    #                                                                              observed_user_movie_preferences])
    #
    #
    # observed_relative_rank_df.to_csv('./movie_lens/rel_rank_obs.txt',
    #                                  sep='\t', header=False, index=True,
    #                                  chunksize=100000)
    #
    # # truth relative ranks
    # truth_user_movie_preferences = list(
    #     map(query_item_preferences(truth_ratings_frame, 'userId', 'movieId', 'rating'),
    #         truth_ratings_frame.userId.unique()
    #         )
    # )
    # truth_relative_rank_df = pd.concat(truth_user_movie_preferences, keys=[df.name for df in truth_user_movie_preferences])
    #
    # truth_relative_rank_df.to_csv('./movie_lens/rel_rank_truth.txt',
    #                               sep='\t', header=False, index=True,
    #                               chunksize=100000)
    #
    # # target relative rank
    # target_relative_rank_series = target_preferences(movie_movie_canopy_series, users)
    #
    # write_path = './movie_lens/rel_rank_targets.txt'
    # filter_and_write_targets(target_relative_rank_series, observed_relative_rank_df, write_path)
