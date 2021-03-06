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
item_ratings_df = pd.read_csv('./learning-from-sets-2019/item_ratings.csv', sep=',')
# scale to between 0 and 1 ratings
item_ratings_df.rating = item_ratings_df.rating / item_ratings_df.rating.max()

set_ratings_df = pd.read_csv('./learning-from-sets-2019/set_ratings.csv', sep=',')
# scale to between 0 and 1 ratings
set_ratings_df.rating = set_ratings_df.rating / set_ratings_df.rating.max()

movies_df = pd.read_csv('./ml-25m/movies.csv', sep=',')
movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies()], axis=1)
movies_df = movies_df.set_index('movieId')
movies_df = movies_df.drop('genres', axis=1)

users = item_ratings_df.userId.unique()
movies = item_ratings_df.movieId.unique()

# subset movies df to at least only those movies for which we have ratings for
movies_df = movies_df.loc[movies_df.index.isin(movies)]

print("Num Users: {}".format(users.shape[0]))
print("Num Movies: {}".format(movies.shape[0]))

"""
Create data directory to write output to
"""
if not os.path.exists('./movielens_set'):
    os.makedirs('./movielens_set')

"""
Dev. subset to only 100 users and 200 movies
"""
n_users = 100
n_movies = 200
np.random.seed(0)

# subset by users
users = item_ratings_df.userId.unique()[:n_users]
item_ratings_df = item_ratings_df[item_ratings_df.userId.isin(users)]
set_ratings_df = set_ratings_df[set_ratings_df.userId.isin(users)]

# subset by movies
movies = item_ratings_df.movieId.unique()
movies = np.random.choice(movies, n_movies, replace=False)
movies_df = movies_df[movies_df.index.isin(movies)]
item_ratings_df = item_ratings_df[item_ratings_df.movieId.isin(movies)]
set_ratings_df = set_ratings_df[set_ratings_df.movieId_1.isin(movies) | set_ratings_df.movieId_2.isin(movies) |
                                set_ratings_df.movieId_3.isin(movies) | set_ratings_df.movieId_4.isin(movies) |
                                set_ratings_df.movieId_5.isin(movies)]

for fold in range(n_folds):
    """
    Create data directory to write output to
    """
    if not os.path.exists('./movielens_set/' + str(fold)):
        os.makedirs('./movielens_set/' + str(fold))

    if not os.path.exists('./movielens_set/' + str(fold) + '/learn/'):
        os.makedirs('./movielens_set/' + str(fold) + '/learn/')

    if not os.path.exists('./movielens_set/' + str(fold) + '/eval/'):
        os.makedirs('./movielens_set/' + str(fold) + '/eval/')

    """
    Partition into learn and eval sets
    """
    item_ratings_permutation = np.random.permutation(item_ratings_df.index)
    learn_item_ratings = item_ratings_permutation[: int(1 * len(item_ratings_permutation) / 2)]
    eval_item_ratings = item_ratings_permutation[int(1 * len(item_ratings_permutation) / 2):]

    item_ratings_frames = {'learn': item_ratings_df.loc[learn_item_ratings], 'eval': item_ratings_df.loc[eval_item_ratings]}

    # Sample set preferences for learn and eval
    set_ratings_permutation = np.random.permutation(set_ratings_df.index)
    learn_set_ratings = set_ratings_permutation[: int(1 * len(set_ratings_permutation) / 2)]
    eval_set_ratings = set_ratings_permutation[int(1 * len(set_ratings_permutation) / 2):]

    set_ratings_frames = {'learn': set_ratings_df.loc[learn_set_ratings], 'eval': set_ratings_df.loc[eval_set_ratings]}

    # Filter so only set ratings by users with item ratings are in frames to set_rating users is a subset of item_rating users
    set_ratings_frames['learn'] = set_ratings_frames['learn'][
        set_ratings_frames['learn'].userId.isin(item_ratings_frames['learn'].userId.unique())
    ]

    set_ratings_frames['eval'] = set_ratings_frames['eval'][
        set_ratings_frames['eval'].userId.isin(item_ratings_frames['eval'].userId.unique())
    ]

    for setting in ['learn', 'eval']:
        """
        Partition into target and observed movie item and set ratings
        """
        item_ratings_permutation = np.random.permutation(item_ratings_frames[setting].index)
        set_ratings_permutation = np.random.permutation(set_ratings_frames[setting].index)

        observed_item_ratings = item_ratings_permutation[: int(1 * len(item_ratings_permutation) / 2)]
        observed_set_ratings = set_ratings_permutation[: int(1 * len(set_ratings_permutation) / 2)]
        truth_item_ratings = item_ratings_permutation[int(1 * len(item_ratings_permutation) / 2):]
        truth_set_ratings = set_ratings_permutation[int(1 * len(set_ratings_permutation) / 2):]

        observed_item_ratings_frame = item_ratings_frames[setting].loc[observed_item_ratings]
        observed_set_ratings_frame = set_ratings_frames[setting].loc[observed_set_ratings]
        truth_item_ratings_frame = item_ratings_frames[setting].loc[truth_item_ratings]
        truth_set_ratings_frame = set_ratings_frames[setting].loc[truth_set_ratings]

        """
        User scoping predicate
        """
        user_scope_series = pd.Series(data=1, index=item_ratings_frames[setting].userId.unique())
        user_scope_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/user_obs.txt',
                                 sep='\t', header=False, index=True)

        """
        Item scoping predicate
        """
        item_scope_series = pd.Series(data=1, index=item_ratings_frames[setting].movieId.unique())
        item_scope_series.to_csv('./movielens_set/' + str(fold) + '/' + setting +'/item_obs.txt',
                                 sep='\t', header=False, index=True)

        """
        Ratings Predicates
        """
        # obs
        observed_ratings_series = observed_item_ratings_frame.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
        observed_ratings_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/rating_obs.txt',
                                       sep='\t', header=False, index=True)

        # truth
        truth_ratings_series = truth_item_ratings_frame.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
        truth_ratings_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/rating_truth.txt',
                                    sep='\t', header=False, index=True)

        # target
        truth_ratings_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/rating_targets.txt',
                                    sep='\t', header=False, index=True)

        """
        Rated Blocking Predicate
        """
        rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
        rated_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/rated_obs.txt',
                            sep='\t', header=False, index=True)


        """
        Average user rating predicates
        """
        avg_user_rating_series = observed_item_ratings_frame.loc[:, ['userId', 'rating']].groupby('userId').mean()
        avg_user_rating_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/avg_user_rating_obs.txt',
                                      sep='\t', header=False, index=True)

        """
        Average user rating predicates
        """
        avg_item_rating_series = observed_item_ratings_frame.loc[:, ['movieId', 'rating']].groupby('movieId').mean()
        avg_item_rating_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/avg_item_rating_obs.txt',
                                      sep='\t', header=False, index=True)

        """
        User Similarity Predicate: sim_cosine_users, built only from observed ratings
        """
        user_cosine_similarity_series = query_relevance_cosine_similarity(
            observed_item_ratings_frame.loc[:, ['userId', 'movieId', 'rating']],
            'userId', 'movieId')
        # take top 50 for each user to define pairwise blocks
        user_cosine_similarity_block_frame = pd.DataFrame(index=users, columns=range(25))
        for u in observed_item_ratings_frame.userId.unique():
            user_cosine_similarity_block_frame.loc[u, :] = user_cosine_similarity_series.loc[u].nlargest(25).index

        # some users may not have rated any movie in common with another user
        user_cosine_similarity_block_frame = user_cosine_similarity_block_frame.dropna(axis=0)

        flattened_frame = user_cosine_similarity_block_frame.values.flatten()
        user_index = np.array([[i] * 25 for i in user_cosine_similarity_block_frame.index]).flatten()
        user_cosine_similarity_block_index = pd.MultiIndex.from_arrays([user_index, flattened_frame])
        user_cosine_similarity_block_series = pd.Series(data=1, index=user_cosine_similarity_block_index)

        user_cosine_similarity_block_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/sim_cosine_users_obs.txt',
                                                   sep='\t', header=False, index=True)

        """
        Item Similarity Predicate: sim_cosine_items, built only from observed ratings
        """
        item_cosine_similarity_series = query_relevance_cosine_similarity(
            observed_item_ratings_frame.loc[:, ['userId', 'movieId', 'rating']],
            'movieId', 'userId')

        # take top 25 for each movie to define pairwise blocks
        item_cosine_similarity_block_frame = pd.DataFrame(index=movies, columns=range(25))
        for m in observed_item_ratings_frame.movieId.unique():
            item_cosine_similarity_block_frame.loc[m, :] = item_cosine_similarity_series.loc[m].nlargest(25).index

        # some movies may not have been rated by any user
        item_cosine_similarity_block_frame = item_cosine_similarity_block_frame.dropna(axis=0)
        flattened_frame = item_cosine_similarity_block_frame.values.flatten()
        item_index = np.array([[i] * 25 for i in item_cosine_similarity_block_frame.index]).flatten()
        item_cosine_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
        item_cosine_similarity_block_series = pd.Series(data=1, index=item_cosine_similarity_block_index)

        item_cosine_similarity_block_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/sim_cosine_items_obs.txt',
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

        # take top 25 for each movie to define pairwise blocks
        movie_content_similarity_block_frame = pd.DataFrame(index=movies_df.index, columns=range(25))
        for m in movies_df.index:
            movie_content_similarity_block_frame.loc[m, :] = movie_jaccard_similarity_series.loc[m].nlargest(25).index

        flattened_frame = movie_content_similarity_block_frame.values.flatten()
        item_index = np.array([[i] * 25 for i in movie_content_similarity_block_frame.index]).flatten()
        item_content_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
        item_content_similarity_block_series = pd.Series(data=1, index=item_content_similarity_block_index)

        item_content_similarity_block_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/sim_content_items_jaccard_obs.txt',
                                                   sep='\t', header=False, index=True)

        """
        Item cluster predicate: item_cluster: based on only content
        """
        movie_jaccard_distance_series = (1 - movie_jaccard_similarity_series)
        movie_cluster_assignments = hac_cluster_from_distance(movie_jaccard_distance_series,
                                                              distance_threshold=0.5)

        item_cluster_predicates_index = pd.MultiIndex.from_arrays([movie_cluster_assignments.index, movie_cluster_assignments.values])
        item_cluster_predicates = pd.Series(data=1, index=item_cluster_predicates_index)
        item_cluster_predicates.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_cluster_obs.txt',
                                       sep='\t', header=False, index=True)

        """
        Item cluster similarity: sim_cosine_item_clusters: based on user ratings
        """
        cluster_groups = item_cluster_predicates.groupby(level=1)

        observed_movie_user_ratings_frame = observed_item_ratings_frame[['movieId', 'userId', 'rating']].set_index(['movieId', 'userId']).rating.unstack()
        truth_movie_user_ratings_frame = item_ratings_frames[setting][['movieId', 'userId', 'rating']].set_index(['movieId', 'userId']).rating.unstack()

        observed_clustered_ratings = pd.DataFrame(data=0, index=np.unique(movie_cluster_assignments.values), columns=users)
        truth_clustered_ratings = pd.DataFrame(data=0, index=np.unique(movie_cluster_assignments.values), columns=users)

        # Find average movie rating for each user over the cluster
        for cluster in cluster_groups.groups:
            cluster_item_ids = [str(x) for x in cluster_groups.groups[cluster].get_level_values(0).values]
            observed_clustered_ratings.loc[cluster, observed_movie_user_ratings_frame.columns] = \
                observed_movie_user_ratings_frame.reindex(cluster_item_ids, copy=False).mean(skipna=True)
            truth_clustered_ratings.loc[cluster, truth_movie_user_ratings_frame.columns] = \
                truth_movie_user_ratings_frame.reindex(cluster_item_ids, copy=False).mean(skipna=True)

        filled_clustered_ratings = observed_clustered_ratings.fillna(0)

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

        item_cluster_cosine_similarity_block_series.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/sim_cosine_item_clusters_obs.txt',
                                                           sep='\t', header=False, index=True)

        """
        Item cluster rank: item_cluster_rank: latent variable only need targets
        """
        user_item_cluster_index = pd.MultiIndex.from_product([users, list(cluster_groups.groups.keys())])
        pd.DataFrame(index=user_item_cluster_index).to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_cluster_rank_targets.txt',
                                                           sep='\t', header=False, index=True)

        """
        Cluster Preference Predicate: item_cluster_preference: only observed those clusters with a rating
        """
        # observed clustered ratings
        observed_clustered_ratings_series = observed_clustered_ratings.stack()
        observed_clustered_ratings_series.index = observed_clustered_ratings_series.index.set_names(['clusterId', 'userId'])
        observed_clustered_ratings_series.name = 'rating'
        observed_cluster_ratings_frame = observed_clustered_ratings_series.reset_index()

        observed_cluster_preferences = list(
            map(query_item_preferences(observed_cluster_ratings_frame, 'userId', 'clusterId', 'rating'),
                observed_cluster_ratings_frame.userId.unique()
                )
        )

        observed_cluster_preferences_df = pd.concat(observed_cluster_preferences, keys=[df.name for df in
                                                                                        observed_cluster_preferences])

        observed_cluster_preferences_df.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_cluster_preference_obs.txt',
                                               sep='\t', header=False, index=True)

        # truth clustered ratings
        truth_clustered_ratings_series = truth_clustered_ratings.stack()
        # filter already observed cluster ratings
        truth_clustered_ratings_series = truth_clustered_ratings_series[~truth_clustered_ratings_series.index.isin(observed_clustered_ratings_series.index)]
        truth_clustered_ratings_series.index = truth_clustered_ratings_series.index.set_names(['clusterId', 'userId'])
        truth_clustered_ratings_series.name = 'rating'
        truth_clustered_ratings_frame = truth_clustered_ratings_series.reset_index()

        truth_cluster_preferences = list(
            map(query_item_preferences(truth_clustered_ratings_frame, 'userId', 'clusterId', 'rating'),
                truth_clustered_ratings_frame.userId.unique()
                )
        )

        truth_cluster_preferences_df = pd.concat(truth_cluster_preferences, keys=[df.name for df in
                                                                                  truth_cluster_preferences])

        truth_cluster_preferences_df.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_cluster_preference_truth.txt',
                                               sep='\t', header=False, index=True)

        # target clustered ratings
        all_preferences_index = pd.MultiIndex.from_product([users, list(cluster_groups.groups.keys()),
                                                            list(cluster_groups.groups.keys())])
        target_preferences_index = all_preferences_index[~(all_preferences_index.isin(observed_cluster_preferences_df.index))]
        target_cluster_preferences_df = pd.DataFrame(index=target_preferences_index)
        target_cluster_preferences_df.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_cluster_preference_targets.txt',
                                             sep='\t', header=False, index=True)

        """
        Preference Predicate: item_preference: blocked by rated()
        """
        # observed relative ranks
        observed_user_movie_preferences = list(
            map(query_item_preferences(observed_item_ratings_frame, 'userId', 'movieId', 'rating'),
                observed_item_ratings_frame.userId.unique()
                )
        )
        observed_relative_rank_df = pd.concat(observed_user_movie_preferences, keys=[df.name for df in
                                                                                     observed_user_movie_preferences])

        observed_relative_rank_df.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_preference_obs.txt',
                                         sep='\t', header=False, index=True, chunksize=100000)

        # target relative ranks
        target_user_movie_preferences = list(
            map(query_item_preferences(item_ratings_frames[setting], 'userId', 'movieId', 'rating'),
                item_ratings_frames[setting].userId.unique()
                )
        )
        target_relative_rank_df = pd.concat(target_user_movie_preferences,
                                            keys=[df.name for df in target_user_movie_preferences])

        target_relative_rank_df = target_relative_rank_df[~target_relative_rank_df.index.isin(observed_relative_rank_df.index)]

        target_relative_rank_df.to_csv('./movielens_set/' + str(fold) + '/' + setting + '/item_preference_targets.txt',
                                       sep='\t', header=False, index=True, chunksize=100000)

        # # target relative rank
        # target_relative_rank_series = target_preferences(movie_movie_canopy_series, users)
        #
        # write_path = './movielens/rel_rank_targets.txt'
        # filter_and_write_targets(target_relative_rank_series, observed_relative_rank_df, write_path)
