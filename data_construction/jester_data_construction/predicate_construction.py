"""
This script constructs the necessary data files needed for psl-ltr-recommender
"""
import pandas as pd

import sys
sys.path.append('../')
from predicate_construction_helpers import query_relevance_cosine_similarity
from predicate_construction_helpers import query_item_preferences
from predicate_construction_helpers import hac_canopy_from_distance
from predicate_construction_helpers import target_preferences
from predicate_construction_helpers import filter_and_write_targets
"""
Jester Configs
"""
data_path = './jester'
dataset_directory_nums = ['0', '1', '2', '3', '4', '5', '6', '7']
dataset_types = ['eval', 'learn']

"""
iterate over all dataset directories and types
"""

for data_dir_num in dataset_directory_nums:
    for data_type in dataset_types:
        """
        Import raw data
        """
        ratings_obs_df = pd.read_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rating_obs.txt',
                                     sep='\t', header=None)
        ratings_obs_df.columns = ['userId', 'jokeId', 'rating']
        ratings_obs_df = ratings_obs_df.astype({'userId': str, 'jokeId': str, 'rating': float})

        ratings_targets_df = pd.read_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rating_targets.txt',
                                         sep='\t', header=None)
        ratings_targets_df.columns = ['userId', 'jokeId']
        ratings_targets_df = ratings_targets_df.astype({'userId': str, 'jokeId': float})

        ratings_truth_df = pd.read_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rating_truth.txt',
                                       sep='\t', header=None)
        ratings_truth_df.columns = ['userId', 'jokeId', 'rating']
        ratings_truth_df = ratings_truth_df.astype({'userId': str, 'jokeId': str, 'rating': float})

        joke_similarity_frame = pd.read_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'simJokeText_obs.txt',
            sep='\t', header=None)
        joke_similarity_frame.columns = ['jokeId_1', 'jokeId_2', 'similarity']
        joke_similarity_frame = joke_similarity_frame.astype({'jokeId_1': str, 'jokeId_2': str, 'similarity': float})
        joke_similarity_series = joke_similarity_frame.set_index(['jokeId_1', 'jokeId_2']).loc[:, 'similarity']

        users = set(ratings_obs_df.userId.unique()).union(set(ratings_truth_df.userId.unique()).union(
            set(ratings_targets_df.userId.unique())))
        jokes = set(ratings_obs_df.jokeId.unique()).union(set(ratings_truth_df.jokeId.unique()).union(
            set(ratings_targets_df.jokeId.unique())))

        """
        User Similarity Predicate: built only from observed ratings
        """
        user_cosine_similarity_series = query_relevance_cosine_similarity(ratings_obs_df, 'userId', 'jokeId', 'rating')
        user_cosine_similarity_series.to_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'sim_users_obs.txt',
            sep='\t', header=False, index=True
        )

        """
        User User Canopy
        """
        user_cosine_distance_series = 1 - user_cosine_similarity_series
        user_user_canopy_series = hac_canopy_from_distance(user_cosine_distance_series,
                                                           user_cosine_distance_series.mean())
        user_user_canopy_series.to_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'user_user_canopy_obs.txt',
            sep='\t', header=False, index=True)

        """
        Joke Joke Canopy
        """

        joke_distance_series = 1 - joke_similarity_series
        joke_joke_canopy_series = hac_canopy_from_distance(joke_distance_series, joke_distance_series.mean())
        joke_joke_canopy_series.to_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'joke_joke_canopy_obs.txt',
            sep='\t', header=False, index=True)

        """
        Relative Rank
        """
        # observed relative ranks
        observed_user_joke_preferences = list(map(
            query_item_preferences(ratings_obs_df, 'userId', 'jokeId', 'rating'), ratings_obs_df.userId.unique()
        ))
        observed_relative_rank_df = pd.concat(observed_user_joke_preferences, keys=[df.name for df in
                                                                                    observed_user_joke_preferences])

        observed_relative_rank_df.to_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_obs.txt',
                                         sep='\t', header=False, index=True)

        # truth relative ranks
        truth_user_joke_preferences = list(map(
            query_item_preferences(ratings_truth_df, 'userId', 'jokeId', 'rating'), ratings_truth_df.userId.unique()
        ))
        truth_relative_rank_df = pd.concat(truth_user_joke_preferences, keys=[df.name for df in
                                                                              truth_user_joke_preferences])

        truth_relative_rank_df.to_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_truth.txt',
                                      sep='\t', header=False, index=True)

        # target relative rank
        target_relative_rank_series = target_preferences(joke_joke_canopy_series, users)

        write_path = data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_targets.txt'
        filter_and_write_targets(target_relative_rank_series, observed_relative_rank_df, write_path)

