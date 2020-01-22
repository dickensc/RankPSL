"""
This script constructs the necessary data files needed for psl-ltr-recommender
"""
import pandas as pd

import sys
sys.path.append('../')
from predicate_construction_helpers import query_item_preferences
from predicate_construction_helpers import target_preferences
from predicate_construction_helpers import filter_and_write_targets
from predicate_construction_helpers import hac_canopy_from_distance

"""
Yelp Configs
"""
data_path = './yelp'
dataset_directory_nums = ['0', '1', '2', '3', '4']
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
        ratings_obs_df.columns = ['userId', 'itemId', 'rating']
        ratings_obs_df = ratings_obs_df.astype({'userId': str, 'itemId': str, 'rating': float})

        ratings_targets_df = pd.read_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rating_targets.txt',
                                         sep='\t', header=None)
        ratings_targets_df.columns = ['userId', 'itemId']
        ratings_targets_df = ratings_targets_df.astype({'userId': str, 'itemId': float})

        ratings_truth_df = pd.read_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rating_truth.txt',
                                       sep='\t', header=None)
        ratings_truth_df.columns = ['userId', 'itemId', 'rating']
        ratings_truth_df = ratings_truth_df.astype({'userId': str, 'itemId': str, 'rating': float})

        item_similarity_frame = pd.read_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'sim_cosine_items_obs.txt',
            sep='\t', header=None)
        item_similarity_frame.columns = ['itemId_1', 'itemId_2', 'similarity']
        item_similarity_frame = item_similarity_frame.astype({'itemId_1': str, 'itemId_2': str, 'similarity': float})
        item_similarity_series = item_similarity_frame.set_index(['itemId_1', 'itemId_2']).loc[:, 'similarity']

        user_similarity_frame = pd.read_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'sim_cosine_items_obs.txt',
            sep='\t', header=None)
        user_similarity_frame.columns = ['userId_1', 'userId_2', 'similarity']
        user_similarity_frame = user_similarity_frame.astype({'userId_1': str, 'userId_2': str, 'similarity': float})
        user_similarity_series = user_similarity_frame.set_index(['userId_1', 'userId_2']).loc[:, 'similarity']

        users = set(ratings_obs_df.userId.unique()).union(set(ratings_truth_df.userId.unique()).union(
            set(ratings_targets_df.userId.unique())))

        """
        User User Canopy
        """
        user_cosine_distance_series = 1 - user_similarity_series
        user_user_canopy_series = hac_canopy_from_distance(user_cosine_distance_series,
                                                           user_cosine_distance_series.mean())
        user_user_canopy_series.to_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'user_user_canopy_obs.txt',
            sep='\t', header=False, index=True)

        """
        Item Item Canopy
        """
        item_distance_series = 1 - item_similarity_series
        item_item_canopy_series = hac_canopy_from_distance(item_distance_series, item_distance_series.mean())
        item_item_canopy_series.to_csv(
            data_path + '/' + data_dir_num + '/' + data_type + '/' + 'item_item_canopy_obs.txt',
            sep='\t', header=False, index=True)

        """
        Relative Rank
        """
        # observed relative ranks
        observed_user_joke_preferences = list(map(
            query_item_preferences(ratings_obs_df, 'userId', 'itemId', 'rating'), ratings_obs_df.userId.unique()
        ))
        observed_relative_rank_df = pd.concat(observed_user_joke_preferences, keys=[df.name for df in
                                                                                    observed_user_joke_preferences])
        observed_relative_rank_df.to_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_obs.txt',
                                         sep='\t', header=False, index=True)

        # truth relative ranks
        truth_user_joke_preferences = list(map(
            query_item_preferences(ratings_truth_df, 'userId', 'itemId', 'rating'), ratings_truth_df.userId.unique()
        ))
        truth_relative_rank_df = pd.concat(truth_user_joke_preferences, keys=[df.name for df in
                                                                              truth_user_joke_preferences])
        truth_relative_rank_df.to_csv(data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_truth.txt',
                                      sep='\t', header=False, index=True)

        # target relative rank
        # target relative rank
        target_relative_rank_series = target_preferences(item_item_canopy_series, users)
        write_path = data_path + '/' + data_dir_num + '/' + data_type + '/' + 'rel_rank_targets.txt'
        filter_and_write_targets(target_relative_rank_series, observed_relative_rank_df, write_path)
