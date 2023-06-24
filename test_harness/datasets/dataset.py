# Import libraries
import pandas as pd
from typing import Dict

# Define class
class Dataset:
    def __init__(self, name, full_df: pd.DataFrame, column_mapping: Dict, window_size: int):
        
        self.name = name
        self.full_df = full_df
        self.column_mapping = column_mapping
        self.window_size = window_size
    
    def get_data_by_idx(self, start_idx, end_idx, split_labels=True):
        """
        Given an index into the full_df, return all records up to that observation.

        Args:
            start_idx (int) - index corresponding to the row in full_df
            end_idx (int) - index corresponding to the row in full_df
            split_labels (bool) - return features and labels separately vs. as one dataframe

        Returns:
            features (pd.DataFrame)
            labels (pd.Series)
        """

        window_data = self.full_df[start_idx:end_idx]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    @staticmethod
    def split_df(df, label_col):
        """Splits the features from labels in a dataframe, returns both."""
        return df.drop(label_col, axis=1), df[label_col]
