import collections

import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreparation:

    def __init__(self):
        self.df = pd.DataFrame
        self.label_encoder = LabelEncoder()
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def add_missing_data_10_percent(self, df):
        """
        random NaN values (10%) are inserted to the data set
        :return:
        """

        replaced = collections.defaultdict(set)
        ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1] - 1)]
        random.shuffle(ix)
        to_replace = int(round(.1 * len(ix)))
        for row, col in ix:
            if len(replaced[row]) < df.shape[1] - 1:
                df.iloc[row, col] = np.nan
                to_replace -= 1
                replaced[row].add(col)
                if to_replace == 0:
                    break

    def fill_na(self):
        """
        fills NaN values.
        Categorical columns were filled with the most common value,
        while numerical columns were filed with the mean of the column.
        :return:
        """

        for col in self.df:
            if self.df[col].dtype == np.object:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(self.df[col].mean(), inplace=True)

    def discretization(self):
        """
         sklearn models can not handle categorical data, label-encoding (similar to dummy data) was applied.
         in addition, we used discretization for continuous-numerical data.
        :return:
        """

        for col in self.df:
            if self.df[col].dtype != np.object:
                self.df[col] = pd.cut(x=self.df[col], bins=3, include_lowest=True)
            self.df[col] = self.label_encoder.fit_transform(self.df[col])

    def partition_data_sets(self):

        self.y = self.df['class']
        self.x = self.df.drop('class', axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.33,
                                                                                random_state=42)
