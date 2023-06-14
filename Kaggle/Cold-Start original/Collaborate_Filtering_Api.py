# -*- Coding UTF-8 -*-
# @Time: 2023/1/25 11:54
# @Author: Yiyang Bian
# @File: Collaborate_Filtering_Api.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt


class Collaborate_Filtering:
    def __init__(self, rate, datasets, models, num_Top_model):
        self.rate = rate
        self.dataset = datasets
        self.model = models
        self.numTopmodel = num_Top_model

    def Matrix_Generation(self, file, index, columns):
        matrix = pd.DataFrame(index=index, columns=columns)
        for row in file.itertuples():
            matrix[row[2]][row[1]] = row[3]
        matrix = matrix.fillna(0)
        return matrix

    def predict(self, ratings, similarity):
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - np.array(mean_user_rating)[:, np.newaxis])
        pred = np.array(mean_user_rating)[:, np.newaxis] + np.dot(similarity, ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
        return pred

    def rmse(self, prediction, ground_truth):
        prediction = prediction.to_numpy()[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    def find_sim_index(self, index, meta_dataset_similarity):
        row1 = meta_dataset_similarity.loc[index]
        row1_max_index = row1[row1 == row1.max()].index[0]
        return row1_max_index

    def datasets_based_recommend(self, ratings, ratings_train, ratings_test):
        datasets = ratings.dataset_id.unique()
        models = ratings.model_id.unique()

        data_model_train_matrix = self.Matrix_Generation(ratings_train, datasets, models)
        data_model_test_matrix = self.Matrix_Generation(ratings_test, datasets, models)

        dataset_similarity = cosine_similarity(data_model_train_matrix)
        model_similarity = cosine_similarity(data_model_train_matrix.T)
        dataset_similarity_matrix = pd.DataFrame(dataset_similarity, index=datasets, columns=datasets)

        model_prediction = self.predict(data_model_train_matrix, dataset_similarity_matrix)
        model_prediction = pd.DataFrame(model_prediction, index=datasets, columns=models).sort_index()

        meta_datasets = pd.read_csv("datasets_v.csv", low_memory=False)
        meta_datasets = meta_datasets.loc[:, ('c1', 'c2', 'c3', 'c4', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8')]
        meta_dataset_similarity = cosine_similarity(meta_datasets.values.tolist())
        meta_dataset_similarity = pd.DataFrame(meta_dataset_similarity)

        i, j = 0, 0
        while i < len(datasets):
            while j < len(datasets):
                meta_dataset_similarity[i][j] = None
                i += 1
                j += 1
        for i in range(len(datasets)):
            if model_prediction.loc[i].isnull().any():
                max_index = self.find_sim_index(i, meta_dataset_similarity)
                model_prediction.loc[i] = model_prediction.loc[max_index]

        data_model_test_matrix = data_model_test_matrix[(data_model_test_matrix.T != 0).any()]
        index_list = data_model_test_matrix.index.to_list()
        model_ranking_groundtruth = pd.DataFrame(index=range(len(datasets)), columns=range(70))

        for i in index_list:
            for j in range(70):
                # index = index_list[i]
                row = data_model_test_matrix.loc[i]
                row = row.sort_values(ascending=False)
                index_row = row.index
                index_row = pd.DataFrame(index_row).T
                model_ranking_groundtruth.loc[i][j] = index_row.loc[0][j]

        model_ranking_groundtruth = model_ranking_groundtruth[~model_ranking_groundtruth.isna().any(axis=1)]

        model_ranking_prediction = pd.DataFrame(index=range(len(datasets)), columns=range(self.numTopmodel))
        for i in range(len(datasets)):
            for j in range(self.numTopmodel):
                row = model_prediction.loc[i]
                row = row.sort_values(ascending=False)
                row = row.index
                row = pd.DataFrame(row).T
                model_ranking_prediction.loc[i][j] = row.loc[0][j]
        model_ranking_prediction = model_ranking_prediction.loc[index_list]
        return model_ranking_prediction


def main():
    CF = Collaborate_Filtering("rate.csv", "datasets_v.csv", "models.csv", 20)

    ratings = pd.read_csv("rate.csv", low_memory=False)
    ratings_train = pd.read_csv("rate_train.csv", low_memory=False)
    ratings_test = pd.read_csv("rate_test.csv", low_memory=False)
    Top_20_models = CF.datasets_based_recommend(ratings, ratings_train, ratings_test)
    print(Top_20_models)


if __name__ == "__main__":
    main()
