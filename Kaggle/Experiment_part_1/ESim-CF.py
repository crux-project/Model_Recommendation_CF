# -*- Coding UTF-8 -*-
# @Time: 6/6/2023 10:36 AM
# @Author: Yiyang Bian
# @File: ESim-CF.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class UserSimilarity:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def CosineSimilarity(self):
        cosine_sim = cosine_similarity(self.df.values)
        return pd.DataFrame(cosine_sim, index=self.df.index, columns=self.df.index)

    def RandomWalk(self):
        def create_bipartite_adjacency_matrix(rating_matrix):
            n_users, n_items = rating_matrix.shape
            adjacency_matrix = np.zeros((n_users + n_items, n_users + n_items))
            adjacency_matrix[:n_users, n_users:] = rating_matrix
            adjacency_matrix[n_users:, :n_users] = rating_matrix.T
            return adjacency_matrix

        def propagation_matrix_withWalkLength(adjacency_matrix, max_walk_length):
            adjacency_matrix = np.nan_to_num(adjacency_matrix)
            propagation_matrix = np.eye(adjacency_matrix.shape[0])
            sum_matrix = np.eye(adjacency_matrix.shape[0])

            for _ in range(max_walk_length):
                propagation_matrix = propagation_matrix @ adjacency_matrix
                sum_matrix += propagation_matrix

            return sum_matrix

        def normalize_kernel(kernel_matrix):
            # 计算矩阵的最小值和最大值
            min_val = np.min(kernel_matrix)
            max_val = np.max(kernel_matrix)

            # 防止除数为零的情况
            if max_val == min_val:
                return np.zeros_like(kernel_matrix)

            # 将矩阵的值缩放到0和1之间
            normalized_kernel_matrix = (kernel_matrix - min_val) / (max_val - min_val)

            return normalized_kernel_matrix

        # 计算二分图邻接矩阵
        bipartite_adjacency_matrix = create_bipartite_adjacency_matrix(self.df)

        propagation_maxLength = propagation_matrix_withWalkLength(bipartite_adjacency_matrix, 1)
        # 提取用户-商品传播矩阵和商品-用户传播矩阵
        n_users = self.df.shape[0]
        user_item_propagation = propagation_maxLength[:n_users, n_users:]
        item_user_propagation = propagation_maxLength[n_users:, :n_users]
        # 计算 Random Walk Kernel
        random_walk_kernel = np.dot(user_item_propagation, item_user_propagation)

        normalized_kernel = normalize_kernel(random_walk_kernel)
        return pd.DataFrame(normalized_kernel, index=self.df.index, columns=self.df.index)

    def DeepWalk(self, walk_length, num_walks, embed_size):
        def create_graph_from_df(df):
            G = nx.Graph()

            for user in df.index:
                for item in df.columns:
                    rating = df.loc[user, item]
                    if not np.isnan(rating):
                        G.add_edge(user, item, weight=rating)

            return G

        def deepwalk(G, walk_length, num_walks, embed_size=32):
            walks = []
            for node in G.nodes():
                if G.degree(node) == 0:
                    continue
                for _ in range(num_walks):
                    walk = [node]
                    while len(walk) < walk_length:
                        cur = walk[-1]
                        cur_nbrs = list(G.neighbors(cur))
                        walk.append(np.random.choice(cur_nbrs))
                    walks.append([str(node) for node in walk])

            model = Word2Vec(walks, vector_size=embed_size, window=5, min_count=0, sg=1, workers=4)
            return model

        def get_similarity_matrix(model, user_nodes):
            embeddings = np.array([model.wv.get_vector(str(user)) for user in user_nodes])
            similarity_matrix = cosine_similarity(embeddings)

            return pd.DataFrame(similarity_matrix, index=user_nodes, columns=user_nodes)

        # 创建图
        G = create_graph_from_df(self.df)

        # 执行DeepWalk算法
        model = deepwalk(G)

        # 获取用户相似性矩阵
        user_nodes = self.df.index.tolist()
        similarity_matrix = get_similarity_matrix(model, user_nodes)
        return pd.DataFrame(similarity_matrix, index=self.df.index, columns=self.df.index)
