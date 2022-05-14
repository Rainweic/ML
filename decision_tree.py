import numpy as np
from ensemble.base.splittler import BestSplitter
from ensemble.base.tree_builder import BestFirstBuilder


class HeteroDecisionTree:

    def __init__(self, 
                 label,
                 n_bins,
                 max_depth,
                 max_leaf_nodes,
                 min_samples_leaf,
                 min_samples_split,
                 max_features
                ):
        self.label = label
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, train_df, val_df=None):

        train_data, train_label = self.process_data(train_df, self.label)
        val_data, val_label = self.process_data(val_df, self.label)

        self.tree = self.train(train_data, train_label, val_data, val_label)

        summary = {"summaries": self.tree.cal_metric()}

        return summary

    def train(self, train_data, train_label, test_data, test_label):

        self.splitter = BestSplitter(train_data, train_label, n_bins=self.n_bins)

        builder = BestFirstBuilder(
            self.n_bins,
            self.splitter,
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf
        )

        tree = builder.build(
            train_data,
            train_label,
            test_data,
            test_label
        )

        return tree

    def process_data(self, dataset, label):
        """
        预处理数据 对传入的数据进行划分为data与label
        """
        # 训练时可能没有val数据 则直接返回None
        if dataset is None:
            return None, None, None

        data_df = dataset.df

        label = data_df[label].to_numpy()
        # 【二分类】将数据中的y统一成0与1 不需要good value
        min_value = np.min(label)
        label = np.where(label == min_value, 0, 1)

        data = data_df.to_numpy()

        return data, label
