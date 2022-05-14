import numpy as np
from collections import namedtuple
from tree import Tree
from tree import _TREE_UNDEFINED

FrontierRecord = namedtuple(
    "FrontierRecord", [
        "node_id", "X", "y", "gini", "bid", "fid", "depth", "X_val", "y_val", "n_samples",
        "n_samples_val"
    ],
    defaults=[None, None, 0, 0]
)


class BestFirstBuilder():

    def __init__(
        self,
        n_bins,
        splitter,
        gini_threshold=1e-5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=_TREE_UNDEFINED,
        max_leaf_nodes=_TREE_UNDEFINED
    ):
        super().__init__()
        self.n_bins = n_bins
        self.splitter = splitter
        self.gini_threshold = gini_threshold
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth if max_depth != _TREE_UNDEFINED else float('inf')
        self.max_leaf_nodes = max_leaf_nodes

    def _add_to_frontier(self, rec, frontier):
        frontier.append(rec)
        return frontier

    def build(self, X, y=None, X_val=None, y_val=None):
        """
        先添加根节点到队列 循环队列直到队列为空
        队列中弹出第一个节点进行分裂
        """
        tree = Tree()
        # 对数据进行分箱
        splitter = self.splitter
        X = splitter.data_2_bins(X)
        # 建立数组
        frontier = []
        # 添加根节点
        frontier = self._add_node_and_set_record(None, None, True, X, y, X_val, y_val, splitter, tree, frontier)
        # 层层建树
        while len(frontier) > 0:

            record = frontier.pop(0)
            node = tree.nodes[record.node_id]
            # 设置当前节点中label为0、1的数量
            tree.nodes[record.node_id].value = [record.n_samples - record.y.sum(), record.y.sum()]
            if y_val is not None:
                tree.nodes[record.node_id].value_val = [
                    record.n_samples_val - record.y_val.sum(),
                    record.y_val.sum()
                ]

            # 对当前数据进行切分为X1 X2
            # 大于或等于阈值的为X1 小于则为X2
            X1_mask, X2_mask, X1_val_mask, X2_val_mask = splitter.split_2_X1_X2(record)

            # 划分训练集
            if record.X is not None:
                X1 = record.X[X1_mask]
                X2 = record.X[X2_mask]
            else:
                X1, X2 = None, None

            if record.y is not None:
                y1 = record.y[X1_mask]
                y2 = record.y[X2_mask]
            else:
                y1, y2 = None, None

            # 划分验证集
            if record.X_val is not None:
                X1_val = record.X_val[X1_val_mask]
                X2_val = record.X_val[X2_val_mask]
            else:
                X1_val, X2_val = None, None

            if record.y_val is not None:
                y1_val = record.y_val[X1_val_mask]
                y2_val = record.y_val[X2_val_mask]
            else:
                y1_val, y2_val = None, None

            """判断当前record是否为叶子节点"""
            is_leaf = bool(
                # 当前节点中样本数量少于min_samples_leaf则停止分裂
                (record.n_samples <= self.min_samples_leaf) or
                # gini小于阈值则为叶子节点
                (record.gini <= self.gini_threshold) or
                # 划分后其中一份数据量小于min_samples_split则停止分裂
                (X1_mask.sum() <= self.min_samples_leaf or X2_mask.sum() <= self.min_samples_leaf) or
                # 当前节点深度达到最大深度则为叶子节点
                (record.depth == self.max_depth - 1)
            )

            # 若为叶子结点
            if is_leaf:
                tree.nodes[record.node_id].left_child = _TREE_UNDEFINED
                tree.nodes[record.node_id].right_child = _TREE_UNDEFINED
                tree.nodes[record.node_id].feature = _TREE_UNDEFINED
                tree.nodes[record.node_id].threshold = _TREE_UNDEFINED
                tree.nodes[record.node_id].n_node_samples = record.n_samples
                tree.nodes[record.node_id].n_samples_val = record.n_samples_val

                # 设置叶子节点类别
                class_type = np.argmax(tree.nodes[record.node_id].value)
                tree.nodes[record.node_id].class_type = class_type

                tree.n_leaf_node += 1
                tree.leaf_nodes_id.append(record.node_id)

            else:
                # ----------------------------- 对X1设置为左结点 -----------------------------
                frontier = self._add_node_and_set_record(record, node, True, X1, y1, X1_val, y1_val, splitter, tree, frontier, f"{key}l")
                # ----------------------------- 对X2设置为右结点 -----------------------------
                frontier = self._add_node_and_set_record(record, node, False, X2, y2, X2_val, y2_val, splitter, tree, frontier, f"{key}r")

        return tree

    def _add_node_and_set_record(self, record, node, is_left, X, y, X_val, y_val, splitter, tree, frontier):
        """
        1. 往树上添加节点
        2. 将当前数据信息记录至record 并将record放入队列
        """
        gini_X, bid_X, fid_X = splitter.get_optimal_gini_bid_fid(X, y)

        X_n_samples = X.shape[0] if X is not None else y.shape[0]
        X_n_samples_val = X_val.shape[0] if X_val is not None else 0

        # 给树添加节点
        threshold = splitter.get_threshold(fid_X, bid_X)
        node_id = tree.add_node(
            parent_id=tree.nodes.index(node) if node else _TREE_UNDEFINED,
            is_left=is_left,
            feature=fid_X,
            threshold=threshold,
            impurity=gini_X,
            n_node_samples=X_n_samples,
            n_samples_val=X_n_samples_val
        )

        depth = record.depth + 1 if record else 0
        frontierRecord = FrontierRecord(
            node_id, X, y, gini_X, bid_X, fid_X, depth, X_val, y_val, X_n_samples, X_n_samples_val
        )
        frontier = self._add_to_frontier(frontierRecord, frontier)

        return frontier
