
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

_TREE_UNDEFINED = -1


@dataclass
class Node:
    """树结点"""

    node_id: int = _TREE_UNDEFINED
    left_child: int = _TREE_UNDEFINED
    feature: int = _TREE_UNDEFINED
    threshold: int = _TREE_UNDEFINED
    impurity: float = _TREE_UNDEFINED
    n_node_samples: int = 0
    n_samples_val: int = 0
    value: list = field(default_factory=list)
    value_val: list = field(default_factory=list)
    class_type: int = _TREE_UNDEFINED

    def __post_init__(self):
        self.value = [_TREE_UNDEFINED, _TREE_UNDEFINED]
        self.value_val = [_TREE_UNDEFINED, _TREE_UNDEFINED]


class Tree:
    """决策树"""

    def __init__(self):
        self.node_count = 0
        self.n_leaf_node = 0
        self.nodes = []
        self.leaf_nodes_id = []
        self.summary = {}
    
    def add_node(
        self,
        parent_id,
        is_left,
        feature,
        threshold,
        impurity,
        n_node_samples,
        n_samples_val
    ):

        node_id = self.node_count

        node = Node(
            node_id=node_id,
            impurity=impurity,
            n_node_samples=n_node_samples,
            n_samples_val=n_samples_val
        )

        if parent_id != _TREE_UNDEFINED:
            if is_left:
                # print("[INFO] 设置父节点的左孩子为", node_id)
                self.nodes[parent_id].left_child = node_id
            else:
                # print("[INFO] 设置父节点的右孩子为", node_id)
                self.nodes[parent_id].right_child = node_id

        node.feature = feature
        node.threshold = threshold

        self.nodes.append(node)
        self.node_count += 1

        return node_id

    def _create_graph(self, G, node, pos={}, x=0, y=0, layer=1):

        G.add_node(node.node_id)
        pos[node.node_id] = (x, y)

        if node.left_child != _TREE_UNDEFINED:
            G.add_edge(node.node_id, self.nodes[node.left_child].node_id)
            l_x, l_y = x - 1 / 2**layer, y - 1
            l_layer = layer + 1
            self._create_graph(G, self.nodes[node.left_child], pos, l_x, l_y, l_layer)
        if node.right_child != _TREE_UNDEFINED:
            G.add_edge(node.node_id, self.nodes[node.right_child].node_id)
            r_x, r_y = x + 1 / 2**layer, y - 1
            r_layer = layer + 1
            self._create_graph(G, self.nodes[node.right_child], pos, r_x, r_y, r_layer)

        return G, pos

    def _create_label(self):
        node_labels = {}
        for node_id in range(self.node_count):
            node = self.nodes[node_id]
            node_labels[node_id] = f"""
                "ID {node.node_id}
                "Owner {node.owner}
                "Feature {node.feature}
                "Threshold {node.threshold}
                "Gini {node.impurity}
                "Value {node.value}
                "Class {node.class_type}
                "N_samples {node.n_node_samples}
                "N_samples_val {node.n_samples_val}
            """
        return node_labels

    def show(self, save_name="test.jpg"):
        """
        图形显示tree
        """
        graph = nx.DiGraph()
        graph, pos = self._create_graph(graph, self.nodes[0])
        fig, ax = plt.subplots(figsize=(8, 10))
        node_labels = self._create_label()
        nx.draw_networkx(graph, pos, ax=ax, node_size=300)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
        if len(save_name) != 0:
            plt.savefig(save_name)
        plt.show()