import pydot
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

import pydot
from IPython.display import Image

def visualize_tree(root, filename='tree.png'):
    """
    Visualize a binary tree with children `._x` and `._alpha` using pydot.
    Non-leaf nodes are deduplicated (cached), while each leaf instance
    is rendered uniquely even if repr(node) is identical.
    """
    graph = pydot.Dot('Tree', graph_type='digraph', rankdir='TB')
    seen = {}  # cache for non-leaf nodes
    leaf_counter = {'count': 0}  # mutable counter for unique leaf IDs

    def recurse(node):
        if node is None:
            return None

        # Determine children
        children = [(attr, getattr(node, attr, None))
                    for attr in ('_x', '_alpha')
                    if getattr(node, attr, None) is not None]

        # Leaf node: always create a fresh graph node
        if not children:
            leaf_id = leaf_counter['count']
            leaf_counter['count'] += 1
            gname = f"leaf_{leaf_id}"
            label = repr(node)
            graph.add_node(pydot.Node(gname, label=label, shape='none'))
            return gname

        # Non-leaf: cache by object id
        nid = id(node)
        if nid in seen:
            return seen[nid]

        # Create non-leaf node with its _type
        label = f"T: {getattr(node, '_type', '')}"
        gname = f"node_{nid}"
        graph.add_node(pydot.Node(gname, label=label, shape='none'))
        seen[nid] = gname

        # Recurse into children
        for attr, child in children:
            cname = recurse(child)
            graph.add_edge(pydot.Edge(gname, cname, label=attr))

        return gname

    recurse(root)
    graph.write_png(filename)
    return Image(filename)



def visualize_all_distributions(x):
    # grid layout: adjust ncols as needed
    n_features = x.shape[1]
    ncols = int(np.ceil(np.sqrt(n_features / 1.77) * 1.77))
    nrows = int(np.ceil(np.sqrt(n_features / 1.77) ))

    # 16×9 overall figure
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(16, 9),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    for idx, col in enumerate(x.columns):
        ax = axes[idx]
        ax.hist(x[col].dropna(), bins=30, zorder=1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            (ylim[0] + ylim[1]) / 2,
            str(idx),
            ha='center',
            va='center',
            fontsize=18,        # slightly smaller
            color='black',      # now black and visible
            alpha=0.8,          # slight transparency
            zorder=2
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0)

    # turn off any extra axes
    for ax in axes[n_features:]:
        ax.axis('off')

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()