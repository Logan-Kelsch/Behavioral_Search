import pydot
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

def visualize_tree(root, filename='tree.png'):
    """
    Recursively visualize a binary tree with children `._x` and `._alpha`
    using pydot/Graphviz. Non-leaf nodes are labeled only by their `_type` attribute,
    while leaf nodes use repr(node).
    """
    graph = pydot.Dot('Tree', graph_type='digraph', rankdir='TB')
    seen = {}  # map python id(node) -> graph node name

    def recurse(node):
        if node is None:
            return None

        nid = id(node)
        if nid in seen:
            return seen[nid]

        # Determine children
        children = [(attr, getattr(node, attr, None)) 
                    for attr in ('_x', '_alpha') 
                    if getattr(node, attr, None) is not None]

        # Create label: non-leaf shows only _type, leaf shows repr(node)
        if children:
            label = f"T: {getattr(node, '_type', '')}"
        else:
            label = repr(node)

        gname = f"node_{nid}"
        graph_node = pydot.Node(gname, label=label, shape='none')
        graph.add_node(graph_node)
        seen[nid] = gname

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