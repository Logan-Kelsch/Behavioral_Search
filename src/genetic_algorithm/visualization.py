import pydot
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

def visualize_tree(root, vizout, run_dir:str=None):
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
    
    if(vizout):
        graph.write_png(str(run_dir / 'best_tree.png'))
        del graph
        return Image(str(run_dir / 'best_tree.png'))
    else:
        graph.write_png('best_tree.png')
        del graph
        return Image('best_tree.png')



def visualize_all_distributions(x, show:bool=False):
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

    if(show):
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    return fig, axes

from sklearn.metrics import r2_score, confusion_matrix

def visualize_regression_eval(
    y_test,
    y_pred,
    plot_lims   :   float   =   0.015,
    title       :   str     =   'Scatter with Quadrant Counts',
    show        :   bool    =   False,
    run_dir     :   str     =   '.',
    vizout      :   bool    =   False
):
    y_test_dir = y_test >= 0
    y_pred_dir = y_pred >= 0
    cm = confusion_matrix(y_test_dir, y_pred_dir)
    counts = cm
    if(np.isnan(y_test).any() or np.isnan(y_pred).any()):
        print('nan found in y_test or y_pred, NN loss set to maximum')
        return 0, 0
    r2 = r2_score(y_test, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    fig, ax = plt.subplots(figsize=(9,6))

    # Shaded quadrants via pcolormesh
    x_edges = [-plot_lims, 0, plot_lims]
    y_edges = [-plot_lims, 0, plot_lims]
    mesh = ax.pcolormesh(x_edges, y_edges, counts, cmap='Greens', alpha=0.4, shading='auto')

    # Overlay scatter of individual points
    ax.scatter(y_pred, y_test, alpha=0.75, color='black', s=5)

    # Annotate each quadrant with its count
    positions = [(-plot_lims/2, -plot_lims/2), (plot_lims/2, -plot_lims/2), (-plot_lims/2, plot_lims/2), (plot_lims/2, plot_lims/2)]
    for (ypos, xpos), count in zip(positions, counts.flatten()):
        ax.text(ypos, xpos, f"{int(count)}", ha='center', va='center', fontsize=16, color='black',
        bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2')
                )

    # Draw quadrant divider lines at zero
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = x_vals  # Since y = x
    plt.plot(x_vals, y_vals, '-', color='black', label='y = x', linewidth=0.5)
    # Set axes limits and labels
    ax.set_xlim(-plot_lims, plot_lims)
    ax.set_ylim(-plot_lims, plot_lims)
    ax.set_ylabel('Actual Value')
    ax.set_xlabel('Predicted Value')
    ax.set_title(f'{title} (R2: {r2:.4}, ACC: {accuracy:.2%})')

    # Add colorbar for quadrant shading
    cbar = fig.colorbar(mesh, ax=ax, extend='neither')
    cbar.set_label('Count per Quadrant')

    if(show):
        plt.tight_layout()
        plt.show()

    if(vizout):
        fig.savefig(str(run_dir / f'{title}.png'))

    plt.close()
    
    del fig, cbar, ax

    return r2, accuracy

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize_opt_path(path, pscores, title=None, dirpath=None):
    """
    Plots the unit square and the path taken by the optimizer.
    - path: list of (x, y) tuples or Nx2 array-like of positions
    - pscores: list/array of scores (0 best → 1 worst), same length as path
    """

    maroon_cmap = LinearSegmentedColormap.from_list(
        'maroon', ["#0099FF", "#87878782", "#DB0000"]
    )
    fig, ax = plt.subplots()
    # Draw square boundary
    square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    ax.plot(square[:,0], square[:,1], linewidth=1, color='black')
    
    # Convert inputs to arrays
    positions = np.array(path)
    scores = np.array(pscores)
    
    # Faint gray line connecting the steps
    ax.plot(positions[:,0], positions[:,1], color='gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Color-coded scatter
    scatter = ax.scatter(positions[:,0], positions[:,1], c=scores, cmap=maroon_cmap, zorder=2)
    plt.colorbar(scatter, ax=ax, label='Score')
    
    # Styling
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title)
    
    plt.savefig(str(dirpath / f'{title}.png'))

    plt.close()
    del ax, fig

