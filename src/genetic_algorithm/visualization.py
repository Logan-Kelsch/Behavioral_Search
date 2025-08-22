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
import pathlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation



def animate_opt_path_bary(path_bary, pscores, dir_path='/mnt/data', title='tetra_rot.gif', frames=36, interval=100):
    """
    Creates and saves a rotating GIF of the optimizer path inside a regular tetrahedron,
    with axis panes removed, uniform rotation about the centroid, colorbar, and vertex labels.
    Labels: M, E, R, C for vertices 0-3 respectively.
    """
    # Regular tetrahedron vertices & edges
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ])
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    labels = ['M', 'E', 'R', 'C']
    
    # Convert barycentric to Cartesian
    path_cart = np.dot(path_bary, verts)
    
    # Custom colormap and normalization
    stops = [
        (0.0, "blue"),
        (0.5, "blue"),
        (0.9, "green"),
        (0.95, "gray"),
        (1.0, "red"),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom", stops)
    norm = Normalize(vmin=0, vmax=1)
    
    # Setup figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    # Remove axis panes and ticks
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # Draw tetrahedron edges
    for i, j in edges:
        xs, ys, zs = verts[[i, j]].T
        ax.plot(xs, ys, zs, color='black', linewidth=1)
    
    # Label vertices
    for label, (x, y, z) in zip(labels, verts):
        ax.text(x, y, z, label, fontsize=12, fontweight='bold', zorder=3)
    
    # Plot path line and scatter
    ax.plot(path_cart[:,0], path_cart[:,1], path_cart[:,2],
            color='gray', alpha=0.2, linewidth=1)
    sc = ax.scatter(path_cart[:,0], path_cart[:,1], path_cart[:,2],
                    c=pscores, cmap=custom_cmap, norm=norm, s=15)
    cb = fig.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Score')
    
    # Equal axes
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)

    # Animation function: rotate around centroid
    def update(frame):
        azim = 360 * frame / frames
        ax.view_init(elev=30, azim=azim)
        return ax,
    
    anim = FuncAnimation(fig, update, frames=frames, interval=interval)
    save_path = pathlib.Path(dir_path) / title
    anim.save(str(save_path), writer='pillow', fps=1000/interval)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import pathlib

def visualize_opt_path_3d(path, pscores, title=None, dirpath=None,
                          frames=90, interval=200):
    """
    Plots the 3D path taken by the optimizer in a box that fits the
    data domain, then animates a full 360° rotation and saves it as a GIF.

    Before plotting, applies np.exp to all values in the second dimension (y),
    and auto‐fits all three axes to the data extents (with a small margin).

    Parameters
    ----------
    path : sequence of length‑3 iterables
        List or array-like of (x, y, z) positions.
    pscores : sequence of floats
        Scores in [0, 1], one per position.
    title : str, optional
        Base name for the output file (no extension).
    dirpath : pathlib.Path or str, optional
        Directory in which to save the GIF. Defaults to cwd.
    frames : int, optional
        Number of frames in the rotation (default 360).
    interval : int, optional
        Delay between frames in milliseconds (default 50).
    """
    # Prepare output path
    dirpath = pathlib.Path(dirpath) if dirpath is not None else pathlib.Path.cwd()
    dirpath.mkdir(parents=True, exist_ok=True)
    fname = (title or "opt_path_3d") + ".gif"
    out_path = dirpath / fname

    # Colormap
    maroon_cmap = LinearSegmentedColormap.from_list(
        "maroon", ["#0099FF", "#87878782", "#DB0000"]
    )

    # Convert inputs and apply exp to the 2nd dimension (y)
    positions = np.asarray(path, dtype=float).copy()
    positions[:, 1] = np.exp(positions[:, 1])
    scores = np.asarray(pscores, dtype=float)

    # Compute axis limits with 5% margin
    def limits(vals):
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin
        m = 0.05 * span if span > 0 else 0.05
        return vmin - m, vmax + m

    x_min, x_max = limits(positions[:, 0])
    y_min, y_max = limits(positions[:, 1])
    z_min, z_max = limits(positions[:, 2])

    # Set up 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw dynamic rectangular prism edges
    # Bottom face (z = z_min)
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            [z_min]*5, color="black", lw=1)
    # Top face (z = z_max)
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            [z_max]*5, color="black", lw=1)
    # Vertical edges
    for xi, yi in [(x_min, y_min), (x_min, y_max),
                   (x_max, y_min), (x_max, y_max)]:
        ax.plot([xi, xi], [yi, yi], [z_min, z_max],
                color="black", lw=1)

    # Connect the path
    ax.plot3D(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="gray", alpha=0.1, linewidth=1, zorder=1
    )

    # Color‑coded scatter
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=scores, cmap=maroon_cmap, s=12, zorder=2
    )
    fig.colorbar(scatter, ax=ax, label="Score")

    # Apply the computed limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("ATR Coef")
    ax.set_ylabel("Exit Ratio")
    ax.set_zlabel("Loss")

    if title:
        ax.set_title(title)

    # Animation: rotate around vertical axis
    def update(frame):
        az = 360 * frame / frames
        ax.view_init(elev=30, azim=az)
        return scatter,

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    anim.save(str(out_path), writer="pillow", fps=1000/interval)
    plt.close(fig)


def visualize_cumulative_first_feature_pl(best_pls, time_index=None, figsize=(10, 4), title=None):
    """
    Plot the cumulative profit/loss series over time for the first feature's best mask.

    Parameters
    ----------
    best_pls : list of np.ndarray
        The list returned by `best_feature_pl`, where each entry is the
        masked-PL array for one feature.
    time_index : array-like, optional
        Sequence of time points (e.g., dates or integer indices). If None,
        uses range(len(best_pls[0])).
    figsize : tuple, optional
        Figure size passed to matplotlib.
    title : str, optional
        If provided, used as the plot title. Otherwise defaults to
        "Cumulative PL Over Time - Feature 1".

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects for further customization.
    """
    # Extract the first feature’s masked PL series
    pl_series = best_pls[0]
    
    # Replace NaNs with zero so they don't break the cumulative sum
    pl_clean = np.where(np.isnan(pl_series), 0.0, pl_series)
    
    # Compute cumulative PL
    cum_pl = np.cumsum(pl_clean)
    
    # Build x-axis
    if time_index is None:
        x = np.arange(len(cum_pl))
    else:
        x = np.asarray(time_index)
        if x.shape[0] != cum_pl.shape[0]:
            raise ValueError("time_index length must match PL series length")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, cum_pl)
    
    # Labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Profit / Loss")
    ax.set_title(title or "Cumulative PL Over Time – Feature 1")
    
    # Grid for readability
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return fig, ax


def visualize_opt_path_bary(path_bary, pscores, title=None, dirpath=None):
    """
    Plots the optimizer path defined by barycentric coordinates inside
    a regular tetrahedron.
    - path_bary: array-like Nx4 of barycentric coords (sum to 1 each row)
    - pscores:   length-N list/array of scores (0 best → 1 worst)
    - title:     optional title (also used for saving if dirpath given)
    - dirpath:   pathlib.Path or str directory to save the figure
    """
    # Regular tetrahedron vertices
    verts = np.array([
        [0.0, 0.0, 0.0],                  # v0
        [1.0, 0.0, 0.0],                  # v1
        [0.5, np.sqrt(3)/2, 0.0],         # v2
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)] # v3
    ])
    # Convert barycentric to Cartesian
    path_cart = np.dot(path_bary, verts)

    # Define edges by vertex indices
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    # Colormap (maroon shades)
    maroon_cmap = LinearSegmentedColormap.from_list('maroon', ['#f7e6e6', '#800000'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw tetrahedron edges
    for i, j in edges:
        xs, ys, zs = verts[[i, j]].T
        ax.plot(xs, ys, zs, color='black', linewidth=1, zorder=0)

    # Faint gray connecting line behind points
    ax.plot(path_cart[:,0], path_cart[:,1], path_cart[:,2],
            color='gray', alpha=0.2, linewidth=1, zorder=1)

    # Scatter points colored by score
    sc = ax.scatter(path_cart[:,0], path_cart[:,1], path_cart[:,2],
                    c=pscores, cmap=maroon_cmap, s=15, zorder=2)
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Score')

    # Equal aspect and labels
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    if title:
        ax.set_title(title)
    if dirpath is not None:
        import pathlib
        fn = f"{title or 'opt_path_bary'}.png"
        plt.savefig(str(pathlib.Path(dirpath) / fn), dpi=150)

    plt.show()
    plt.close()


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
    ax.plot(positions[:,0], positions[:,1], color='gray', alpha=0.1, linewidth=1, zorder=1)
    
    # Color-coded scatter
    scatter = ax.scatter(positions[:,0], positions[:,1], c=scores, cmap=maroon_cmap, zorder=2, s=6)
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

import numpy as np
import matplotlib.pyplot as plt

def plot_spec_ensemble_performances(datasets, n_points=300, info='EV', show_band=False, alpha_lines=0.3):
    """
    datasets: list of (xvals, yvals), one per ensemble
    n_points: number of points for interpolation grid
    show_band: if True, draw ±1 std dev band around the mean (normalized-space mean)
    alpha_lines: transparency for individual raw lines

    Plots:
      - all raw curves (faint) vs their original xvals
      - mean curve (computed in normalized space, mapped back)
      - median and 25/75% quantile curves (computed directly in raw-x space)
    Returns:
      x_mean_grid, y_mean (the normalized-mean overlay)
    """
    # Normalize each curve to [0,1] on its x-axis, interpolate onto a common normalized grid
    x_norm_grid = np.linspace(0, 1, n_points)

    Y_interp = []
    Xraw_at_grid = []

    # Collect all raw x-ranges to build a global grid for medians/quantiles
    all_x_min = min(np.min(x) for x, _ in datasets)
    all_x_max = max(np.max(x) for x, _ in datasets)
    x_raw_grid = np.linspace(all_x_min, all_x_max, n_points)
    Y_raw_interp = []  # y interpolated onto common raw-x grid

    plt.figure(figsize=(8, 5))

    for xvals, yvals in datasets:
        xvals = np.asarray(xvals)
        yvals = np.asarray(yvals)

        order = np.argsort(xvals)
        xvals = xvals[order]
        yvals = yvals[order]

        plt.plot(xvals, yvals, color='steelblue', alpha=alpha_lines, linewidth=1.5)

        x_min, x_max = xvals[0], xvals[-1]
        if x_max == x_min:
            continue

        # Normalized interpolation for mean
        x_norm = (xvals - x_min) / (x_max - x_min)
        y_interp_norm = np.interp(x_norm_grid, x_norm, yvals)
        Y_interp.append(y_interp_norm)
        Xraw_at_grid.append(x_min + x_norm_grid * (x_max - x_min))

        # Raw-grid interpolation for medians/quantiles
        y_interp_raw = np.interp(x_raw_grid, xvals, yvals)
        Y_raw_interp.append(y_interp_raw)

    if not Y_interp:
        plt.title("No valid curves")
        plt.show()
        return np.array([]), np.array([])

    Y_interp = np.vstack(Y_interp)
    Xraw_at_grid = np.vstack(Xraw_at_grid)

    # Mean across ensembles (normalized space)
    y_mean = Y_interp.mean(axis=0)
    y_std = Y_interp.std(axis=0)
    x_mean_grid = Xraw_at_grid.mean(axis=0)

    # Medians and quantiles across ensembles (raw space)
    Y_raw_interp = np.vstack(Y_raw_interp)
    y_median = np.median(Y_raw_interp, axis=0)
    y_q1 = np.percentile(Y_raw_interp, 25, axis=0)
    y_q3 = np.percentile(Y_raw_interp, 75, axis=0)

    if show_band:
        y_lo = y_mean - y_std
        y_hi = y_mean + y_std
        plt.fill_between(x_mean_grid, y_lo, y_hi, alpha=0.2, linewidth=0, label="±1 std (norm)")

    # Overlays
    plt.plot(x_mean_grid, y_mean, color='black', linewidth=2.5, label="Mean (normalized, mapped back)")
    plt.plot(x_raw_grid, y_median, color='black', linestyle='--', linewidth=2.0, label="Median (raw grid)")
    plt.plot(x_raw_grid, y_q1, color='black', linestyle=':', linewidth=1.8, label="25th / 75th percentile (raw grid)")
    plt.plot(x_raw_grid, y_q3, color='black', linestyle=':', linewidth=1.8)

    plt.title(f"{info} across ensembles (raw x) with normalized-mean and raw-median overlays")
    plt.xlabel("Voting threshold (raw)")
    plt.ylabel(info)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return x_mean_grid, y_mean



# NOTE LATEX OUTPUT WORK BEGIN NOTE # ___________________________________________

IDX2NAME = {
    0: r"\mathrm{time}",
    1: r"\mathrm{high}",
    2: r"\mathrm{low}",
    3: r"\mathrm{close}",
    4: r"\mathrm{volume}",
}

def _get(obj, name, default=None):
    return getattr(obj, name, default)

def _is_node(x):
    return hasattr(x, "_type")

def _fmt_num(x):
    if isinstance(x, (int, float)):
        return str(int(x)) if float(x).is_integer() else f"{x:.6g}"
    raise TypeError(f"Expected numeric constant, got {x!r}")

def _expand_raw_field(x):
    """Return the LaTeX name for a raw field (no time), e.g., \mathrm{close}."""
    if not isinstance(x, int) or x not in IDX2NAME:
        raise ValueError(f"type 0 expects _x in 0..4, got {x!r}")
    return IDX2NAME[x]

def _atom_at(x, tau: str, *, continuous: bool) -> str:
    """
    Render the FULL expression for x evaluated at 'tau'.
      - If x is a raw field node (_type==0): \mathrm{name}(tau) or \mathrm{name}_{tau}
      - If x is a composite node: ( to_latex(x) )(tau) or ( ... )_{tau}
      - If x is a bare index (0..4): same as raw field
      - If x is a string literal: literal(tau) or literal_{tau}
    continuous=True  -> use parentheses  (...)(tau)
    continuous=False -> use subscripts    ..._{tau}
    """
    # bare index
    if isinstance(x, int) and x in IDX2NAME:
        base = IDX2NAME[x]
        return f"{base}({tau})" if continuous else fr"{base}_{{{tau}}}"

    # literal symbol
    if isinstance(x, str):
        return f"{x}({tau})" if continuous else fr"{x}_{{{tau}}}"

    # node
    if _is_node(x):
        if _get(x, "_type") == 0:
            ix = _get(x, "_x")
            base = _expand_raw_field(ix)
            return f"{base}({tau})" if continuous else fr"{base}_{{{tau}}}"
        # composite expression: subscript/parenthesize the WHOLE thing
        inner = to_latex(x)  # recursively expanded LaTeX for the expression itself
        return (r"\left(" + inner + r"\right)" + f"({tau})") if continuous else \
               (r"\left(" + inner + r"\right)_{" + tau + r"}")

    raise ValueError(f"Unrecognized _x: {x!r}")

def _alpha_expr(alpha) -> str:
    return to_latex(alpha) if _is_node(alpha) else _fmt_num(alpha)

def _window_min(x, Delta):
    D = _fmt_num(Delta)
    # discrete window min over k -> x evaluated at t-k (expanded)
    return fr"\min_{{0\leq k< {D}}} " + _atom_at(x, r"t-k", continuous=False)

def _window_max(x, Delta):
    D = _fmt_num(Delta)
    return fr"\max_{{0\leq k< {D}}} " + _atom_at(x, r"t-k", continuous=False)

def _window_avg(x, Delta):
    D = _fmt_num(Delta)
    return fr"\frac{{1}}{{{D}}}\sum_{{k=0}}^{{{D}-1}} " + _atom_at(x, r"t-k", continuous=False)

def _value_t(x) -> str:
    return _atom_at(x, "t", continuous=False)

# --- main dispatcher (only the changed parts shown; keep your existing other cases) ---

def to_latex(node) -> str:
    t  = _get(node, "_type")
    x  = _get(node, "_x")
    a  = _get(node, "_alpha")
    d1 = _get(node, "_delta")
    d2 = _get(node, "_delta2")
    kp = _get(node, "_kappa")

    if t == 0:
        # raw field at time t
        return _value_t(node)  # this calls _atom_at(node, "t", False) and expands
    if t == 1:
        return _window_max(x, d1)
    if t == 2:
        return _window_min(x, d1)
    if t == 3:
        return _window_avg(x, d1)
    if t == 4:
        return r"-" + _value_t(x)
    if t == 5:
        return _value_t(x) + r" - " + _alpha_expr(a)
    if t == 6:
        return r"\left(" + _value_t(x) + r" - " + _alpha_expr(a) + r"\right)^{2}"
    if t == 7:
        base_t = _value_t(x)
        min_d1 = _window_min(x, d1)
        max_d2 = _window_max(x, d2)
        return (fr"\frac{{ {base_t} - \left({min_d1}\right) }}"
                fr"{{ \left({max_d2}\right) - \left({min_d1}\right) }} - \frac{{1}}{{2}}")

    if t == 8:
        # Continuous-time, finite start at 0; ALWAYS expand _x at s:
        # hkp(t) = e^{-κ t} hkp(0) + ∫_0^t e^{-κ(t-s)} e^{ x(s) } ds
        kappa = _fmt_num(kp)
        x_of_s = _atom_at(x, "k", continuous=True)  # fully expanded x(s)
        # If you assume h(0)=0, comment out the IC term.
        return (r"\int_{-\infty}^{t} e^{-" + kappa + r"(t-k)}\,e^{" + x_of_s + r"}\,dk")
        # If you prefer the discrete version (unit step), use:
        # x_tminus_i = _atom_at(x, r"t-i", continuous=False)
        # return (r"\mathrm{hkp}_t = e^{-" + kappa + r" t}\,\mathrm{hkp}_0 + "
        #         r"\sum_{i=0}^{t} e^{-" + kappa + r" i}\, e^{" + x_tminus_i + r"}")

    raise ValueError(f"Unknown _type: {t!r}")

from pathlib import Path as path
import os, time
import matplotlib as mpl

def render_latex(latex: str, filename: str | None = 'latextree.png', 
                 dpi: int = 220, fontsize: int = 24, times: bool = True) -> str:
    """
    Render a LaTeX math string to a PNG image using matplotlib's mathtext.
    
    Parameters
    ----------
    latex : str
        The LaTeX content. If it doesn't start with "$", it will be wrapped as inline math `$...$`.
    filename : str | None
        Output filename (PNG). If None, a timestamped name will be created under /mnt/data.
    dpi : int
        Image resolution.
    fontsize : int
        Font size for the rendered equation.
    times : bool
        If True, use a Times New Roman–like math font (STIX). Otherwise use the default (Computer Modern).
        
    Returns
    -------
    path : str
        Absolute path to the saved PNG file.
    """
    if not isinstance(latex, str):
        raise TypeError("latex must be a string")
        
    text = latex.strip()
    if not text.startswith("$"):
        text = f"${text}$"
    
    if filename is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"/mnt/data/latex_{stamp}.png"
    else:
        if not os.path.isabs(filename):
            filename = os.path.join(path.cwd(), filename)
        if not filename.lower().endswith(".png"):
            filename += ".png"
    
    # Configure font
    if times:
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
    else:
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'DejaVu Serif'
    
    # Render
    fig = plt.figure(figsize=(10, 2.5), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close(fig)
    return filename

# NOTE END LATEX WORK NOTE #_______________________________________________________________