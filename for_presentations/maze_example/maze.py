# fundamental numerical library
import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.animation as animation

# plotting
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

# my dynamics
from roomba import RoombaDynamics
from util.closed_loop import ClosedLoopTrajectory
from util.maze_builder import Smaze

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
font = {"size": 20}
plt.rc("font", **font)

# specify the dynamics we are considering
MODEL = RoombaDynamics()

# specify the time horizon of the problem
T = 200

# Initial state
X0 = np.array((0.0, 9.0))

# Number of steps to take in closed loop
STEPS = 10


def compute_value(res=100, Lambda=1.0):

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = res
    y_voxels = res
    t_voxels = 400

    # Specify PDE grid corners
    x_min = -10
    y_min = -10

    x_max = +10
    y_max = +10

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min], [x_max, y_max]),
        [x_voxels + 1, y_voxels + 1],
    )
    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the running cost
    smaze = Smaze()
    walls = smaze.obstacle_sdf(grid)
    target = smaze.target_sdf(grid)

    q = np.where(
        walls <= 0.0,
        -np.ones(walls.shape),
        0.0 * np.ones(walls.shape),
    )

    r = np.where(
        target > 0.0,
        np.ones(target.shape),
        0.0 * np.ones(target.shape),
    )

    def hamiltonian_postprocessor(H):
        return H + r + Lambda * q

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", hamiltonian_postprocessor=hamiltonian_postprocessor
    )

    # solve for the value function
    V = hj.solve(solver_settings, MODEL, grid, times, 0.0 * r)

    cl = ClosedLoopTrajectory(MODEL, grid, times, V, initial_state=X0, steps=STEPS)

    return V, grid, cl


def compute_value_HJR(res=100):

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = res
    y_voxels = res
    t_voxels = 400

    # Specify PDE grid corners
    x_min = -10
    y_min = -10

    x_max = +10
    y_max = +10

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min], [x_max, y_max]),
        [x_voxels + 1, y_voxels + 1],
    )
    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the running cost
    smaze = Smaze()
    walls = smaze.obstacle_sdf(grid)
    target = smaze.target_sdf(grid)

    q = walls 
    r = target

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, r), q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    # solve for the value function
    V = hj.solve(
        solver_settings,
        MODEL,
        grid,
        times,
        jnp.minimum(r,q),
    )

    cl = ClosedLoopTrajectory(MODEL, grid, times, V, initial_state=X0, steps = STEPS)

    return V, grid, cl


# %%
def plot_value_function(
    V,
    grid,
    cl,
    fig,
    ax,
    show_colorbar=True,
    show_title=True,
    level=0.0,
):
    """
    Plot a 2D HJ reachability value function.

    Parameters
    ----------
    V : (N, M) ndarray
        2D value function array.

    grid: from HJR output.

    title : str
        Plot title.

    cmap : str
        Matplotlib colormap.

    show_colorbar : bool
        Whether to display a colorbar.

    contour_levels : int
        Number of contour levels.

    figsize : tuple
        Figure size.
    """

    V = np.asarray(V)
    V = np.clip(V, -10.0, +10.0)

    x = grid.states[:, 0, 0]
    y = grid.states[0, :, 1]
    X, Y = np.meshgrid(x, y)

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green", ["red", "white", "green"]
    )

    # Center at zero
    norm = TwoSlopeNorm(vmin=min(V.min(), -1.00), vcenter=0.0, vmax=max(V.max(), +1.00))

    mesh = ax.pcolormesh(
        y,
        x,
        V.T,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if show_title:
        ax.set_title(r"$V(x,y,t=T - 20)$")
    ax.set_aspect("equal")

    zero_contour = ax.contour(
        Y, X, V, levels=[level], colors="black", linestyles="--", linewidths=2
    )

    if show_colorbar:
        fig.colorbar(mesh, ax=ax, label="Value", fraction=0.035, pad=0.04)

    ts = np.linspace(-T, 0, 100)
    #ax.plot(
    #    [cl.x(t)[0] for t in ts],
    #    [cl.x(t)[1] for t in ts],
    #    color="white",
    #    linewidth=3,
    #)


# %%
def plot_grid(
    Vs,
    grids,
    cls,
    Lambdas,
    val_slice=-1,
    term=r"$\lambda$",
    level=0.0,
    show_colorbar=True,
):
    fig, axs = plt.subplots(3, 1, figsize=(9, 27))

    for i1 in range(3):
        plot_value_function(
            Vs[i1][val_slice, ...],
            grids[i1],
            cls[i1],
            fig,
            axs[i1],
            show_colorbar=show_colorbar,
            show_title=True,
            level=level,
        )

    # Reserve space on the left for row labels and arrow
    fig.tight_layout(rect=[0.12, 0.02, 0.98, 0.98])

    # --- Row labels (penalty) on the left ---
    for i1, Lambda in enumerate(Lambdas):
        pos = axs[i1].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            0.07,
            y_mid,
            term + r" $= " + str(Lambda) + "$",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

    # --- Arrow: Increasing Penalty (top → bottom, left of row labels) ---
    top_pos = axs[0].get_position()
    bot_pos = axs[2].get_position()
    arrow_x_left = 0.025
    fig.add_artist(
        FancyArrowPatch(
            posA=(arrow_x_left, top_pos.y1),
            posB=(arrow_x_left, bot_pos.y0),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=30,
            lw=2.5,
            color="black",
        )
    )
    fig.text(
        0.012,
        (top_pos.y1 + bot_pos.y0) / 2,
        "Increasing " + term,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )

    return fig, axs


def make_movie(
    Vs,
    grids,
    cls,
    Lambdas,
    output_path,
    n_frames=200,
    fps=20,
    level=0.0,
    term=r"$\lambda$",
):
    """Animate the value function column from val_slice=0 (t=0) to val_slice=-1 (t=-T)."""

    n_t = Vs[0].shape[0]
    frame_indices = np.round(np.linspace(0, n_t - 1, n_frames)).astype(int)

    fig, axs = plot_grid(
        Vs, grids, cls, Lambdas,
        val_slice=0, term=term, level=level, show_colorbar=False,
    )

    def update(frame_k):
        idx = frame_indices[frame_k]
        t_frac = idx / (n_t - 1)
        t_current = -T * t_frac

        for i1 in range(3):
            axs[i1].cla()
            plot_value_function(
                Vs[i1][idx, ...],
                grids[i1],
                cls[i1],
                fig,
                axs[i1],
                show_colorbar=False,
                show_title=False,
                level=level,
            )
            axs[i1].set_title(rf"$V(x,y,\;t={t_current:.1f})$")

        return list(axs)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Movie saved to {output_path}")


def make_movie_single(
    V,
    grid,
    cl,
    output_path,
    n_frames=200,
    fps=20,
    level=0.0,
):
    """Animate a single value function plot from t=0 to t=-T."""

    n_t = V.shape[0]
    frame_indices = np.round(np.linspace(0, n_t - 1, n_frames)).astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plot_value_function(V[0, ...], grid, cl, fig, ax, show_colorbar=False, show_title=False, level=level)

    def update(frame_k):
        idx = frame_indices[frame_k]
        t_current = -T * idx / (n_t - 1)
        ax.cla()
        plot_value_function(V[idx, ...], grid, cl, fig, ax, show_colorbar=False, show_title=False, level=level)
        ax.set_title(rf"$V(x,y,\;t={t_current:.1f})$")
        return [ax]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Movie saved to {output_path}")


if __name__ == "__main__":
    # %%
    res = 160
    Lambdas = [10, 1000, 100_000]

    Vs = []
    grids = []
    cls = []
    for Lambda in Lambdas:
        V, grid, cl = compute_value(res=res, Lambda=Lambda)
        Vs.append(V)
        grids.append(grid)
        cls.append(cl)

    make_movie(
        Vs,
        grids,
        cls,
        Lambdas,
        output_path="/Users/dylanhirsch/Desktop/reach_avoid_sum_of_rewards.mp4",
        level=0.1,
    )

    V_hjr, grid_hjr, cl_hjr = compute_value_HJR(res=res)
    make_movie_single(
        V_hjr,
        grid_hjr,
        cl_hjr,
        output_path="/Users/dylanhirsch/Desktop/reach_avoid_hjr.mp4",
        level=0.0,
    )
