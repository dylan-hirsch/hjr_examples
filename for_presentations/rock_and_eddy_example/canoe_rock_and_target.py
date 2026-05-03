# fundamental numerical library
import hj_reachability as hj
import numpy as np
import jax.numpy as jnp
import scipy as sp

# my dynamics
from canoe import CanoeDynamicsBall
from closed_loop import ClosedLoopTrajectory

# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
font = {"size": 20}
plt.rc("font", **font)

# specify the dynamics we are considering
MODEL = CanoeDynamicsBall()

# specify the time horizon of the problem
T = 20

# Initial state
X0 = np.array((0.0, 9.0))

# Number of steps to take in closed loop
STEPS = 10


def compute_value(res = 100, Lambda = 1.0):

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = res
    y_voxels = res
    t_voxels = 2 * res

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
    rock = (
        np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 0.0) ** 2)
        - 5.0
    )

    target = (
        -np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 7.0) ** 2)
        + 2.0
    )

    q = np.where(
        rock <= 0.0,
        -np.ones(rock.shape),
        0.0 * np.ones(rock.shape),
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

    cl = ClosedLoopTrajectory(MODEL, grid, times, V, initial_state=X0, steps = STEPS)

    return V, grid, cl


def compute_value_HJR(res = 100, Lambda = 1.0):

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = res
    y_voxels = res
    t_voxels = 2 * res

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

    # specify the target and obstacle
    q = (
        np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 0.0) ** 2)
        - 5.0
    )

    r = (
        -np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 7.0) ** 2)
        + 2.0
    )

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, Lambda * r), (1 - Lambda) * q)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    # solve for the value function
    V = hj.solve(
        solver_settings,
        model,
        grid,
        times,
        jnp.minimum(Lambda * r, (1 - Lambda) * q),
    )

    cl = ClosedLoopTrajectory(model, grid, times, V, initial_state=x0)

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
    norm = TwoSlopeNorm(
        vmin=min(V.min(), -1.00), vcenter=0.0, vmax=max(V.max(), +1.00)
    )

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

    thetas = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        [0.0 + 5 * np.cos(theta) for theta in thetas],
        [0.0 + 5 * np.sin(theta) for theta in thetas],
        color="red",
    )
    ax.plot(
        [0 + 2 * np.cos(theta) for theta in thetas],
        [-7 + 2 * np.sin(theta) for theta in thetas],
        color="green",
    )

    zero_contour = ax.contour(
        Y, X, V, levels=[level], colors="black", linestyles="--", linewidths=2
    )

    if show_colorbar:
        fig.colorbar(mesh, ax=ax, label="Value", fraction=0.035, pad=0.04)

    ts = np.linspace(-T, 0, 100)
    ax.plot(
        [cl.x(t)[0] for t in ts],
        [cl.x(t)[1] for t in ts],
        color="white",
        linewidth=3,
    )

# %%
def plot_grid(Vs, grids, cls, Lambdas, reses, val_slice=-1, term=r"$\lambda$", level=0.0, show_colorbar=True):

    fig, axs = plt.subplots(3, 3, figsize=(21, 21))

    # i1 = row = penalty index, j1 = col = resolution index
    for i1 in [0, 1, 2]:
        for j1 in [0, 1, 2]:
            plot_value_function(
                Vs[j1][i1][val_slice, ...],
                grids[j1][i1],
                cls[j1][i1],
                fig,
                axs[i1, j1],
                show_colorbar=show_colorbar,
                show_title=True,
                level=level,
            )

    # Reserve space: left for row labels/arrow, top for col labels/arrow
    fig.tight_layout(rect=[0.12, 0.02, 0.98, 0.90])

    # --- Row labels (penalty) on the left ---
    for i1, Lambda in enumerate(Lambdas):
        pos = axs[i1, 0].get_position()
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

    # --- Column labels (resolution) on the top ---
    for j1, res in enumerate(reses):
        pos = axs[0, j1].get_position()
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(
            x_mid,
            0.935,
            r"$" + str(res) + r"\times" + str(res) + r"$",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            transform=fig.transFigure,
        )

    # --- Arrow: Decreasing Resolution (left → right, above column labels) ---
    left_pos = axs[0, 0].get_position()
    right_pos = axs[0, 2].get_position()
    arrow_y_top = 0.96
    fig.add_artist(
        FancyArrowPatch(
            posA=(left_pos.x0, arrow_y_top),
            posB=(right_pos.x1, arrow_y_top),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=30,
            lw=2.5,
            color="black",
        )
    )
    fig.text(
        (left_pos.x0 + right_pos.x1) / 2,
        0.975,
        "Decreasing Resolution",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        transform=fig.transFigure,
    )

    # --- Arrow: Increasing Penalty (top → bottom, left of row labels) ---
    top_pos = axs[0, 0].get_position()
    bot_pos = axs[2, 0].get_position()
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
        fontweight="bold",\
        rotation=90,
        transform=fig.transFigure,
    )

    return fig, axs


def make_movie(Vs, grids, cls, Lambdas, reses, output_path, n_frames=200, fps=20, level=0.0, term=r"$\lambda$"):
    """Animate the value function grid from val_slice=-1 (t=-T) to val_slice=0 (t=0)."""

    # Number of time steps for the finest resolution (index 0)
    n_t = Vs[0][0].shape[0]

    # Map each frame to a time index: start at n_t-1 (earliest), end at 0 (terminal)
    frame_indices = np.round(np.linspace(0, n_t - 1, n_frames)).astype(int)

    # Build the figure once (no colorbars so layout stays fixed across frames)
    fig, axs = plot_grid(
        Vs, grids, cls, Lambdas, reses,
        val_slice=-1, term=term, level=level, show_colorbar=False,
    )

    def update(frame_k):
        idx_finest = frame_indices[frame_k]
        # Fractional position in time: 1.0 = earliest (t=-T), 0.0 = terminal (t=0)
        t_frac = idx_finest / (n_t - 1)
        t_current = -T * t_frac

        for i1 in range(3):
            for j1 in range(3):
                axs[i1, j1].cla()
                n_t_ij = Vs[j1][i1].shape[0]
                idx_ij = round(t_frac * (n_t_ij - 1))
                plot_value_function(
                    Vs[j1][i1][idx_ij, ...],
                    grids[j1][i1],
                    cls[j1][i1],
                    fig,
                    axs[i1, j1],
                    show_colorbar=False,
                    show_title=False,
                    level=level,
                )
                axs[i1, j1].set_title(rf"$V(x,y,\;t={t_current:.1f})$")

        return list(axs.flat)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Movie saved to {output_path}")


if __name__ == '__main__':
    # %%
    reses = [100, 50, 25]
    Lambdas = [10, 1000, 100_000]

    Vs = []
    grids = []
    cls = []
    for i in [0, 1, 2]:
        Vs.append([])
        grids.append([])
        cls.append([])
        for j in [0, 1, 2]:
            V, grid, cl = compute_value(res=reses[i], Lambda=Lambdas[j])
            Vs[i].append(V)
            grids[i].append(grid)
            cls[i].append(cl)

    make_movie(
        Vs, grids, cls, Lambdas, reses,
        output_path="/Users/dylanhirsch/Desktop/reach_avoid_sum_of_rewards.mp4",
        level=0.1,
    )

