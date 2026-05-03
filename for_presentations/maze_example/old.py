import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from matplotlib.patches import FancyArrowPatch

    # fundamental numerical library
    import hj_reachability as hj
    import numpy as np
    import jax.numpy as jnp
    import scipy as sp

    # my dynamics
    from roomba import RoombaDynamics
    from util.closed_loop import ClosedLoopTrajectory
    from util.maze_builder import Smaze

    # plotting
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 20}
    plt.rc("font", **font)
    return (
        ClosedLoopTrajectory,
        FancyArrowPatch,
        LinearSegmentedColormap,
        RoombaDynamics,
        Smaze,
        TwoSlopeNorm,
        hj,
        jnp,
        np,
        plt,
    )


@app.cell
def _(ClosedLoopTrajectory, RoombaDynamics, Smaze, hj, jnp, np):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the dynamics we are considering
    model = RoombaDynamics()
    maze = Smaze()

    # specify the time horizon of the problem
    T = 200

    # Initial state
    x0 = np.array((-9, 9))


    def compute_value(res=100, Lambda=0.5):

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
        walls = maze.obstacle_sdf(grid)
        goal = maze.target_sdf(grid)

        q = np.where(
            walls <= 0.0,
            -np.ones(walls.shape),
            0.0 * np.ones(walls.shape),
        )

        r = np.where(
            goal > 0.0,
            np.ones(goal.shape),
            0.0 * np.ones(goal.shape),
        )

        def hamiltonian_postprocessor(H):
            return H + (1 - Lambda) * r + Lambda * q

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", hamiltonian_postprocessor=hamiltonian_postprocessor
        )

        # solve for the value function
        V = hj.solve(solver_settings, model, grid, times, 0.0 * r)

        cl = ClosedLoopTrajectory(
            model, grid, times, V, initial_state=x0, steps=1000
        )

        return V, grid, cl


    def compute_value_HJR(res=100, Lambda=0.5):

        # specify the number of voxels to divide the spatial and temporal axes:
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

        # specify the target and obstacle
        q = maze.obstacle_sdf(grid)
        r = maze.target_sdf(grid)

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

        cl = ClosedLoopTrajectory(
            model, grid, times, V, initial_state=x0, steps=1000
        )

        return V, grid, cl

    return T, compute_value, compute_value_HJR


@app.cell
def _(compute_value, compute_value_HJR):
    res = 160
    Lambdas1 = [0.9, 0.999, 0.99999]
    Lambdas2 = [0.1, 0.5, 0.9]

    Vs = []
    grids = []
    cls = []
    HJR_Vs = []
    HJR_grids = []
    HJR_cls = []

    for i in [0, 1, 2]:
        V, grid, cl = compute_value(res=res, Lambda=Lambdas1[i])
        Vs.append(V[-1, ...])
        grids.append(grid)
        cls.append(cl)
        V, grid, cl = compute_value_HJR(res=res, Lambda=Lambdas2[i])
        HJR_Vs.append(V[-1, ...])
        HJR_grids.append(grid)
        HJR_cls.append(cl)
    return HJR_Vs, HJR_cls, HJR_grids, Lambdas1, Lambdas2, Vs, cls, grids


@app.cell
def _(LinearSegmentedColormap, T, TwoSlopeNorm, np):
    def plot_value_function(
        V,
        grid,
        cl,
        fig,
        ax,
        cmap="viridis",
        show_colorbar=True,
        contour_levels=20,
        figsize=(6, 5),
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
            vmin=min(V.min(), -0.05), vcenter=0.0, vmax=max(V.max(), +0.05)
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
            ax.set_title(r"$V(x,y,t=T - " + str(T) + r")$")
        ax.set_aspect("equal")

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

    return (plot_value_function,)


@app.cell
def _(
    FancyArrowPatch,
    HJR_Vs,
    HJR_cls,
    HJR_grids,
    Lambdas1,
    Lambdas2,
    Vs,
    cls,
    grids,
    plot_value_function,
    plt,
):
    def _(Vs, grids, cls, Lambdas, term=r"$\lambda$", level=0.0):

        fig, axs = plt.subplots(3, 1, figsize=(7, 21))

        # i1 = row = penalty index, j1 = col = resolution index
        for i1 in [0, 1, 2]:
            plot_value_function(
                Vs[i1],
                grids[i1],
                cls[i1],
                fig,
                axs[i1],
                show_title=True,
                level=level,
            )

        # Reserve space: left for row labels/arrow, top for col labels/arrow
        fig.tight_layout(rect=[0.12, 0.02, 0.98, 0.90])

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


    fig, axs = _(Vs, grids, cls, Lambdas1, level=0.00)
    fig.savefig(
        "/Users/dylanhirsch/Desktop/reach_avoid_sum_of_rewards.png",
        bbox_inches="tight",
    )

    fig, axs = _(HJR_Vs, HJR_grids, HJR_cls, Lambdas2)
    fig.savefig(
        "/Users/dylanhirsch/Desktop/reach_avoid_hjr.png", bbox_inches="tight"
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
