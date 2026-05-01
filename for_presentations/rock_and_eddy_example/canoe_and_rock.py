import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # fundamental numerical library
    import hj_reachability as hj
    import numpy as np
    import jax.numpy as jnp
    import scipy as sp

    # my dynamics
    from canoe import CanoeDynamics

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 20}
    plt.rc("font", **font)
    return CanoeDynamics, hj, jnp, mo, np, plt


@app.cell
def _(CanoeDynamics, hj, jnp, np):
    # This is where all of the work of HJR happens, namely computing the value function V


    def compute_value(vox=100, exp=6):
        # specify the dynamics we are considering
        model = CanoeDynamics()

        # specify the time horizon of the problem
        T = 20

        # specify the number of voxels to divide the spatial and temporal axes
        x_voxels = vox
        y_voxels = vox
        t_voxels = 2 * vox

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
            np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 3.0) ** 2)
            - 2.0
        )
        r = np.where(
            rock < 0.0,
            -(10.0**exp) * np.ones(rock.shape),
            0.0 * np.ones(rock.shape),
        )

        def hamiltonian_postprocessor(H):
            return H + np.minimum(r, 0.0 * r)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", hamiltonian_postprocessor=hamiltonian_postprocessor
        )

        # solve for the value function
        V = hj.solve(solver_settings, model, grid, times, 0.0 * r)

        return V, grid


    def compute_value_HJR(vox=100):
        # specify the dynamics we are considering
        model = CanoeDynamics()

        # specify the time horizon of the problem
        T = 20

        # specify the number of voxels to divide the spatial and temporal axes
        x_voxels = vox
        y_voxels = vox
        t_voxels = 2 * vox

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

        # specify the rock
        g = (
            np.sqrt((grid.states[..., 0]) ** 2 + (grid.states[..., 1] + 3.0) ** 2)
            - 2.0
        )

        def value_postprocessor(t, V):
            return jnp.minimum(V, g)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", value_postprocessor=value_postprocessor
        )

        # solve for the value function
        V = hj.solve(solver_settings, model, grid, times, g)

        return V, grid

    return compute_value, compute_value_HJR


@app.cell
def _(compute_value, compute_value_HJR):
    voxes = [500, 100, 20]
    exps = [2, 4, 6]

    Vs = []
    grids = []
    HJR_Vs = []
    HJR_grids = []
    for i in [0, 1, 2]:
        Vs.append([])
        grids.append([])
        for j in [0, 1, 2]:
            V, grid = compute_value(vox=voxes[i], exp=exps[j])
            Vs[i].append(V[-1, ...])
            grids[i].append(grid)
        V, grid = compute_value_HJR(vox=voxes[i])
        HJR_Vs.append(V)
        HJR_grids.append(grid)
    return HJR_Vs, HJR_grids, Vs, exps, grids, voxes


@app.cell
def _(np):
    def plot_value_function(
        V,
        grid,
        fig,
        ax,
        vox,
        exp,
        cmap="viridis",
        show_colorbar=True,
        contour_levels=20,
        figsize=(6, 5),
        show_title=True,
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

        mesh = ax.pcolormesh(
            y, x, V.T,
            cmap=cmap,
            vmin=-10.0,
            vmax=10.0,
            shading="auto",
        )

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if show_title:
            ax.set_title(r"$V(x,y,t=T - 20)$")
        ax.set_aspect("equal")

        thetas = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            [0 + 2 * np.cos(theta) for theta in thetas],
            [-3 + 2 * np.sin(theta) for theta in thetas],
            color="white",
        )

        if show_colorbar:
            fig.colorbar(mesh, ax=ax, label="Value", fraction=0.035, pad=0.04)

    return (plot_value_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First let's solve the running cost problem
    """)
    return


@app.cell
def _(Vs, exps, grids, plot_value_function, plt, voxes):
    from matplotlib.patches import FancyArrowPatch

    fig, axs = plt.subplots(3, 3, figsize=(21, 21))

    # i1 = row = penalty index, j1 = col = resolution index
    for i1 in [0, 1, 2]:
        for j1 in [0, 1, 2]:
            plot_value_function(
                Vs[j1][i1],
                grids[j1][i1],
                fig,
                axs[i1, j1],
                vox=voxes[j1],
                exp=exps[i1],
                show_title=True,
            )

    # Reserve space: left for row labels/arrow, top for col labels/arrow
    fig.tight_layout(rect=[0.12, 0.02, 0.98, 0.90])

    # --- Row labels (penalty) on the left ---
    for i1, exp in enumerate(exps):
        pos = axs[i1, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            0.07,
            y_mid,
            r"Penalty $= 10^{" + str(exp) + r"}$",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

    # --- Column labels (resolution) on the top ---
    for j1, vox in enumerate(voxes):
        pos = axs[0, j1].get_position()
        x_mid = (pos.x0 + pos.x1) / 2
        fig.text(
            x_mid,
            0.935,
            r"$" + str(vox) + r"\times" + str(vox) + r"$",
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
        "Increasing Penalty",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )

    plt.savefig(
        "/Users/dylanhirsch/Desktop/sum_of_rewards.svg", bbox_inches="tight"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Now let's actually use HJR
    """)
    return


@app.cell
def _(HJR_Vs, HJR_grids, plot_value_function, plt, voxes):
    from matplotlib.patches import FancyArrowPatch as _FAP

    fig_hjr, axs_hjr = plt.subplots(1, 3, figsize=(21, 7))

    for j1_hjr in [0, 1, 2]:
        plot_value_function(
            HJR_Vs[j1_hjr][-1, ...],
            HJR_grids[j1_hjr],
            fig_hjr,
            axs_hjr[j1_hjr],
            vox=voxes[j1_hjr],
            exp=None,
            show_title=True,
        )

    fig_hjr.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])

    # --- Column labels (resolution) on the top ---
    for j1_hjr, vox_hjr in enumerate(voxes):
        pos_hjr = axs_hjr[j1_hjr].get_position()
        x_mid_hjr = (pos_hjr.x0 + pos_hjr.x1) / 2
        fig_hjr.text(
            x_mid_hjr, 0.935,
            r"$" + str(vox_hjr) + r"\times" + str(vox_hjr) + r"$",
            ha="center", va="center", fontsize=26, fontweight="bold",
            transform=fig_hjr.transFigure,
        )

    # --- Arrow: Decreasing Resolution (left → right, above column labels) ---
    lp = axs_hjr[0].get_position()
    rp = axs_hjr[2].get_position()
    fig_hjr.add_artist(_FAP(
        posA=(lp.x0, 0.96), posB=(rp.x1, 0.96),
        transform=fig_hjr.transFigure,
        arrowstyle="->", mutation_scale=30, lw=2.5, color="black",
    ))
    fig_hjr.text(
        (lp.x0 + rp.x1) / 2, 0.975,
        "Decreasing Resolution",
        ha="center", va="center", fontsize=28, fontweight="bold",
        transform=fig_hjr.transFigure,
    )

    plt.savefig("/Users/dylanhirsch/Desktop/hjr_value_functions.svg", bbox_inches="tight")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
