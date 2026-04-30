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
def _(CanoeDynamics, hj, np):
    # This is where all of the work of HJR happens, namely computing the value function V


    def compute_value(vox=100, exp=6):
        # specify the dynamics we are considering
        model = CanoeDynamics()

        # specify the time horizon of the problem
        T = 5

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

    return (compute_value,)


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

        # Create coordinate mesh
        x = grid.states[:, 0, 0]
        y = grid.states[0, :, 1]

        X, Y = np.meshgrid(x, y)

        # Filled contour plot
        contour = ax.contourf(
            Y,
            X,
            V,
            levels=contour_levels,
            cmap=cmap,
        )

        # Zero level set (often the reachable set boundary)
        # ax.contour(
        #    X,
        #    Y,
        #    V,
        #    levels=[0],
        #    colors="red",
        #    linewidths=2,
        # )

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        title = r'$V(x,y, t=0)$' + '\n' + r'Penalty = $10^{' + str(exp) + r'}$, Res = $' + str(vox) + r'\times' + str(vox) + r'$'
        ax.set_title(title)
        ax.set_aspect("equal")

        thetas = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            [0 + 2 * np.cos(theta) for theta in thetas],
            [-3 + 2 * np.sin(theta) for theta in thetas],
            color="white",
        )

        if show_colorbar:
            fig.colorbar(contour, ax=ax, label="Value")

    return (plot_value_function,)


@app.cell
def _(compute_value):
    voxes = [500, 100, 20]
    exps = [2, 4, 6]

    Vs = [[None, None, None], [None, None, None], [None, None, None]]
    grids = [[None, None, None], [None, None, None], [None, None, None]]
    for i in [0,1,2]:
        for j in [0,1,2]:
            V, grid = compute_value(vox=voxes[i], exp=exps[j])
            Vs[i][j] = V[-1,...]
            grids[i][j] = grid
    return Vs, exps, grids, voxes


@app.cell
def _(Vs, exps, grids, plot_value_function, plt, voxes):
    fig, axs = plt.subplots(3,3, figsize = (30,30))

    for i1 in [0,1,2]:
        for j1 in [0,1,2]:
            plot_value_function(Vs[i1][j1], grids[i1][j1], fig, axs[i1,j1], vox = voxes[i1], exp = exps[j1])
    fig.tight_layout()
    plt.savefig('/Users/dylanhirsch/Desktop/sum_of_rewards.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make the grid less high-fidelity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now make a really low resolution grid
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Now let's actually use HJR
    """)
    return


@app.cell
def _(CanoeDynamics, hj, jnp, np):
    # This is where all of the work of HJR happens, namely computing the value function V


    def compute_value_HJR(vox=100):
        # specify the dynamics we are considering
        model = CanoeDynamics()

        # specify the time horizon of the problem
        T = 5

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

    return (compute_value_HJR,)


@app.cell
def _(compute_value_HJR, plot_value_function):
    W1, grid_1 = compute_value_HJR(vox=500)
    plot_value_function(
        W1[-1, ...], x=grid_1.states[:, 0, 0], y=grid_1.states[0, :, 1]
    )
    return


@app.cell
def _(compute_value_HJR, plot_value_function):
    W2, grid_2 = compute_value_HJR(vox=100)
    plot_value_function(
        W2[-1, ...], x=grid_2.states[:, 0, 0], y=grid_2.states[0, :, 1]
    )
    return


@app.cell
def _(compute_value_HJR, plot_value_function):
    W3, grid_3 = compute_value_HJR(vox=20)
    plot_value_function(
        W3[-1, ...], x=grid_3.states[:, 0, 0], y=grid_3.states[0, :, 1]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
