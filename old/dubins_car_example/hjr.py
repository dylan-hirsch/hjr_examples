import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np

    from IPython.display import HTML
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import matplotlib.animation as anim
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import dynamics

    import hj_reachability as hj

    import random

    random.seed(123)
    np.random.seed(123)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 20}
    plt.rc("font", **font)
    return (
        FFMpegWriter,
        FuncAnimation,
        LinearSegmentedColormap,
        TwoSlopeNorm,
        dynamics,
        hj,
        jnp,
        np,
        plt,
    )


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.model()

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-3, -3, -np.pi]), jnp.array([+3, +3, np.pi])),
        (51, 51, 51),
        periodic_dims=[2],
    )

    l = np.linalg.norm(grid.states[..., [0, 1]] - np.array([0.0, 0.0]), axis=-1) - 1

    solver_settings = hj.SolverSettings.with_accuracy("very_high")
    return grid, l, model, solver_settings


@app.cell
def _(grid, hj, l, model, np, solver_settings):
    t0 = -np.pi
    times = np.linspace(0.0, t0, 20)
    V0 = hj.solve(solver_settings, model, grid, times, l)
    return (V0,)


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    LinearSegmentedColormap,
    TwoSlopeNorm,
    V0,
    np,
    plt,
):
    # Example 3D tensor: shape (n_time, n_rows, n_cols)
    V = V0[:, :, :, 25]
    n_time, n_rows, n_cols = V.shape

    # Set up the figure and initial heatmap
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    colors = ["cyan", "white", "pink"]
    cmap = LinearSegmentedColormap.from_list("custom_blue_black_white", colors, N=256)
    vmin = np.min(V)
    vmax = np.max(V)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=vmax)

    im = ax.imshow(V[0], cmap=cmap, origin="lower", norm=norm, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$V(x,t)$")

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    contour = ax.contour(
        V[0], levels=[0], colors="black", linewidths=0.7, origin="lower"
    )

    # Update function for animation
    def update(frame):
        ax.clear()
        im = ax.imshow(V[frame], cmap=cmap, origin="lower", norm=norm, aspect="auto")
        new_contour = ax.contour(
            V[frame], levels=[0], colors="black", linewidths=0.7, origin="lower"
        )

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

        return [im, new_contour]

    # Create animation
    ani = FuncAnimation(fig, update, frames=n_time, blit=True, interval=100)

    # To display in a Jupyter notebook (optional)
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())

    # Save animation to MP4 (requires ffmpeg installed)
    writer = FFMpegWriter(fps=10, bitrate=1800)
    ani.save("/Users/dylanhirsch/Desktop/tensor_evolution.mp4", writer=writer)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
