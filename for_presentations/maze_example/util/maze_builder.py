import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

class Wall:

    def __init__(self, x0: float, y0: float, x1: float, y1: float):

        """
        Creates a 2 wall of the maze. Note that walls have thickness.
        (x0, y0): bottom left corner of wall.
        (x1, y1): top right corner of wall.
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.cx = (x0 + x1) / 2
        self.cy = (y0 + y1) / 2

        self.hx = (x1 - x0) / 2 
        self.hy = (y1 - y0) / 2


    def __call__(self, grid):
        X = grid.states[...,0]
        Y = grid.states[...,1]
        qx = np.abs(X - self.cx) - self.hx
        qy = np.abs(Y - self.cy) - self.hy

        sdf = np.sqrt(np.maximum(qx, 0.0 * qx)**2 + np.maximum(qy, 0.0 * qy)**2) + np.minimum(np.maximum(qx, qy), 0.0 * qx)
        return sdf

class Maze:

    def __init__(self):

        self.walls = []

    def __call__(self, grid):

        sdf = np.inf * np.ones(grid.shape[:-1])
        for wall in self.walls:
            sdf = np.minimum(sdf, wall(grid))

        return sdf


    def add(self, wall):
        self.walls.append(wall)



if __name__ == '__main__':
    res = 100

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

    maze = Maze()
    wall = Wall(-9., -9., -8., 0.)
    maze.add(wall)
    value = maze(grid)

    # Plot

    x = grid.states[:, 0, 0]
    y = grid.states[0, :, 1]
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(1,1)

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        ["red", "white", "green"]
    )

    # Center at zero
    norm = TwoSlopeNorm(vmin=value.min(), vcenter=0.0, vmax=value.max())

    mesh = ax.pcolormesh(
        y,
        x,
        value.T,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    zero_contour = ax.contour(
        Y, X, value, levels=[0.0], colors="black", linestyles="--", linewidths=2
    )

    fig.savefig('/Users/dylanhirsch/Desktop/hi.png')
