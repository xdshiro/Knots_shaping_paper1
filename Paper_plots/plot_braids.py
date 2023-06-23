from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg
import plotly.graph_objects as go
import numpy as np
# from vispy import app, gloo, visuals, scene
# from mayavi import mlab
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.spatial import distance as dist
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from simanneal import Annealer
import random
# import mlrose
import mlrose_hiive as mlrose
from itertools import combinations

def plot_3D_dots_go(dots, mode='markers', marker=None, fig=None, show=False, **kwargs):
    """
    plotting dots in the interactive window in browser using plotly.graph_objects
    :param dots: [[x,y,z],...]
    :param show: True if you want to show it instantly
    :return: fig
    """
    if marker is None:
        marker = {'size': 8, 'color': 'black'}
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=dots[:, 0], y=dots[:, 1], z=dots[:, 2],
                               mode=mode, marker=marker, **kwargs))
    if show:
        fig.show()
    return fig


def plotDots(dots, dots_bound=None, show=True, color='black', size=15, width=185, fig=None,
             save=None):
    """
    Function plots the array of dots in a beautiful and interactive way in your browser.
    Plots both numpy array and dict
    :param dots: array of dots
    :param dots_bound: which dots to use to get the box frames. By default it uses the dots itself,
        but if you want to make the frames the same with other plots, you can use other dots here, same
        for all plots.
    :param show: plotting in the browser. Can be turned off to add extra plots on top
    :param color: color of the dots
    :param size: size of the dots
    :param width: width of the shell of the dots (for a better visualization)
    :param fig: figure for the combination with other plots
    :return: fig
    """
    if isinstance(dots, dict):
        dots = np.array([dot for dot in dots])
    if isinstance(dots_bound, dict):
        dots_bound = np.array([dot for dot in dots_bound])
    if dots_bound is None:
        dots_bound = dots
    if fig is None:
        fig = plot_3D_dots_go(dots, marker={'size': size, 'color': color,
                                            'line': dict(width=width, color='white')})
    else:
        plot_3D_dots_go(dots, fig=fig, marker={'size': size, 'color': color,
                                               'line': dict(width=width, color='white')})
    pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    return fig


def plot_line_colored(dots, dots_bound=None, show=True, color=(0, 'black'), width=25, fig=None, save=None):
    if dots_bound is None:
        dots_bound = dots
    x, y, z = dots[:, 0], dots[:, 1], dots[:, 2]
    redscale = color
    color = z
    trace_curve = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=width, color=color, colorscale=redscale, colorbar=dict(thickness=20)),
        name='curve'
    )
    sphere_size = width / 3.5  # adjust to match the line width
    trace_spheres = go.Scatter3d(
        x=[x[0], x[-1]], y=[y[0], y[-1]], z=[z[0], z[-1]],
        mode='markers',
        marker=dict(size=sphere_size, color=[color[0], color[-1]], colorscale=redscale),
        name='line ends'
    )
    if fig is None:
        fig = go.Figure()
    # fig = go.Figure(data=[trace_curve, trace_spheres])
    
    fig.add_trace(trace_curve)
    fig.add_trace(trace_spheres)
    # fig = plot_3D_dots_go(dots, marker={'size': size, 'color': color,
    #                                     'line': dict(width=width, color='white')})
    # plot_3D_dots_go(dots, fig=fig, marker={'size': size, 'color': color,
    #                                        'line': dict(width=width, color='white')})
    pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    return fig


def plot_cylinder(radius=1., height=2., center=(0, 0, 0), segments=50, fig=None,
                  save=None, show=False, dots_bound=None, aspects=(2, 2, 2)):
    theta = np.linspace(0, 2 * np.pi, segments)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z_bottom = np.full_like(theta, center[2] - height / 2)
    z_top = np.full_like(theta, center[2] + height / 2)
    
    # The vertices are the points on the bottom and top circles of the cylinder
    vertices = np.column_stack([np.append(x, x), np.append(y, y), np.append(z_bottom, z_top)])
    
    # Generate the faces of the cylinder
    # faces = [[i, i + 1, i + segments + 1, i + segments] for i in range(segments - 1)]
    # faces.append([segments - 1, 0, segments, 2 * segments - 1])  # last face connects back to the first
    faces = []
    # intensity2 = []
    for i in range(segments - 1):
        faces.append([i, i + 1, i + segments])
        faces.append([i + segments, i + 1, i + segments + 1])
    
    # faces.append([i + segments, i + 1, i + segments + 1])
    # faces.append([segments - 1, 0, 2 * segments - 1])
    # intensity2.append((theta[segments - 1] + theta[0] + theta[2 * segments - 1]) / 3 / (2 * np.pi))
    # faces.append([2 * segments - 1, 0, segments])
    # colorscale = [[0, '#cccccc'], [0.5, '#666666'], [1, '#cccccc']]
    colorscale = [[0, '#c3dbd7'], [0.5, '#666666'], [1, '#c3dbd7']]
    # colorscale = [[0, 'blue'], [0.5, 'red'], [1, 'blue']]
    intensity = np.append(theta / (2 * np.pi), theta / (2 * np.pi))
    print(x)
    print(intensity)
    # Create a Mesh3d trace for the cylinder
    trace_cylinder = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=[face[0] for face in faces],
        j=[face[1] for face in faces],
        k=[face[2] for face in faces],
        intensity=intensity,  # use theta for color
        # intensity=theta / (2 * np.pi),  # use theta for color
        colorscale=colorscale,
        opacity=.5,  # semi-transparent
        name='cylinder'
    )
    if fig is None:
        fig = go.Figure()
    
    fig.add_trace(trace_cylinder)
    pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01, aspects=aspects)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    
    return fig


def plot_torus(r_center=3, r_tube=1., center=(0, 0, 0), segments=100, rings=100, fig=None,
               save=None, show=False, dots_bound=None, aspects=(2, 2, 2)):
    theta = np.linspace(0, 2 * np.pi, segments)  # coordinates around the tube
    phi = np.linspace(0, 2 * np.pi, rings)  # coordinates around the torus
    theta, phi = np.meshgrid(theta, phi)
    x = (r_center + r_tube * np.cos(theta)) * np.cos(phi)
    y = (r_center + r_tube * np.cos(theta)) * np.sin(phi)
    z = r_tube * np.sin(theta)
    
    # The vertices are the points on the bottom and top circles of the cylinder
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Generate the faces of the cylinder
    # faces = [[i, i + 1, i + segments + 1, i + segments] for i in range(segments - 1)]
    # faces.append([segments - 1, 0, segments, 2 * segments - 1])  # last face connects back to the first
    # Create the faces of the torus
    faces = []
    for i in range(rings - 1):
        for j in range(segments - 1):
            faces.append([i * segments + j, i * segments + j + 1, (i + 1) * segments + j + 1])
        # faces.append([(i + 1) * segments + j + 1, (i + 1) * segments + j, i * segments + j])
    
    # Append faces for the last ring connects with the first ring
    # for j in range(segments - 1):
    # 	faces.append([(rings - 1) * segments + j, (rings - 1) * segments + j + 1, j + 1])
    # 	faces.append([j + 1, j, (rings - 1) * segments + j])
    #
    # faces.append([segments - 1, 2 * segments - 1, segments])
    # colorscale = [[0, '#cccccc'], [0.5, '#666666'], [1, '#cccccc']]
    colorscale = [[0, '#c3dbd7'], [0.5, '#666666'], [1, '#c3dbd7']]
    colorscale = [[0, '#c3dbd7'], [0.5, 'red'], [1, '#c3dbd7']]
    # Create a Mesh3d trace for the cylinder
    trace_torus = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=[face[0] for face in faces],
        j=[face[1] for face in faces],
        k=[face[2] for face in faces],
        intensity=(phi.flatten() / (2 * np.pi)),  # use the face index for color
        colorscale=colorscale,
        opacity=1,  # semi-transparent
        name='torus'
    )
    # intensity=np.append(theta / (2 * np.pi), theta / (2 * np.pi)),
    
    if fig is None:
        fig = go.Figure()
    
    fig.add_trace(trace_torus)
    # pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01, aspects=aspects)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    
    return fig


def fit_3D(x, y, z, degree=2):
    X = np.column_stack([x, y])
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, z)
    return model.predict


def curve_3D(x, y, z, resolution=50):
    points = np.array([x, y, z])
    tck, u = splprep(points, s=0)
    u_new = np.linspace(u.min(), u.max(), resolution)
    new_points = splev(u_new, tck)
    return new_points


def curve_3D_smooth(x, y, z, resolution=50, s=0.02, k=2, b_imp=10):
    x = np.concatenate(([x[0]] * b_imp, x, [x[-1]] * b_imp))
    y = np.concatenate(([y[0]] * b_imp, y, [y[-1]] * b_imp))
    z = np.concatenate(([z[0]] * b_imp, z, [z[-1]] * b_imp))
    points = np.array([x, y, z])
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=1) ** 2, axis=0)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    spl_x = UnivariateSpline(distance, points[0, :], k=k, s=s)
    spl_y = UnivariateSpline(distance, points[1, :], k=k, s=s)
    spl_z = UnivariateSpline(distance, points[2, :], k=k, s=s)
    distance_new = np.linspace(0, 1, resolution)
    x_new = spl_x(distance_new)
    y_new = spl_y(distance_new)
    z_new = spl_z(distance_new)
    return x_new, y_new, z_new


def braids_xy(z, angle=0):
    x = np.cos(z + angle)
    y = np.sin(z + angle)
    return x, y, z


class TravelingSalesmanProblem(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super(TravelingSalesmanProblem, self).__init__(state)  # important!

    def move(self):
        # swap two cities in the route
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        e = 0
        for i in range(len(self.state)):
            e += self.distance_matrix[self.state[i-1]][self.state[i]]
        return e


def salesman(dots):
    dist_matrix = dist.cdist(dots, dots)

    # 1
    # permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    # 2
    # permutation, distance = solve_tsp_simulated_annealing(dist_matrix)
    # 3
    # init_state = list(range(len(dots)))
    #
    # tsp = TravelingSalesmanProblem(init_state, dist_matrix)
    # tsp.set_schedule(tsp.auto(minutes=1))  # set approximate time to find solution
    #
    # # Find an approximate solution
    # permutation, distance = tsp.anneal()
    # 4
    index_combinations = list(combinations(range(len(dots)), 2))

    # Calculate pairwise distances without loops
    dist_matrix_ml = [(i, j, dist.euclidean(dots[i], dots[j])) for i, j in index_combinations]

    #
    fitness_dists = mlrose.TravellingSales(distances=dist_matrix_ml)
    problem_fit = mlrose.TSPOpt(length=len(dots), fitness_fn=fitness_dists, maximize=False)

    permutation, _, _ = mlrose.genetic_alg(problem_fit, random_state=2)
    # print(mlrose.genetic_alg(problem_fit, random_state=2))
    # exit()
    print(permutation)
    return dots[permutation]
def find_path(dots):
    nbrs = NearestNeighbors(n_neighbors=len(dots), algorithm='ball_tree').fit(dots)
    print(dots)
    distances, indices = nbrs.kneighbors(dots)

    visited = set()
    current_index = 0
    path = [current_index]
    visited.add(current_index)

    while len(path) < len(dots):
        for idx in indices[current_index]:
            if idx not in visited:
                current_index = idx
                break
        path.append(current_index)
        visited.add(current_index)

    return path


def find_path2(coords):
    current_index = 0
    path = [current_index]
    while len(path) < len(coords):
        remaining_indices = np.delete(np.arange(len(coords)), path)
        distances = np.linalg.norm(coords[remaining_indices] - coords[current_index], axis=1)
        current_index = remaining_indices[np.argmin(distances)]
        path.append(current_index)
    return path





# XYZ = curve_3D(x, y, z, resolution=40)
# x, y, z = curve_3D_smooth(x, y, z, resolution=100, k=3, s=10, b_imp=25)
boundary_mult = 1
z = np.linspace(-np.pi * boundary_mult, np.pi * boundary_mult, 100)
x, y, z = braids_xy(z)
x2, y2, z2 = braids_xy(z, angle=np.pi)

boundary_3D = [[x.min() - 1, y.min() - 1, z.min()],
               [x.max() + 1, y.max() + 1, z.max()]]
dots = np.stack([x, y, z], axis=1)
dots2 = np.stack([x2, y2, z2], axis=1)
# boundary_3D = [[0, 0, 0], [100, 100, 100]]
# dp.plotDots(hopf_braid1, boundary_3D, color='red', show=True, size=7)
# plt.show()


# normal mine plotly
if 0:
    plotDots(dots, boundary_3D, color='black', show=True, size=7)

# braids
if 1:
    fig = plot_cylinder(
        radius=0.93, height=2 * np.pi, fig=None, show=False, dots_bound=boundary_3D
    )
    # fig = plot_torus(
    # 	r_center=0.7, r_tube=0.4, fig=None, show=False, segments=100, rings=100, dots_bound=boundary_3D
    # )
    color = ([0, '#660000'], [1, '#ff0000'])
    plot_line_colored(dots, dots_bound=boundary_3D, show=False, color=color, width=25, fig=fig, save=None)
    color2 = ([0, '#007dff'], [1, '#000099'])
    # color = ([0, '#660000'], [1, '#ff0000'])
    plot_line_colored(dots2, dots_bound=boundary_3D, show=False, color=color2, width=25, fig=fig, save=None)
    pl.box_set_go(fig, mesh=None, autoDots=boundary_3D, perBox=0.01, aspects=[1.5, 1.5, 1.5 * 1.5])
    fig.show()
    exit()
