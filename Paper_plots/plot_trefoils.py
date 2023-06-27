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
            faces.append([(i + 1) * segments + j + 1, (i + 1) * segments + j, i * segments + j])
    
    # Append faces for the last ring connects with the first ring
    # for j in range(segments - 1):
    # 	faces.append([(rings - 1) * segments + j, (rings - 1) * segments + j + 1, j + 1])
    # 	faces.append([j + 1, j, (rings - 1) * segments + j])
    #
    # faces.append([segments - 1, 2 * segments - 1, segments])
    # colorscale = [[0, '#cccccc'], [0.5, '#666666'], [1, '#cccccc']]
    colorscale = [[0, '#c3dbd7'], [0.5, '#666666'], [1, '#c3dbd7']]
    # colorscale = [[0, '#c3dbd7'], [0.5, 'red'], [1, '#c3dbd7']]
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
        opacity=0.4,  # semi-transparent
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


def curve_3D(x, y, z, resolution=50, smoothing_factor=0, k=3):
    points = np.array([x, y, z])
    tck, u = splprep(points, s=smoothing_factor, k=k)
    u_new = np.linspace(u.min(), u.max(), resolution)
    new_points = splev(u_new, tck)
    return np.column_stack(new_points)


def curve_3D_smooth(x, y, z, resolution=50, s=0.02, k=2, b_imp=10):
    x = np.concatenate(([x[0]] * b_imp, x, [x[-1]] * b_imp))
    y = np.concatenate(([y[0]] * b_imp, y, [y[-1]] * b_imp))
    z = np.concatenate(([z[0]] * b_imp, z, [z[-1]] * b_imp))
    # print(x, y, z)
    # exit()
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
    
    return np.column_stack((x_new, y_new, z_new))


def braids_xy(z, angle=0):
    x = np.cos(z + angle)
    y = np.sin(z + angle)
    return x, y, z


def find_path(coords, start_index=0):
    nbrs = NearestNeighbors(n_neighbors=len(coords)).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    visited = set()
    current_index = start_index
    path = [current_index]
    visited.add(current_index)

    while len(path) < len(coords):
        for idx in indices[current_index]:
            if idx not in visited:
                current_index = idx
                break
        path.append(current_index)
        visited.add(current_index)

    return path


# trefoil = np.load('trefoil_math_w=0d95_x=2d5_resXY=211_resZ=211.npy')
trefoil = np.load('trefoil_milnor_w=1d3_x=3d0_z=1d0_resXY=251_resZ=251.npy')
# sorted_indices = np.argsort(hopf_braid1[:, -1])[::-1]
# hopf_braid1_sorted = hopf_braid1[sorted_indices]
res = 251
x = trefoil[:, 0] - res // 2
y = trefoil[:, 1] - res // 2
z = (trefoil[:, 2] - res // 2) / 3
# x = (trefoil[:, 0] - res // 2) / res * 5
# y = (trefoil[:, 1] - res // 2) / res * 5
# z = (trefoil[:, 2] - res // 2) / res * 10
# boundary_3D = [[x.min() - 1, y.min() - 1, z.min()],
#                [x.max() + 1, y.max() + 1, z.max()]]
boundary_3D = [[- res // 2, - res // 2, - res // 2],
               [res // 2, res // 2, res // 2]]
dots = np.stack([x, y, z], axis=1)
# Find the path
# print(dots)
length = np.shape(dots)[0]
dots = dots[find_path(dots * [1, 1, 1], start_index=length // 2)]
# dots = np.concatenate((dots[:-2], [dots[0]]), axis=0)
dots1 = dots[:length//3]
# print(dots1)
# dots1 = curve_3D_smooth(dots1[:, 0], dots1[:, 1], dots1[:, 2], resolution=100, k=3, s=10, b_imp=25)
# dots1 = curve_3D_smooth(dots1[:, 0], dots1[:, 1], dots1[:, 2])
dots1 = curve_3D(dots1[:, 0], dots1[:, 1], dots1[:, 2], resolution=200, smoothing_factor=50)
dots2 = dots[length//3:2 * length//3]
dots2 = curve_3D(dots2[:, 0], dots2[:, 1], dots2[:, 2], resolution=200, smoothing_factor=50)
dots3 = dots[2 * length//3:-3]
dots3 = curve_3D(dots3[:, 0], dots3[:, 1], dots3[:, 2], resolution=200, smoothing_factor=75)
# print(dots1)

# dots = salesman(dots) * np.array([5, 5, 1])

# trefoil and torus
if 0:
    # fig = plot_cylinder(
    # 	radius=0.93, height=2 * np.pi, fig=None, show=False, dots_bound=boundary_3D
    # )
    fig = plot_torus(
        r_center=int(res//4 * 0.91), r_tube=(res//10) * 1.35, fig=None, show=False, segments=100, rings=100, dots_bound=boundary_3D
    )
    color = ([0, '#660000'], [1, '#ff0000'])
    plot_line_colored(dots1, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    color = ([0, '#007dff'], [1, '#000099'])
    plot_line_colored(dots2, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    color = ([0, '#19ff19'], [1, '#134a0d'])
    plot_line_colored(dots3, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    pl.box_set_go(fig, mesh=None, autoDots=boundary_3D, perBox=0.01, aspects=[1.5, 1.5, 1.5])
    fig.show()
    exit()

rotate_meshgrid(*mesh_3D, np.radians(180-44), np.radians(00), np.radians(0))

if 1:
    # fig = plot_cylinder(
    # 	radius=0.93, height=2 * np.pi, fig=None, show=False, dots_bound=boundary_3D
    # )
    fig = plot_torus(
        r_center=int(res//4 * 0.91), r_tube=(res//10) * 1.35, fig=None, show=False, segments=100, rings=100, dots_bound=boundary_3D
    )
    color = ([0, '#660000'], [1, '#ff0000'])
    plot_line_colored(dots1, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    color = ([0, '#007dff'], [1, '#000099'])
    plot_line_colored(dots2, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    color = ([0, '#19ff19'], [1, '#134a0d'])
    plot_line_colored(dots3, dots_bound=boundary_3D, show=False, color=color, width=25,
                      fig=fig, save=None)
    pl.box_set_go(fig, mesh=None, autoDots=boundary_3D, perBox=0.01, aspects=[1.5, 1.5, 1.5])
    fig.show()
    exit()