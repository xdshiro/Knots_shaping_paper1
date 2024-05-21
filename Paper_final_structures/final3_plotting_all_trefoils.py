from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io
from sklearn.neighbors import NearestNeighbors


def angle_vectors(a, b):
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cos_angle, -1, 1))  # Clip ensures that potential rounding errors don't cause issues

    # Convert the angle from radians to degrees if needed
    angle_degrees = np.degrees(angle)
    print(angle_degrees)
    return angle_degrees


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
    diffs = coords[path][1:] - coords[path][:-1]
    distances = np.linalg.norm(diffs, axis=1)
    return path, distances


# np.save(f'trefoil_rot{int(rotation_on)}_shift{int(shifting_z_on)}'
#         f'_resX{res_x_3D_k}_resZ{res_z_3D_k}_limX{x_lim_3D_k[1]}_limZ{z_lim_3D_k[1]}', dots_init)


dotSize = 12
# red is rotated, blue, green, counter
angles = 0
if angles:
    # normal
    a1, a2 = np.array([30, 111, 0]), np.array([(54 + 72) / 2, (49 + 60) / 2, 0])
    b1, b2 = np.array([30, 9, 0]), np.array([(54 + 72) / 2, (72 + 60) / 2, 0])
    a = a2 - a1
    b = b2 - b1
    angle_vectors(a, b)
    # rotated_shifted
    a1, a2 = np.array([111, 90, 0]), np.array([(49 + 62) / 2, (74 + 50) / 2, 0])  # green
    b1, b2 = np.array([31, 111, 0]), np.array([(62 + 72) / 2, (50 + 57) / 2, 0])  # red
    c1, c2 = np.array([30, 8, 0]), np.array([(49 + 72) / 2, (74 + 57) / 2, 0])  # blue

    # a1, a2 = np.array([305, 331, 0]), np.array([(116 + 93) / 2, (154 + 242) / 2, 0])  # green
    # b1, b2 = np.array([37, 401, 0]), np.array([(116 + 170) / 2, (154 + 215) / 2, 0])  # red
    # c1, c2 = np.array([0, 0, 0]), np.array([(170 + 93) / 2, (215 + 242) / 2, 0])  # blue
    # 82.52873788298527
    # 153.47807094569424
    # 123.99319117132049
    95.27899531225907
    144.71418463784505
    120.0068200498959
    a = a2 - a1
    b = b2 - b1
    c = c2 - c1
    angle_vectors(a, b)
    angle_vectors(a, c)
    angle_vectors(c, b)
trefoil_normal = 0
if trefoil_normal:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dots_init = np.load(f'trefoil_rot{int(False)}_shift{int(False)}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = N_dots // 5 + 18
    N_segment1 = N_dots // 3 + 19
    N_segment2 = N_dots // 3 + 25
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)

    dots_z0 = np.array([[118, 60, 60], [72, 60, 60], [54, 71, 60], [54, 49, 60], [30, 111, 60], [30, 9, 60]])
    dots_z0 = np.array(dots_z0)
    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.show()

trefoil_normal_intensity = 0
if trefoil_normal_intensity:
    field_abs = np.load(f'trefoil_field_80.npy')[:, ::-1, :]
    res_xy_f, res_z_f = 80, 80
    scale = 1
    limXY_f, limZ_f = 2.2 * scale, 0.75

    # dot_center = [0, 0, 0]
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-limXY, limXY), (-limXY, limXY), (-limZ, limZ)
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-res_xy, res_xy), (-res_xy, res_xy), (-res_z, res_z)
    # x_3D = np.linspace(*x_lim_3D_k, res_xy)
    # y_3D = np.linspace(*y_lim_3D_k, res_xy)
    # z_3D = np.linspace(*z_lim_3D_k, res_z)
    # boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_xy]]
    mesh_3D_res = np.meshgrid(np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_z_f, dtype='float64'), indexing='ij')
    # plt.imshow(field_abs[:, :, res_z // 2 - 25])
    # plt.show()
    # exit()

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dot_center = [int(80 * 1.5 * scale / 4), int(80 * 1.5 * scale / 4), 0]
    dot_center = [int(res_xy / 2 * (scale - 1)), int(res_xy / 2 * (scale - 1)), 0]
    trans = 2
    fig = pl.plot_3D_density(field_abs / np.abs(field_abs).max(), mesh=mesh_3D_res, show=False,
                             scaling=[res_xy / res_xy_f * scale,
                                      res_xy / res_xy_f * scale,
                                      res_z / res_z_f],
                             surface_count=10,
                             opacity=1, colorscale='Jet',
                             # opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45], [1, 0.70]]
                             opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45 / trans], [1, 0.70 / trans]]
                             )
    dots_init = np.load(f'trefoil_rot{int(False)}_shift{int(False)}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = N_dots // 5 + 18
    N_segment1 = N_dots // 3 + 19
    N_segment2 = N_dots // 3 + 25
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind])  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    dot_border = [3, 3, 3]
    # dot_border = [0, 0, 0]
    boundary_3D_k = [[0, 0, 0] + dot_border, np.array([res_xy * scale, res_xy * scale, res_z]) - np.array(dot_border)]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[1.0, 1.0, 1])
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[2.0, 2.0, 1])
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[1.0, 1.0, 1])
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=0, z=-1)
            )
        )
    )
    # fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()



# rotated
trefoil_rotated = 0
if trefoil_rotated:

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dots_init = np.load(f'trefoil_rot{1}_shift{0}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')[:]

    N_dots = np.shape(dots_init)[0]
    N_roll = 118
    N_segment1 = N_dots // 3 + 20 # green
    N_segment2 = N_dots // 3 - 53  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index-2]
    distances_clean = distances_unrolled[:largest_index-2]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind])  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dots_z0 = np.array([[112, 90, 60], [72, 57, 60], [62, 50, 60], [30, 8, 60], [49, 74, 60], [31, 111, 60]])

    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
    #             up=dict(x=0, y=0, z=-1)
    #         )
    #     )
    # )

    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
    #             up=dict(x=0, y=1, z=0)
    #         )
    #     )
    # )
    # fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.show()

trefoil_rotated_intensity = 0
if trefoil_rotated_intensity:
    field_abs = np.load(f'trefoil_rotated_field_80.npy')[:, ::-1, :]
    res_xy_f, res_z_f = 80, 80
    scale = 1
    limXY_f, limZ_f = 2.2 * scale, 0.75

    # dot_center = [0, 0, 0]
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-limXY, limXY), (-limXY, limXY), (-limZ, limZ)
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-res_xy, res_xy), (-res_xy, res_xy), (-res_z, res_z)
    # x_3D = np.linspace(*x_lim_3D_k, res_xy)
    # y_3D = np.linspace(*y_lim_3D_k, res_xy)
    # z_3D = np.linspace(*z_lim_3D_k, res_z)
    # boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_xy]]
    mesh_3D_res = np.meshgrid(np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_z_f, dtype='float64'), indexing='ij')
    # plt.imshow(field_abs[:, :, res_z // 2 - 25])
    # plt.show()
    # exit()

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dot_center = [int(80 * 1.5 * scale / 4), int(80 * 1.5 * scale / 4), 0]
    dot_center = [int(res_xy / 2 * (scale - 1)), int(res_xy / 2 * (scale - 1)), 0]
    trans = 2
    fig = pl.plot_3D_density(field_abs / np.abs(field_abs).max(), mesh=mesh_3D_res, show=False,
                             scaling=[res_xy / res_xy_f * scale,
                                      res_xy / res_xy_f * scale,
                                      res_z / res_z_f],
                             surface_count=10,
                             opacity=1, colorscale='Jet',
                             # opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.10], [0.45, 0.35], [1, 0.70]]
                             opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45 / trans], [1, 0.70 / trans]]
                             )
    dots_init = np.load(f'trefoil_rot{1}_shift{0}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')[:]

    N_dots = np.shape(dots_init)[0]
    N_roll = 118
    N_segment1 = N_dots // 3 + 20  # green
    N_segment2 = N_dots // 3 - 53  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index - 2]
    distances_clean = distances_unrolled[:largest_index - 2]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind])  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    dot_border = [3, 3, 3]
    # dot_border = [0, 0, 0]
    boundary_3D_k = [[0, 0, 0] + dot_border, np.array([res_xy * scale, res_xy * scale, res_z]) - np.array(dot_border)]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[1.0, 1.0, 1])
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[2.0, 2.0, 1])
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig,
                     lines=True, aspects=[1.0, 1.0, 1])
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=0, z=-1)
            )
        )
    )
    # fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()


# rotated shifted
trefoil_rotated_shifted = 0
if trefoil_rotated_shifted:

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dots_init = np.load(f'trefoil_rot{1}_shift{1}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = 85
    N_segment1 = N_dots // 3 - 42  # red
    N_segment2 = N_dots // 3 + 10  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind])  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='green', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dots_z0 = np.array([[51, 74, 60], [107, 94, 60], [38, 113, 60], [71, 63, 60], [59, 50, 60], [28, 9, 60]])
    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
    #             up=dict(x=0, y=0, z=-1)
    #         )
    #     )
    # )

    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
    #             up=dict(x=0, y=1, z=0)
    #         )
    #     )
    # )
    # fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.show()

trefoil_rotated_shifted_intensity = 0
if trefoil_rotated_shifted_intensity:
    field_abs = np.load(f'trefoil_rotated_field_80.npy')[:, ::-1, :]
    res_xy, res_z = 80, 80
    limXY, limZ = 2.2 * 2.2, 0.75
    scale = 1
    dot_center = [int(80 * 1.5 * scale / 4), int(80 * 1.5 * scale / 4), 0]
    dot_center = [0, 0, 0]

    # dot_center = [0, 0, 0]
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-limXY, limXY), (-limXY, limXY), (-limZ, limZ)
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-res_xy, res_xy), (-res_xy, res_xy), (-res_z, res_z)
    # x_3D = np.linspace(*x_lim_3D_k, res_xy)
    # y_3D = np.linspace(*y_lim_3D_k, res_xy)
    # z_3D = np.linspace(*z_lim_3D_k, res_z)
    # boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_xy]]
    mesh_3D_res = np.meshgrid(np.arange(res_xy, dtype='float64'),
                              np.arange(res_xy, dtype='float64'),
                              np.arange(res_z, dtype='float64'), indexing='ij')
    # plt.imshow(field_abs[:, :, res_z // 2 - 25])
    # plt.show()
    # exit()
    trans = 2
    fig = pl.plot_3D_density(field_abs / np.abs(field_abs).max(), mesh=mesh_3D_res, show=False,
                             scaling=[1.5 * scale, 1.5 * scale, 1.5],
                             surface_count=10,
                             opacity=1, colorscale='Jet',
                             # opacityscale=[[0, 0.15], [0.15, 0.25], [0.25, 0.35], [1, 0.70]]
                             opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45 / trans], [1, 0.70 / trans]]
                             )

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dots_init = np.load(f'trefoil_rot{1}_shift{1}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = 85
    N_segment1 = N_dots // 3 - 42  # red
    N_segment2 = N_dots // 3 + 10  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind]) + np.array(dot_center)  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    dot_border = [3, 3, 3]
    boundary_3D_k = [[0, 0, 0] + dot_border, np.array([res_xy * scale, res_xy * scale, res_z]) - np.array(dot_border)]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots1, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=0, z=-1)
            )
        )
    )
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()

trefoil_rotated_shifted_intensity_beam = 0
if trefoil_rotated_shifted_intensity_beam:
    field_abs = np.load(f'trefoil_rotated_shifted_field_beam_60.npy')[:, ::-1, :]
    res_xy_f, res_z_f = 60, 60
    scale = 2.5
    limXY_f, limZ_f = 2.2 * scale, 0.75

    # dot_center = [0, 0, 0]
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-limXY, limXY), (-limXY, limXY), (-limZ, limZ)
    # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-res_xy, res_xy), (-res_xy, res_xy), (-res_z, res_z)
    # x_3D = np.linspace(*x_lim_3D_k, res_xy)
    # y_3D = np.linspace(*y_lim_3D_k, res_xy)
    # z_3D = np.linspace(*z_lim_3D_k, res_z)
    # boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_xy]]
    mesh_3D_res = np.meshgrid(np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_xy_f, dtype='float64'),
                              np.arange(res_z_f, dtype='float64'), indexing='ij')
    # plt.imshow(field_abs[:, :, res_z // 2 - 25])
    # plt.show()
    # exit()

    # Fig.show()
    # exit()
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dot_center = [int(80 * 1.5 * scale / 4), int(80 * 1.5 * scale / 4), 0]
    dot_center = [int(res_xy / 2 * (scale - 1)), int(res_xy / 2 * (scale - 1)), 0]
    trans = 2.5
    fig = pl.plot_3D_density(field_abs / np.abs(field_abs).max(), mesh=mesh_3D_res, show=False,
                             scaling=[res_xy / res_xy_f * scale,
                                      res_xy / res_xy_f * scale,
                                      res_z / res_z_f],
                             surface_count=40,
                             opacity=1, colorscale='Jet',
                             # opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.10], [0.45, 0.35], [1, 0.70]]
                             opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45 / trans], [1, 0.70 / trans]]
                             )
    dots_init = np.load(f'trefoil_rot{1}_shift{1}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = 85
    N_segment1 = N_dots // 3 - 42  # red
    N_segment2 = N_dots // 3 + 10  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = np.array(dots_init[path_ind]) + np.array(dot_center)  # [:]
    # print(dots_init)
    # exit()
    dots1 = dots_init[0:N_segment1]

    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    dot_border = [3, 3, 3]
    dot_border = [0, 0, 0]
    boundary_3D_k = [[0, 0, 0] + dot_border, np.array([res_xy * scale, res_xy * scale, res_z]) - np.array(dot_border)]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots1, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig,
                     lines=False, aspects=[1, 1, 1])
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig,
                     lines=False, aspects=[1, 1, 1])
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig,
                     lines=False, aspects=[0.8, 0.8, 1])
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=0, z=-1)
            )
        )
    )
    # fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()
trefoil_rotated = 0
if trefoil_rotated:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2, 0.75
    dots_init = np.load(f'trefoil_rot{1}_shift{0}'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = 117
    N_segment1 = N_dots // 3 + 19  # red
    N_segment2 = N_dots // 3 - 52  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()
trefoil_better = 0
if trefoil_better:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.4, 0.75
    dots_init = np.load(f'trefoil_fixed'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = 226
    N_segment1 = N_dots // 3 + 21  # blue
    N_segment2 = N_dots // 3 + 4  # red
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:-2])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='royalblue', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()
trefoil_better_shift = 0
if trefoil_better_shift:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.4, 0.75
    dots_init = np.load(f'trefoil_fixed_scaled14'
                        f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = -150
    N_segment1 = N_dots // 3 + 16  # red
    N_segment2 = N_dots // 3 - 48  # green
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:-9])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()

trefoil_mine_best = 0
if trefoil_mine_best:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2 * 1.6, 1.5
    dots_init = np.load(f'trefoil_mine_best_w=1d4_x=3d5200000000000005_resXY=120_resZ=120.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = -121
    N_segment1 = N_dots // 3 + 6  # red
    N_segment2 = N_dots // 3 - 32  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:-16])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dots_z0_clean = np.array(dots_z0)
    # dots_z0_clean = np.array([
    #     [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    # ])
    # for dot in dots_z0:
    #     if dot in np.array([
    #             [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    #             ]):
    #         dots_z0_clean.append(dot)

    dp.plotDots_Hopf(dots_z0_clean, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.show()

trefoil_mine_best_6 = 1
if trefoil_mine_best_6:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2 * 1.6, 1.5
    dots_init = np.load(f'trefoil_mine_best_6_w=1d4_x=3d5200000000000005_resXY=120_resZ=120.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = -121
    N_segment1 = N_dots // 3 + 6  # red
    N_segment2 = N_dots // 3 - 32  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:-5])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dots_z0_clean = np.array(dots_z0)
    # dots_z0_clean = np.array([
    #     [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    # ])
    # for dot in dots_z0:
    #     if dot in np.array([
    #             [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    #             ]):
    #         dots_z0_clean.append(dot)

    dp.plotDots_Hopf(dots_z0_clean, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.show()

trefoil_Dennis = 0
if trefoil_Dennis:
    res_xy, res_z = 120, 120
    limXY, limZ = 2.2 * 1.6, 1.5
    dots_init = np.load(f'trefoil_Dennis_w=1d6_x=3d5200000000000005_resXY=120_resZ=120.npy')

    N_dots = np.shape(dots_init)[0]
    N_roll = -120
    N_segment1 = N_dots // 3 + 1  # red
    N_segment2 = N_dots // 3 - 35  # blue
    # N_segment3 = N_dots - N_segment1 - N_segment2
    path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
    sorted_indices = np.argsort(distances_unrolled[:-16])
    largest_index = sorted_indices[-1]
    path_ind_clean = path_ind_unrolled[:largest_index]
    distances_clean = distances_unrolled[:largest_index]
    path_ind = np.roll(path_ind_clean, shift=N_roll)
    distances = np.roll(distances_clean, shift=N_roll)
    # avg_distance = np.average(distances)

    N = np.shape(path_ind)[0]
    dots_init = dots_init[path_ind]  # [:]
    dots1 = dots_init[0:N_segment1]
    dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
    dots3 = dots_init[N_segment1 + N_segment2:]
    boundary_3D_k = [[0, 0, 0], [res_xy, res_xy, res_z]]
    # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    dp.plotDots_Hopf(dots3, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig)
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z // 2:
            dots_z0.append(dot)
    dots_z0 = np.array(dots_z0)
    dots_z0_clean = np.array(dots_z0)
    # dots_z0_clean = np.array([
    #     [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    # ])
    # for dot in dots_z0:
    #     if dot in np.array([
    #             [35, 114, 60], [35, 6, 60], [58, 36, 60], [58, 83, 60], [72, 62, 60], [103, 62, 60]
    #             ]):
    #         dots_z0_clean.append(dot)

    dp.plotDots_Hopf(dots_z0_clean, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()
exit()
# trefoil_rotated_shifted_intensity_beam = 00
# if trefoil_rotated_shifted_intensity_beam:
#     field_abs = np.load(f'trefoil_rotated_shifted_field_60.npy')[:, ::-1, :]
#     res_xy_f, res_z_f = 60, 60
#     scale = 1
#     limXY_f, limZ_f = 2.2 * scale, 0.75
#
#     # dot_center = [0, 0, 0]
#     # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-limXY, limXY), (-limXY, limXY), (-limZ, limZ)
#     # x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-res_xy, res_xy), (-res_xy, res_xy), (-res_z, res_z)
#     # x_3D = np.linspace(*x_lim_3D_k, res_xy)
#     # y_3D = np.linspace(*y_lim_3D_k, res_xy)
#     # z_3D = np.linspace(*z_lim_3D_k, res_z)
#     # boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_xy]]
#     mesh_3D_res = np.meshgrid(np.arange(res_xy_f, dtype='float64'),
#                               np.arange(res_xy_f, dtype='float64'),
#                               np.arange(res_z_f, dtype='float64'), indexing='ij')
#     # plt.imshow(field_abs[:, :, res_z // 2 - 25])
#     # plt.show()
#     # exit()
#
#     # Fig.show()
#     # exit()
#     res_xy, res_z = 120, 120
#     limXY, limZ = 2.2, 0.75
#     dot_center = [int(80 * 1.5 * scale / 4), int(80 * 1.5 * scale / 4), 0]
#     dot_center = [int(res_xy / 2 * (scale - 1)), int(res_xy / 2 * (scale - 1)), 0]
#     fig = pl.plot_3D_density(field_abs / np.abs(field_abs).max(), mesh=mesh_3D_res, show=False,
#                              scaling=[res_xy / res_xy_f * scale,
#                                       res_xy / res_xy_f * scale,
#                                       res_z / res_z_f],
#                              surface_count=10,
#                              opacity=1, colorscale='Jet',
#                              # opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.10], [0.45, 0.35], [1, 0.70]]
#                              opacityscale=[[0, 0.0], [0.2, 0.0], [0.25, 0.45], [1, 0.70]]
#                              )
#     dots_init = np.load(f'trefoil_rot{1}_shift{1}'
#                         f'_resX{res_xy}_resZ{res_z}_limX{limXY}_limZ{limZ}.npy')
#
#     N_dots = np.shape(dots_init)[0]
#     N_roll = 85
#     N_segment1 = N_dots // 3 - 42  # red
#     N_segment2 = N_dots // 3 + 10  # blue
#     # N_segment3 = N_dots - N_segment1 - N_segment2
#     path_ind_unrolled, distances_unrolled = find_path(dots_init, 0)
#     sorted_indices = np.argsort(distances_unrolled[:])
#     largest_index = sorted_indices[-1]
#     path_ind_clean = path_ind_unrolled[:largest_index]
#     distances_clean = distances_unrolled[:largest_index]
#     path_ind = np.roll(path_ind_clean, shift=N_roll)
#     distances = np.roll(distances_clean, shift=N_roll)
#     # avg_distance = np.average(distances)
#
#     N = np.shape(path_ind)[0]
#     dots_init = np.array(dots_init[path_ind]) + np.array(dot_center)  # [:]
#     # print(dots_init)
#     # exit()
#     dots1 = dots_init[0:N_segment1]
#
#     dots2 = dots_init[N_segment1:N_segment1 + N_segment2]
#     dots3 = dots_init[N_segment1 + N_segment2:]
#     dot_border = [3, 3, 3]
#     # dot_border = [0, 0, 0]
#     boundary_3D_k = [[0, 0, 0] + dot_border, np.array([res_xy * scale, res_xy * scale, res_z]) - np.array(dot_border)]
#     # fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
#     dp.plotDots_Hopf(dots1, boundary_3D_k, color='green', show=False, size=dotSize, fig=fig,
#                      lines=True, aspects=[1.0, 1.0, 1])
#     dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig,
#                      lines=True, aspects=[1.0, 1.0, 1])
#     dp.plotDots_Hopf(dots3, boundary_3D_k, color='red', show=False, size=dotSize, fig=fig,
#                      lines=True, aspects=[1.0, 1.0, 1])
#     dots_z0 = []
#     for dot in dots_init:
#         if dot[2] == res_z // 2:
#             dots_z0.append(dot)
#     dots_z0 = np.array(dots_z0)
#     fig.update_layout(
#         scene=dict(
#             camera=dict(
#                 eye=dict(x=4, y=4, z=4),  # Adjust x, y, and z to set the default angle of view
#                 up=dict(x=0, y=0, z=-1)
#             )
#         )
#     )
#     fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
#     # dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
#     fig.show()
# dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
# fig.write_html(f'trefoil_rotated_shifted.html')
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=-2.3, y=-2.3, z=2.3)  # Adjust x, y, and z to set the default angle of view
        )
    )
)
fig.write_html(f'trefoil_{name_f}_t.html')
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=0, y=-4, z=0)  # Adjust x, y, and z to set the default angle of view
        )
    )
)
fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
fig.write_html(f'trefoil_{name_f}_xz.html')
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=-4, y=0, z=0),  # Adjust x, y, and z to set the default angle of view
            up=dict(x=0, y=0, z=1)
        )
    )
)
fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))

fig.write_html(f'trefoil_{name_f}_yz.html')
# fig.update_layout(
#     scene=dict(
#         camera=dict(
#             eye=dict(x=0, y=0, z=3),  # Adjust x, y, and z to set the default angle of view
#             up=dict(x=0, y=1, z=0)
#         )
#     )
# )
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
            up=dict(x=0, y=1, z=0)
        )
    )
)
fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
fig.write_html(f'trefoil_{name_f}_xy.html')
# fig.write_html('hopf_rotated_shifted.html')
# plt.show()
# plt.show()
###################################################################
