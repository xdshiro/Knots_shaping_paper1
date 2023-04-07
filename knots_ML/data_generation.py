"""
This script reads the field from mat file and produces all the necessary pre-processing procedures, and
creates a 3D array of singularity dots.

First main function is main_field_processing:
    1) reading the field from matlab file
    2) converting it into numpy array
    3) normalizing
    4) finding the beam waste
    5) rescaling the field, using the interpolation, for faster next steps
    6) finding the beam center
    7) rescaling field to the scale we want for 3D calculations
    8) removing the tilt and shift

Second main function is !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

import os
import math
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import my_functions.singularities as sing
import my_functions.functions_general as fg
import knots_ML.dots_processing as dp
import knots_ML.center_beam_search as cbs
import my_functions.beams_and_pulses as bp
from os import listdir
from os.path import isfile, join


def read_field_2D_single(path, field=None):
    """
    Function reads .mat 2D array from matlab and convert it into numpy array

    If field is None, it will try to find the field name automatically

    :param path: full path to the file
    :param field: the name of the column with the field you want to read
    """
    field_read = sio.loadmat(path, appendmat=False)
    if field is None:
        for field_check in field_read:
            if len(np.shape(np.array(field_read[field_check]))) == 2:
                field = field_check
                break
    return np.array(field_read[field])


def normalization_field(field):
    """
    Normalization of the field for the beam center finding
    """
    field_norm = field / np.sqrt(np.sum(np.abs(field) ** 2))
    return field_norm


def plot_field(field, save=None):
    """
    Function plots intensity and phase of the field in 1 plot.
    Just a small convenient wrapper
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    image1 = ax1.imshow(np.abs(field))
    ax1.set_title('|E|')
    plt.colorbar(image1, ax=ax1, shrink=0.4, pad=0.02, fraction=0.1)
    image2 = ax2.imshow(np.angle(field), cmap='jet')
    ax2.set_title('Phase(E)')
    plt.colorbar(image2, ax=ax2, shrink=0.4, pad=0.02, fraction=0.1)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, format='png')
    plt.show()


def plot_field_3D_multi_planes(field3D, number=6, columns=3):
    """
    Function plots |E| and phase of the field in 1 plot at different z.
    Just a small convenient wrapper

    :param field3D: 3D complex field, any resolution
    :param number: how many slices do we want to see
    :param columns: number of columns to split the plots into
    """
    fig, axis = plt.subplots(math.ceil(number / columns), 3, figsize=(10, 3 * math.ceil(number / columns)))
    reso_z = np.shape(field3D)[2]
    for i, ax_r in enumerate(axis):
        for j, ax in enumerate(ax_r):
            image = ax.imshow(np.abs(field3D[:, :, int((reso_z - 1) / (number - 1) * (len(ax_r) * i + j))]))
            ax.set_title(f'|E|, index z={int((reso_z - 1) / (number - 1) * (len(ax_r) * i + j))}')
            plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02, fraction=0.1)
    plt.tight_layout()
    plt.show()
    fig, axis = plt.subplots(math.ceil(number / columns), 3, figsize=(10, 3 * math.ceil(number / columns)))
    for i, ax_r in enumerate(axis):
        for j, ax in enumerate(ax_r):
            image = ax.imshow(np.angle(field3D[:, :, int((reso_z - 1) / (number - 1) * (len(ax_r) * i + j))]),
                              cmap='jet')
            ax.set_title(f'phase(E), index z={int((reso_z - 1) / (number - 1) * (len(ax_r) * i + j))}')
            plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02, fraction=0.1)

    plt.tight_layout()
    plt.show()


def find_beam_waist(field, mesh=None):
    """
    wrapper for the beam waste finder. More details in knots_ML.center_beam_search
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    width = cbs.find_width(field, mesh=mesh, width=shape[1] // 8, widthStep=1, print_steps=False)
    return width


def field_interpolation(field, mesh=None, resolution=(100, 100),
                        xMinMax_frac=(1, 1), yMinMax_frac=(1, 1), fill_value=True):
    """
    Wrapper for the field interpolation fg.interpolation_complex
    :param resolution: new field resolution
    :param xMinMax_frac: new dimension for the field. (x_dim_old * frac)
    :param yMinMax_frac: new dimension for the field. (y_dim_old * frac)
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    interpol_field = fg.interpolation_complex(field, mesh=mesh, fill_value=fill_value)
    xMinMax = int(-shape[0] // 2 * xMinMax_frac[0]), int(shape[0] // 2 * xMinMax_frac[1])
    yMinMax = int(-shape[1] // 2 * yMinMax_frac[0]), int(shape[1] // 2 * yMinMax_frac[1])
    xyMesh_interpol = fg.create_mesh_XY(
        xRes=resolution[0], yRes=resolution[1],
        xMinMax=xMinMax, yMinMax=yMinMax)
    return interpol_field(*xyMesh_interpol), xyMesh_interpol


def one_plane_propagator(field, dz, stepsNumber_p, stepsNumber_m=None, n0=1, k0=1):
    """
    Double side propagation wrapper for fg.propagator_split_step_3D_linear
    :param field: 2D complex field
    :param dz: step along z
    :param stepsNumber_p: number of steps (forward, p - plus) [there is a chance it's confused with m direction)
    :param stepsNumber_m: number of steps (back, m - minus)
    :param n0: refractive index
    :param k0: wave number
    :return: 3D field
    """
    if stepsNumber_m is None:
        stepsNumber_m = stepsNumber_p
    fieldPropMinus = fg.propagator_split_step_3D_linear(field, dz=-dz, zSteps=stepsNumber_p, n0=n0, k0=k0)
    fieldPropPLus = fg.propagator_split_step_3D_linear(field, dz=dz, zSteps=stepsNumber_m, n0=n0, k0=k0)
    fieldPropTotal = np.concatenate((np.flip(fieldPropMinus, axis=2), fieldPropPLus[:, :, 1:]), axis=2)
    return fieldPropTotal


def main_field_processing(
        path,
        plotting=True,
        resolution_iterpol_center=(70, 70),
        xMinMax_frac_center=(1, 1),
        yMinMax_frac_center=(1, 1),
        resolution_interpol_working=(150, 150),
        xMinMax_frac_working=(1, 1),
        yMinMax_frac_working=(1, 1),
        resolution_crop=(120, 120),
        moments_init=None,
        moments_center=None,
):
    """
    This function:
     1) reading the field from matlab file
     2) converting it into numpy array
     3) normalizing
     4) finding the beam waste
     5) rescaling the field, using the interpolation, for faster next steps
     6) finding the beam center
     7) rescaling field to the scale we want for 3D calculations
     8) removing the tilt and shift

    Assumption
    ----------
    Beam waist finder only works with a uniform grid (dx = dy)

    :param path: file name
    :param plotting: if we want to see the plots and extra information
    :param resolution_iterpol_center: resolution for the beam center finder
    :param xMinMax_frac_center: rescale ration along X axis for the beam center
    :param yMinMax_frac_center: rescale ration along Y axis for the beam center
    :param resolution_interpol_working: resolution for the final field before the cropping
    :param xMinMax_frac_working: rescale ration along X axis for the beam center
    :param yMinMax_frac_working: rescale ration along X axis for the beam center
    :param resolution_crop: actual final resolution of the field
    :param moments_init: the moments for the LG spectrum
    :param moments_center: the moments for the beam center finder
    :return: 2D complex field
    """
    # beam width search work only with x_res==y_res
    if moments_init is None:
        moments_init = {'p': (0, 6), 'l': (-4, 4)}
    if moments_center is None:
        moments_center = {'p0': (0, 4), 'l0': (-4, 2)}

    # reading file
    field_init = read_field_2D_single(path)
    if plotting:
        plot_field(field_init)

    # normalization
    field_norm = normalization_field(field_init)
    if plotting:
        plot_field(field_norm)

    # creating mesh
    mesh_init = fg.create_mesh_XY(xRes=np.shape(field_norm)[0], yRes=np.shape(field_norm)[1])

    # finding beam waste
    width = find_beam_waist(field_norm, mesh=mesh_init)
    if plotting:
        print(f'Approximate beam waist: {width}')

    # rescaling field
    field_interpol, mesh_interpol = field_interpolation(
        field_norm, mesh=mesh_init, resolution=resolution_iterpol_center,
        xMinMax_frac=xMinMax_frac_center, yMinMax_frac=yMinMax_frac_center
    )
    if plotting:
        plot_field(field_interpol)

    # rescaling the beam width
    width = width / np.shape(field_norm)[0] * np.shape(field_interpol)[0]

    # plotting spec to select moments. .T because Danilo's code saving it like that
    if plotting:
        _ = cbs.LG_spectrum(field_interpol.T, **moments_init, mesh=mesh_interpol, plot=plotting, width=width, k0=1)

    # finding the beam center
    ## moments_init.update(moments_center)
    ## moments = moments_init
    x, y, eta, gamma = cbs.beamFullCenter(
        field_interpol, mesh_interpol,
        stepXY=(1, 1), stepEG=(3 / 180 * np.pi, 0.25 / 180 * np.pi),
        x=0, y=0, eta2=0., gamma=0.,
        **moments_center, threshold=1, width=width, k0=1, print_info=plotting
    )
    # x, y, eta, gamma = 0, 0, 0, 0

    # rescaling field to the scale we want for 3D calculations
    field_interpol2, mesh_interpol2 = field_interpolation(
        field_norm, mesh=mesh_init, resolution=resolution_interpol_working,
        xMinMax_frac=xMinMax_frac_working, yMinMax_frac=yMinMax_frac_working, fill_value=False
    )
    if plotting:
        plot_field(field_interpol2)

    # removing the tilt
    field_untilted = cbs.removeTilt(field_interpol2, mesh_interpol2, eta=-eta, gamma=gamma, k=1)
    if plotting:
        plot_field(field_untilted)

    # scaling the beam center
    shape = np.shape(field_untilted)
    x = int(x / np.shape(field_interpol)[0] * shape[0])
    y = int(y / np.shape(field_interpol)[1] * shape[1])

    # cropping the beam around the center
    field_cropped = field_untilted[
                    shape[0] // 2 - x - resolution_crop[0] // 2:shape[0] // 2 - x + resolution_crop[0] // 2,
                    shape[1] // 2 - y - resolution_crop[1] // 2:shape[1] // 2 - y + resolution_crop[1] // 2]
    if plotting:
        plot_field(field_cropped)

    # selecting the working field and mesh
    mesh = fg.create_mesh_XY(xRes=np.shape(field_cropped)[0], yRes=np.shape(field_cropped)[1])
    field = field_cropped
    print(f'field finished: {path[-20:]}')
    return field, mesh


def main_dots_building(
        field2D,
        plotting=True,
        dz=5,
        steps_both=25,
        resolution_crop=(100, 100),
        r_crop=30
):
    if plotting:
        plot_field(field2D)

    # 2-directional propagation
    # total resolution along z is (steps_both * x + 1)
    field3D = one_plane_propagator(field2D, dz=dz, stepsNumber_p=steps_both, stepsNumber_m=None, n0=1, k0=1)
    if plotting:
        plot_field_3D_multi_planes(field3D, number=6, columns=3)

    # cropping the 3D field, rectangular prism shape
    # it's used for the faster dots calculation
    shape = np.shape(field3D)
    if shape[0] >= resolution_crop[0] and shape[1] >= resolution_crop[1]:
        field3D_cropped = field3D[
                          shape[0] // 2 - resolution_crop[0] // 2:shape[0] // 2 + resolution_crop[0] // 2,
                          shape[1] // 2 - resolution_crop[1] // 2:shape[1] // 2 + resolution_crop[1] // 2,
                          :]
    else:
        field3D_cropped = field3D
        print('Resolution is lower than the crop resolution')

    # getting singularity dots using all 3 cross-sections
    dots_init_dict, dots_init = sing.get_singularities(np.angle(field3D_cropped), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, dots_init, color='black', show=plotting, size=10)

    # cropping dots, in a square <= R
    # x0, y0 = resolution_crop[0] // 2, resolution_crop[1] // 2
    # dots_cropped_dict = {dot: 1 for dot in dots_init_dict if
    #                      (np.abs(dot[0] - x0) <= r_crop and np.abs(dot[1] - y0) <= r_crop)}
    # if plotting:
    #     dp.plotDots(dots_cropped_dict, dots_cropped_dict, color='black', show=plotting, size=10)

    # cropping dots, in a radius <= R
    x0, y0 = resolution_crop[0] // 2, resolution_crop[1] // 2
    dots_cropped_dict = {dot: 1 for dot in dots_init_dict if
                         np.sqrt((dot[0] - x0) ** 2 + (dot[1] - y0) ** 2) < r_crop}
    dp.plotDots(dots_cropped_dict, dots_cropped_dict, color='black', show=plotting, size=10)


    # moving dots to the corner to make 3D array smaller
    dots_moved_dict = {
        (dot[0] - (x0 - r_crop), dot[1] - (y0 - r_crop), dot[2]): 1 for dot in dots_cropped_dict}
    dp.plotDots(dots_moved_dict, dots_moved_dict, color='grey', show=plotting, size=12)

    # applying the dot simplification algorithm from dots_processing.py
    dots_filtered = dp.filtered_dots(dots_moved_dict)
    dp.plotDots(dots_filtered, dots_filtered, color='grey', show=plotting, size=12)
    dots_filtered_rounded = dots_rounded(dots_filtered, resolution_crop, x0=0, y0=0, z0=0)
    dots_filtered_dict = {tuple(dot): 1 for dot in dots_filtered_rounded}
    dots_filtered_twice = dp.filtered_dots(dots_filtered_dict, single_dot=True)
    dp.plotDots(dots_filtered_twice, dots_filtered_twice, color='red', show=plotting, size=12)
    # dots = np.array([dot for dot in dots_filtered_twice])
    # return dots
    dots_raw = dp.filtered_dots(dots_moved_dict, single_dot=True)
    # dots_raw = [dot for dot in dots_moved_dict]
    return dots_raw, dots_filtered_twice
    # saving dots into a data_frame. Both processed as well as unprocessed
    # also saving the frames of the 3D knot for both of the
    ##########################################################
    print(dots_raw)

    print(dots_raw_rounded)
    dp.plotDots(dots_moved_dict, dots_moved_dict, color='black', show=True, size=12)
    dp.plotDots(dots_raw , dots_moved_dict, color='black', show=True, size=12)
    dp.plotDots(dots_raw_rounded, dots_raw_rounded, color='black', show=True, size=12)
    exit()

    z_max = np.shape(field3D_cropped)[-1]
    test_dots = np.zeros((r_crop * 2 + 1, r_crop * 2 + 1, z_max))
    for dot in dots_filtered:
        dot = list(map(round, dot))
        dot[0] -= x0
        dot[1] -= y0
        i, j, k = dot
        test_dots[i, j, k] = 1
    test_dots_dict = []
    for i in range(2 * r_crop + 1):
        for j in range(2 * r_crop + 1):
            for k in range(np.shape(field3D_cropped)[-1]):
                if test_dots[i, j, k] != 0:
                    test_dots_dict.append([i, j, k])
    print(test_dots)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test_dots[:, 0], test_dots[:, 1], test_dots[:, 2])
    exit()
    # test_dots = dots_with_zeros(
    #     dots_raw, radius=r_crop, z_dim=[0, np.shape(field3D_cropped)[-1]],
    #     resolution_crop=resolution_crop)
    # print(len(test_array))
    # test_array = dots_array_with_zeros(
    #     dots_filtered, radius=r_crop, z_dim=[0, np.shape(field3D_cropped)[-1]],
    #     resolution_crop=resolution_crop)
    # print(len(test_array))
    # test_dots = test_array.reshape((2 * r_crop + 1, 2 * r_crop + 1, np.shape(field3D_cropped)[-1]))
    # print(test_dots)
    # test_dots_dict = []
    # for i in range(2 * r_crop + 1):
    #     for j in range(2 * r_crop + 1):
    #         for k in range(np.shape(field3D_cropped)[-1]):
    #             if test_dots[i, j, k] != 0:
    #                 test_dots_dict.append([i, j, k])
    # print(test_dots_dict)
    # print(dots_raw)
    # test_dots_dict = np.array(test_dots_dict)
    dots_moved = np.array([dot for dot in dots_moved_dict])
    dp.plotDots(dots_moved, dots_raw, color='grey', show=True, size=12)
    dp.plotDots(dots_filtered, dots_raw, color='grey', show=True, size=12)
    # dp.plotDots(test_dots_dict, dots_raw, color='grey', show=True, size=12)
    # for i, value in test_array:
    #     x =
    exit()
    # exit()
    return {
        'dots_raw': list(dots_raw),
        'dots_filtered': list(dots_filtered),
        'radius': r_crop,
        'z_min': 0,
        'x0': x0,
        'y0': y0,
        'z_max': np.shape(field3D_cropped)[-1],
    }


def dots_rounded(dots, resolution_crop, x0=None, y0=None, z0=0):
    """

    :param dots:
    :param radius:
    :param z_dim:
    :return:
    """
    if x0 is None:
        x0 = resolution_crop[0] // 2
    if y0 is None:
        y0 = resolution_crop[1] // 2
    dots_centered = dots - [x0, y0, z0]
    dots_rounded = dots_centered.astype(int)
    return dots_rounded



def files_list(mypath, end='.mat'):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(end)]
    return onlyfiles

if __name__ == '__main__':
    directory_field = f'data/test/'
    directory_field_saved = f'data/test/saved/'
    directory_field_saved_dots = f'data/test/saved/dots/'
    directory_field_saved_plots = f'data/test/saved/plots/'
    if not os.path.isdir(directory_field_saved):
        os.makedirs(directory_field_saved)
    if not os.path.isdir(directory_field_saved_dots):
        os.makedirs(directory_field_saved_dots)
    if not os.path.isdir(directory_field_saved_plots):
        os.makedirs(directory_field_saved_plots)

    file_processing_ = True
    if file_processing_:

        files = files_list(directory_field, end='.mat')
        print(files)
        for file in files:
            print(file)
            field2D, _ = main_field_processing(
                path=directory_field + file,
                plotting=False,
                resolution_iterpol_center=(100, 100),
                xMinMax_frac_center=(1, 1),
                yMinMax_frac_center=(1, 1),
                resolution_interpol_working=(115, 115),
                xMinMax_frac_working=(1.2, 1.2),
                yMinMax_frac_working=(1.2, 1.2),
                resolution_crop=(100, 100),
                moments_init={'p': (0, 5), 'l': (-4, 4)},
                moments_center={'p0': (0, 5), 'l0': (-4, 4)},
            )
            file_save = directory_field_saved + file[:-4] + '.npy'
            print(file_save)
            np.save(file_save, field2D)
            plot_field(field2D, save=directory_field_saved_plots + file[:-4] + '.png')

    dots_building_ = True
    resolution_crop = (70, 70)
    if dots_building_:
        files = files_list(directory_field_saved, end='.npy')
        print(files)
        for file in files:
            print(file)
            field2D = np.load(directory_field_saved + file)
            dots_raw, dots_filtered = main_dots_building(
                field2D=field2D,
                plotting=False,
                dz=5.5,
                steps_both=25,
                resolution_crop=resolution_crop,
                r_crop=25
            )
            file_save_dots_raw = directory_field_saved_dots + 'raw_' + file[:-4] + '.npy'
            file_save_dots_filtered = directory_field_saved_dots + 'filtered_' + file[:-4] + '.npy'
            dp.plotDots(dots_raw, dots_raw, color='green', show=False, size=15,
                        save=directory_field_saved_plots + file[:-4] + '_3D.html')
            np.save(file_save_dots_raw, dots_raw)
            np.save(file_save_dots_filtered, dots_filtered)

    exit()
    test_name = directory_field_saved_dots + 'raw_' + 'Efield_1_SR_9.500000e-01.npy'
    dots = np.load(test_name)
    dp.plotDots(dots, dots, color='green', show=True, size=15)
    test_name = directory_field_saved_dots + 'filtered_' + 'Efield_1_SR_9.500000e-01.npy'
    dots = np.load(test_name)
    dp.plotDots(dots, dots, color='green', show=True, size=15)
    # print(dots)
# x=0, y=0, eta=6.0*, gamma=2.0*, var=0.08804003185287904  70 70
