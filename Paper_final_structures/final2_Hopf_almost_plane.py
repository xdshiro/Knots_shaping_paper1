from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io
from sklearn.neighbors import NearestNeighbors

def gauss_z(x, y, z, width):
    if width is None:
        return 1
    else:
        return np.exp(-(z) ** 2 / width ** 2)  # + np.exp(-(z + 0.5) ** 2 / width ** 2)


def LG_simple_xyz(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0, width_gauss=None):
    """
    Classic LG beam
    :param l: azimuthal index
    :param p: radial index
    :param width: beam waste
    :param k0: wave number
    :param x0: center of the beam in x
    :param y0: center of the beam in y
    :return: complex field
    """

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    x = x - x0
    y = y - y0
    z = z - z0
    zR = k0 * width ** 2

    E = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
         * fg.rho(x, y) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-fg.rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    if width_gauss is not None:
        return E * gauss_z(x=x, y=y, z=z, width=width_gauss)
    else:
        return E


def LG_spectre_coeff_3D(field, l, p, xM=(-1, 1), yM=(-1, 1), zM=(-1, 1), width=1., k0=1., mesh=None,
                        functions=bp.LG_simple):
    """
    Function calculates a single coefficient of LG_l_p in the LG spectrum of the field
    :param field: complex electric field
    :param l: azimuthal index of LG beam
    :param p: radial index of LG beam
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :return: complex weight of LG_l_p in the spectrum
    """
    if mesh is None:
        shape = np.shape(field)
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, zMinMax=yM, xRes=shape[0], yRes=shape[1], zRes=shape[2])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray, zArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0]) * (zArray[1] - zArray[0])
    # print(123, xArray)
    # shape = np.shape(field)
    # xyMesh = fg.create_mesh_XY_old(xMax=xM[1], yMax=yM[1], xRes=shape[0], yRes=shape[1], xMin=xM[0], yMin=yM[0])
    LGlp = functions(*mesh, l=l, p=p, width=width, k0=k0)
    # plt.imshow(LGlp)
    # plt.show()
    # print('hi')
    return np.sum(field * np.conj(LGlp)) * dS


def LG_spectrum(beam, l=(-3, 3), p=(0, 5), xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, plot=True,
                functions=bp.LG_simple, **kwargs):
    """

    :param beam:
    :param l:
    :param p:
    :param xM:
    :param yM:
    :param width:
    :param k0:
    :param mesh:
    :param plot:
    :return:
    """
    print('hi')
    l1, l2 = l
    p1, p2 = p
    spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1), dtype=complex)
    # spectrumReal = []
    # modes = []
    for l in np.arange(l1, l2 + 1):
        for p in np.arange(p1, p2 + 1):
            value = LG_spectre_coeff_3D(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
                                        functions=functions, **kwargs)
            # print(l, p, ': ', value, np.abs(value))
            spectrum[l - l1, p] = value
    # if np.abs(value) > 0.5:
    # spectrumReal.append(value)
    # modes.append((l, p))
    # print(modes)
    if plot:
        import matplotlib.pyplot as plt
        pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='l', yname='p', show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
    return spectrum


def u(x, y, z):
    numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def v(x, y, z):
    numerator = 2 * (x + 1j * y)
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


"""used modules"""
plot_milnor_field = 1
plot_milnor_lines = 0
plot_braids = 0
plot_real_field = 1
plot_real_lines = 0
cmapF = 'hsv'
cmapE = 'afmhot'
cmapE = 'inferno'
cmapE = 'hot'
# ANGLE
alpha_size = 5
rotation_angle = 45 - np.degrees(np.arcsin(1 / alpha_size / np.sqrt(2)))
rotation_angle = (45 - np.degrees(np.arcsin(1 / alpha_size / np.sqrt(2))))

# rotation_angle = (40)
# rotation_angle = (35)
# rotation_angle = 0
# rotation_angle = 35
# rotation_angle = 41

print(rotation_angle)
# exit()
# A, B, C = -1 * np.pi,  1 * np.pi, 0.25 * np.pi
# A, B, C = 0 - 0.0 * np.pi,  0 + 0.1 * np.pi, 0.5 * np.pi
# C_lobe1, C_lobe2, C_lobe3 = 0.25 * np.pi, 0.0 * np.pi, 0.0 * np.pi
x_lobe1, y_lobe1 = 1, 1
# x_lobe1, y_lobe1 = 1, 1

x_lobe2, y_lobe2 = 1, 1
# x_lobe2, y_lobe2 = 1, 1
C_lobe1, C_lobe2 = 0 / 6 * np.pi, 0.0 * np.pi
C_lobe1, C_lobe2 = np.arctan(np.tan(C_lobe1) / x_lobe1), C_lobe2 * 00

a_cos_array_CONST = [1, 1]
a_sin_array_CONST = [1, 1]
# a_cos_array_CONST = [1.3, 1.3]
# a_sin_array_CONST = [1.3, 1.3]
# shift = 0.3  # 0.2
shift = 0.0  # 0.2
l1, l2, l3 = 1, 0, 0
x_shift1, x_shift2 = +shift * l1, -shift * l2
y_shift1, y_shift2 = -0.0 * l1, +0
x_shift1 = +shift * np.cos(C_lobe1) * l1
y_shift1 = -shift * np.sin(C_lobe1) * l1
x_shift2 = -shift * np.cos(C_lobe2) * l2
y_shift2 = -shift * np.sin(C_lobe2) * l2
# z_shift1 = - 0.35
z_shift1 = 0
z_shift2 = 0
# z_shift2 = - 0
x_lim_3D, y_lim_3D, z_lim_3D = (-6.0, 6.0), (-6.0, 6.0), (-1.5 / alpha_size, 1.5 / alpha_size)
x_lim_3D, y_lim_3D, z_lim_3D = (-8.0, 8.0), (-8.0, 8.0), (-1.5 / alpha_size, 1.5 / alpha_size)
res_x_3D, res_y_3D, res_z_3D = 90, 90, 91
res_x_3D, res_y_3D, res_z_3D = 251, 251, 3
res_x_3D_k, res_y_3D_k, res_z_3D_k = 70, 70, 70
res_x_3D_k, res_y_3D_k, res_z_3D_k = 120, 120, 120
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.8, 2.8), (-2.8, 2.8), (-1.38, 1.38)
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.8, 2.8), (-2.8, 2.8), (-1.38 / alpha_size, 1.38 / alpha_size)
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.8, 2.8), (-2.8, 2.8), (-1.38 / alpha_size, 1.38 / alpha_size)


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1,
          braids_modification=None):
    def cos_v(x, y, z, power=1):
        return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

    x_new = np.array(x)
    y_new = np.array(y)
    z_new = np.array(z)
    angle_3D = np.ones(np.shape(z)) * angle
    a_cos_3D = np.ones(np.shape(z)) * a_cos
    a_sin_3D = np.ones(np.shape(z)) * a_sin
    phase = np.angle(x_new + 1j * y_new)
    # plot_field(phase)
    # plt.show()

    if 1:
        if braids_modification == 0:
            # plot_field(x_new)
            # plt.show()
            print('Braid 1:\nLobe 1')
            A, B = -3 / 3 * np.pi, 3 / 3 * np.pi
            # braid_scale = 1.0  # 1.2
            # x_scale = 1  # 1/1.2
            # x_shift = 0  # 0.5

            # phase_mask = (phase >= A) & (phase <= B)
            phase_mask = (phase > A) & (phase < B)
            # indexes = np.where(phase_mask)
            # angle_3D[indexes] += C_lobe1
            angle_3D[phase_mask] += C_lobe1
            # x_new[phase_mask] *= x_scale
            # angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] *= x_lobe1
            y_new[phase_mask] *= y_lobe1
            x_new[phase_mask] += x_shift1
            y_new[phase_mask] += y_shift1
            z_new[phase_mask] += z_shift1
        # x_new[phase_mask] *= x_lobe1
        # y_new[phase_mask] *= y_lobe1
        # plot_field(x_new)
        # plt.show()
        elif braids_modification == 1:
            # Lobe 2
            print('Braid 2\nLobe 2')
            # A, B = 0, 0
            A, B = -3 / 3 * np.pi, 3 / 3 * np.pi
            phase_mask = (phase >= A) & (phase <= B)
            angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] *= x_lobe2
            y_new[phase_mask] *= y_lobe2
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2
    # x_new[phase_mask] *= x_lobe2
    # y_new[phase_mask] *= y_lobe2

    # plot_field(x_new)
    # plt.show()
    # exit()

    # else:
    #     x_new = x
    #     y_new = y
    #     z_new = z
    #     angle_3D = angle
    #     a_cos_3D = a_cos
    #     a_sin_3D = a_sin
    if braids_modification in [0, 1]:
        return u(x_new, y_new, z_new) * np.exp(1j * theta) - (
                cos_v(x_new, y_new, z_new, pow_cos) / a_cos_3D + 1j
                * sin_v(x_new, y_new, z_new, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)
    else:
        return 1


# cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle_3D)


def braid_before_trans(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1,
                       braids_modification=None):
    def cos_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power + np.conj(np.exp(1j * z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power - np.conj(np.exp(1j * z)) ** power) / 2j

    # if z>0:
    #     angle +=np.py//4
    angle_3D = np.ones(np.shape(z)) * angle
    a_cos_3D = np.ones(np.shape(z)) * a_cos
    a_sin_3D = np.ones(np.shape(z)) * a_sin
    _, _, zAr = fg.arrays_from_mesh((x, y, z))
    # shape = np.shape(angle_3D)
    # angle_3D[:, :, shape[2] // 2 :] *= 1.3
    x_new = np.array(x)
    y_new = np.array(y)
    z_new = np.array(z)

    if 1:
        if braids_modification == 0:
            # plot_field(x_new)
            # plt.show()
            print('Braid 1:\nLobe 1')
            A, B = -3 / 3 * np.pi, 3 / 3 * np.pi
            # braid_scale = 1.0  # 1.2
            # x_scale = 1  # 1/1.2
            # x_shift = 0  # 0.5

            # phase_mask = (phase >= A) & (phase <= B)
            phase_mask = (zAr > A) & (zAr < B)
            # indexes = np.where(phase_mask)
            # angle_3D[indexes] += C_lobe1
            angle_3D[phase_mask] += C_lobe1
            # x_new[phase_mask] *= x_scale
            # angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] += x_shift1
            y_new[phase_mask] += y_shift1
            z_new[phase_mask] += z_shift1

        # plot_field(x_new)
        # plt.show()
        elif braids_modification == 1:
            # Lobe 2
            print('Braid 2\nLobe 2')
            # A, B = 0, 0
            A, B = -3 / 3 * np.pi, 3 / 3 * np.pi
            phase_mask = (zAr >= A) & (zAr <= B)
            angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2

    # angle_3D[phase_mask] += C_lobe3

    return (x + 1j * y) * np.exp(1j * theta) - (
            cos_v(x, y, z, pow_cos) / a_cos_3D + 1j * sin_v(x, y, z, pow_sin) / a_sin_3D
    ) * np.exp(1j * angle_3D)


# return (x + 1j * y) * np.exp(1j * theta) - (
#     cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)


def field_of_braids_separate_trefoil(mesh_3D, braid_func=braid, scale=None):
    # mesh_3D_rotated = rotate_meshgrid(*mesh_3D, np.radians(45), np.radians(30), np.radians(30))
    mesh_3D_rotated = rotate_meshgrid(*mesh_3D, np.radians(180 - rotation_angle), np.radians(00), np.radians(0))
    mesh_3D_rotated = rotate_meshgrid(*mesh_3D, np.radians(-rotation_angle), np.radians(00), np.radians(0))
    # mesh_3D_rotated = rotate_meshgrid(*mesh_3D, np.radians(180-43), np.radians(00), np.radians(0))
    mesh_3D_rotated_2 = rotate_meshgrid(*mesh_3D, np.radians(-180 + rotation_angle), np.radians(00), np.radians(0))
    mesh_3D_rotated_2 = rotate_meshgrid(*mesh_3D, np.radians(rotation_angle), np.radians(00), np.radians(0))
    # mesh_3D_rotated_2 = rotate_meshgrid(*mesh_3D, np.radians(43-180), np.radians(00), np.radians(0))
    # mesh_3D_rotated_2 =mesh_3D
    xyz_array = [
        (mesh_3D_rotated_2[0], mesh_3D_rotated_2[1], mesh_3D_rotated_2[2]),
        (mesh_3D_rotated[0], mesh_3D_rotated[1], mesh_3D_rotated[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [1, 1]
    pow_sin_array = [1, 1]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = a_cos_array_CONST
    a_sin_array = a_sin_array_CONST
    braids_modification = [0, 1]

    if theta_array is None:
        theta_array = [0] * len(angle_array)
    if a_cos_array is None:
        a_cos_array = [1] * len(angle_array)
    if a_sin_array is None:
        a_sin_array = [1] * len(angle_array)

    if braid_func is not braid and scale is None:
        scale = [1, 1, 1 * np.pi]
    if scale is not None:
        for i, xyz in enumerate(xyz_array):
            shape = np.shape(xyz)
            xyz_new = np.array(xyz).reshape((3, -1)).T * scale
            xyz_array[i] = tuple(xyz_new.T.reshape(shape))

    ans = 1

    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i],
                                           braids_modification=braids_modification[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i],
                              braids_modification=braids_modification[i])

    return ans


"""beam parameters"""
w = 1.6

# LG spectrum
moments = {'p': (0, 5), 'l': (-5, 5)}
"""mesh parameters"""

x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')  #
phase = np.angle(mesh_3D[0] + 1j * mesh_3D[1])
mesh_2D_xz = np.meshgrid(x_3D, z_3D, indexing='ij')  #
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
boundary_3D_k = [[0, 0, 0], [res_x_3D_k, res_y_3D_k, res_z_3D_k]]
"""creating the field"""
# mesh for each brade (in "Milnor" space)
# xyz_array = [
# 	(mesh_3D[0], mesh_3D[1], mesh_3D[2]),
# 	(mesh_3D[0], mesh_3D[1], mesh_3D[2])
# ]
y_ind = res_y_3D // 2 + 0
# # starting angle for each braid
# angle_array = [0, 1. * np.pi]
# # powers in cos in sin
# pow_cos_array = [1.5, 1.5]
# pow_sin_array = [1.5, 1.5]
# # conjugating the braid (in "Milnor" space)
# conj_array = [0, 0]
# # moving x+iy (same as in the paper)
# theta_array = [0.0 * np.pi, 0 * np.pi]
# # braid scaling
# a_cos_array = [1, 1]
# a_sin_array = [1, 1]
field = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid)
"""field transformations"""
# cone transformation
field_milnor = field * (1 + R ** 2) ** 2
field_gauss = field_milnor * bp.LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
field_norm = dg.normalization_field(field_gauss)
moment0 = moments['l'][0]
values_total = 0
y_value = 0
w_spec = 1
width_gauss = 0.75
width_gauss = 1
k_0_spec = 1.6
k_0_spec = 1
# field_norm = field_norm * gauss_z(*mesh_3D, width=width_gauss)
# pl.plot_3D_density(np.abs(field_norm))
# plt.show()
if plot_milnor_field:
    plot_field(field_norm)
    plt.show()
    # plot_field(field_norm[:, :, res_z_3D//2 - 10])
    # plt.show()
    # plot_field(field_norm[:, :, res_z_3D//2 - 20])
    # plt.show()
    # plot_field(field_norm[:, :, res_z_3D//2 - 30])
    # plt.show()
    # plot_field(field_norm[:, :, res_z_3D//2 - 40])
    # plt.show()
    # plot_field(field_norm[:, :, res_z_3D // 2 - 50])
    # plt.show()
    # plot_field(field_norm[:, :, res_z_3D // 2 - 60])
    # plt.show()
    plot_field(field_norm[:, y_ind, :])
    plt.show()
if plot_milnor_lines:
    _, dots_init = sing.get_singularities(np.angle(field_norm), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='blue', show=True, size=7)
    plt.show()

if plot_braids:
    scale = [0.25, 0.25, 2 * np.pi / (z_lim_3D[1] - z_lim_3D[0])]
    braid = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid_before_trans,
                                             scale=scale)
    # plot_field(braid)
    # plt.show()
    _, dots_init = sing.get_singularities(np.angle(braid), axesAll=False, returnDict=True)
    # print(dots_init)
    file_name = (
            f'hopf_braid1_w={str(w).replace(".", "d")}_x={str(x_3D.max()).replace(".", "d")}' +
            f'_scale={str(scale[0]).replace(".", "d")}_resXY={res_x_3D}_resZ={res_z_3D}'
    )
    # np.save(file_name, np.array(dots_init))
    dp.plotDots(dots_init, boundary_3D, color='red', show=True, size=7)
    plt.show()

# building 'LG' field
#################################################################################

# new_function = functools.partial(LG_simple_xz, y=y_value, width_gauss=width_gauss)#, width=w * w_spec)
new_function = functools.partial(LG_simple_xyz, width_gauss=width_gauss)  # , width=w * w_spec)
field_norm = field_norm * gauss_z(*mesh_3D, width=width_gauss)
# field_norm = np.load('trefoil3d.npy') * gauss_z(x=mesh_2D_xz[0], y=0, z=mesh_2D_xz[1], width=width_gauss)
# plot_field(new_function(*mesh_2D_xz, l=1, p=1))
# plot_field(np.load('trefoil3d.npy'))
# plt.show()
# exit()
# plot_field(field_norm[:, y_ind, :])
# plot_field(field_norm)
# plt.show()
# exit()
# values = LG_spectrum(
#     field_norm[:, y_ind, :], **moments, mesh=mesh_2D_xz, plot=True, width=w * w_spec, k0=1,
#     functions=new_function
# )
# !!!!!!!!!!!

##################################################################################################
values = cbs.LG_spectrum(
    field_norm[:, :, res_z_3D // 2], **moments, mesh=mesh_2D, plot=True, width=w * w_spec, k0=1,
)
# values = LG_spectrum(
#     field_norm[:, :, :], **moments, mesh=mesh_3D, plot=True, width=w * w_spec, k0=k_0_spec,
#     functions=new_function
# )

field_new_3D = np.zeros((res_x_3D, res_y_3D, res_z_3D)).astype(np.complex128)
total = 0
l_save = []
p_save = []
weight_save = []
modes_cutoff = 0.001

for l, p_array in enumerate(values):
    for p, value in enumerate(p_array):
        if abs(value) > modes_cutoff * abs(values).max():
            total += 1
            l_save.append(l + moment0)
            p_save.append(p)
            weight_save.append(value)
            # weights_important[f'{l + moment0}, {p}'] = value
            field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
                                                 width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
print(np.array(weight_save) / np.linalg.norm(np.array(weight_save)))
weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
field_new_3D = field_new_3D / np.abs(field_new_3D).max()
print(weights_important)
# scipy.io.savemat('weights_hopf_short_x5.mat', weights_important)
# exit()
if plot_real_field:
    plot_field(field_new_3D, titles=('', ''), intensity=False, cmapF=cmapF, cmapE=cmapE, axes=False)

    plt.show()
    # plot_field(field_new_3D[:, y_ind, :])
    # plt.show()
if plot_real_lines:
    x_3D_k = np.linspace(*x_lim_3D_k, res_x_3D_k)
    y_3D_k = np.linspace(*y_lim_3D_k, res_y_3D_k)
    z_3D_k = np.linspace(*z_lim_3D_k, res_z_3D_k)
    mesh_3D_k = np.meshgrid(x_3D_k, y_3D_k, z_3D_k, indexing='ij')  #
    field_new_3D_k = np.zeros((res_x_3D_k, res_y_3D_k, res_z_3D_k)).astype(np.complex128)
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                field_new_3D_k += value * bp.LG_simple(*mesh_3D_k, l=l + moment0, p=p,
                                                       width=w * w_spec, k0=1, x0=0, y0=0, z0=0)


    def find_path(coords, start_index=0):
        nbrs = NearestNeighbors(n_neighbors=len(coords)).fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        visited = set()
        current_index = start_index
        path = [current_index]
        dist = [0]
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


    _, dots_init = sing.get_singularities(np.angle(field_new_3D_k), axesAll=True, returnDict=True)
    dots_init = np.array([[dot[0], res_y_3D_k - dot[1], dot[2]] for dot in dots_init])
    path_ind, distances = find_path(dots_init, 0)
    cut = -1  # 2 other
    # cut = -7  # 2 other
    name_f = 'rotated_colored'
    # name_f = 'rotated_colored_zoomed5'
    # cut = -2  # normal 30
    # if not rotation_on and not shifting_z_on:
    #     cut = -4  # normal 25 27
    dots_init = dots_init[path_ind][:cut]

    sorted_indices = np.argsort(distances[:cut])
    # print(distances[:cut])
    two_largest_indices = sorted_indices[-2:]
    # print(two_largest_indices)
    dots1 = dots_init[:two_largest_indices.min()]  # normal
    dots2 = dots_init[two_largest_indices.min() + 1:two_largest_indices.max() - 3]  # normal
    # if not rotation_on and not shifting_z_on:
    #     name_f = ''
    #     dots1 = dots_init[:two_largest_indices.min()]  # normal
    #     dots2 = dots_init[two_largest_indices.min() + 1:two_largest_indices.max()]  # normal
    # if rotation_on and not shifting_z_on:
    #     name_f = 'rotated'
    #     dots1 = dots_init[:two_largest_indices.min()]  # rotated
    #     dots2 = dots_init[two_largest_indices.max()+1:]  # rotated
    # if rotation_on and shifting_z_on:
    #     name_f = 'rotated_shifted'
    #     dots1 = dots_init[:two_largest_indices.min()]  # rotated  + shifted
    #     dots2 = dots_init[two_largest_indices.min()+1:two_largest_indices.max()]  # rotated + shifted
    # name_f += name_extention
    dotSize = 12
    dots1_z0 = []
    dots2_z0 = []
    for dot in dots1:
        if dot[2] == res_z_3D_k // 2:
            dots1_z0.append(dot)
    if len(dots1_z0) > 2:
        dots1_z0 = [dots1_z0[0], dots1_z0[-1]]
    dots1_z0 = np.array(dots1_z0)
    for dot in dots2:
        if dot[2] == res_z_3D_k // 2:
            dots2_z0.append(dot)
    if len(dots2_z0) > 2:
        dots2_z0 = [dots2_z0[0], dots2_z0[-1]]
    dots2_z0 = np.array(dots2_z0)
    fig = dp.plotDots_Hopf(dots1, boundary_3D_k, color='red', show=False, size=dotSize)
    # dp.plotDots_Hopf(dots1_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    dp.plotDots_Hopf(dots2, boundary_3D_k, color='royalblue', show=False, size=dotSize, fig=fig)
    # dp.plotDots_Hopf(dots2_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)
    fig.show()
    exit()
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=-2.3, y=-2.3, z=2.3)  # Adjust x, y, and z to set the default angle of view
            )
        )
    )
    # fig.show()
    # exit()
    fig.write_html(f'hopf_{name_f}_t.html')
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=-4, z=0)  # Adjust x, y, and z to set the default angle of view
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.write_html(f'hopf_{name_f}_xz.html')
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=-4, y=0, z=0),  # Adjust x, y, and z to set the default angle of view
                up = dict(x=0, y=0, z=1)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.write_html(f'hopf_{name_f}_yz.html')
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
    fig.write_html(f'hopf_{name_f}_xy.html')
    # fig.write_html('hopf_rotated_shifted.html')
    plt.show()
if plot_real_lines and 0:
    x_3D_k = np.linspace(*x_lim_3D_k, res_x_3D_k)
    y_3D_k = np.linspace(*y_lim_3D_k, res_y_3D_k)
    z_3D_k = np.linspace(*z_lim_3D_k, res_z_3D_k)
    mesh_3D_k = np.meshgrid(x_3D_k, y_3D_k, z_3D_k, indexing='ij')  #
    field_new_3D_k = np.zeros((res_x_3D_k, res_y_3D_k, res_z_3D_k)).astype(np.complex128)
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                field_new_3D_k += value * bp.LG_simple(*mesh_3D_k, l=l + moment0, p=p,
                                                       width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
    _, dots_init = sing.get_singularities(np.angle(field_new_3D_k), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, boundary_3D_k, color='black', show=True, size=7)
    plt.show()
###################################################################
