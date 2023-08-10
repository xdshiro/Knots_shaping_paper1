# %%
import numpy as np
import sys
import matplotlib.pyplot as plt
import knots_ML.dots_processing as dp
import my_functions.singularities as sing
import knots_ML.data_generation as dg
# import my_functions.functions_general as fg
import knots_ML.center_beam_search as cbs
import my_functions.beams_and_pulses as bp
import my_functions.plotings as pl
import functools
sys.path.append("C:\\WORK\\CODES\\OAM_research")

# %%
def u(x, y, z):
    numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def v(x, y, z):
    numerator = 2 * (x + 1j * y)
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
    def cos_v(x, y, z, power=1):
        return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

    return u(x, y, z) * np.exp(1j * theta) - (
            cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)


def braid_before_trans(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
    def cos_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power + np.conj(np.exp(1j * z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power - np.conj(np.exp(1j * z)) ** power) / 2j

    return (x + 1j * y) * np.exp(1j * theta) - (
            cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)


def field_of_braids(xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array, theta_array=None,
                    a_cos_array=None, a_sin_array=None, braid_func=braid, scale=None):
    ans = 1
    if theta_array is None:
        theta_array = [0] * len(angle_array)
    if a_cos_array is None:
        a_cos_array = [1] * len(angle_array)
    if a_sin_array is None:
        a_sin_array = [1] * len(angle_array)
    if scale is not None:

        for i, xyz in enumerate(xyz_array):
            shape = np.shape(xyz)
            # print(np.array(xyz).reshape((3, -1)))
            xyz_new = np.array(xyz).reshape((3, -1)).T * scale
            xyz_array[i] = tuple(xyz_new.T.reshape(shape))

    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])

    return ans


def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotation_matrix_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def rotate_meshgrid(x, y, z, rx, ry, rz):
    R_x = rotation_matrix_x(rx)
    R_y = rotation_matrix_y(ry)
    R_z = rotation_matrix_z(rz)

    R = np.dot(R_z, np.dot(R_y, R_x))

    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()])
    rotated_xyz = np.dot(R, xyz)

    x_rotated = rotated_xyz[0].reshape(x.shape)
    y_rotated = rotated_xyz[1].reshape(y.shape)
    z_rotated = rotated_xyz[2].reshape(z.shape)

    return x_rotated, y_rotated, z_rotated


def rotate_3d_field_90(field, axis):
    """
    Rotate a 3D field clockwise by 90 degrees around the specified axis.

    :param field: A 3-dimensional list representing the 3D field.
    :param axis: The axis to rotate around, one of 'x', 'y', or 'z'.
    :return: A new 3-dimensional list representing the rotated field.
    """
    if axis not in {'x', 'y', 'z'}:
        raise ValueError("Invalid axis. Must be one of 'x', 'y', or 'z'.")

    z_len = len(field)
    y_len = len(field[0])
    x_len = len(field[0][0])
    new_field = None

    if axis == 'x':
        new_field = [
            [
                [field[z][y][x] for z in range(z_len)]
                for y in reversed(range(y_len))
            ]
            for x in range(x_len)
        ]

    elif axis == 'y':
        new_field = [
            [
                [field[z][y][x] for x in reversed(range(x_len))]
                for z in range(z_len)
            ]
            for y in range(y_len)
        ]

    elif axis == 'z':
        new_field = [
            [
                [field[z][y][x] for x in range(x_len)]
                for y in range(y_len)
            ]
            for z in reversed(range(z_len))
        ]

    return new_field


# %%
def plot_field(field, axes=True, titles=('Amplitude', 'Phase'), cmapE='afmhot', cmapF='jet', intensity=False):
    if len(np.shape(field)) == 3:
        field2D = field[:, :, np.shape(field)[2] // 2]
    else:
        field2D = field

    plt.rc('font', size=16, family='Times New Roman')  # controls default text sizes
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    plt.rc('axes', labelsize=24)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=122)  # fontsize of the figure title
    cbar_size = 25 # colorbar fontsize
    plt.subplots(1, 2, figsize=(11.5, 5))
    plt.subplot(1, 2, 1)
    if intensity:
        plt.imshow(np.abs(field2D.T) ** 2, cmap=cmapE, interpolation='nearest')
    else:
        plt.imshow(np.abs(field2D.T), cmap=cmapE, interpolation='nearest')
    cbar1 = plt.colorbar(fraction=0.04, pad=0.02)
    cbar1.ax.tick_params(labelsize=cbar_size)
    plt.title(titles[0])
    if not axes:
        plt.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field2D.T), cmap=cmapF, interpolation='nearest')  # , cmap='twilight', interpolation='nearest'
    cbar2 = plt.colorbar(fraction=0.04, pad=0.02)
    cbar2.ax.tick_params(labelsize=cbar_size)
    plt.title(titles[1])
    if not axes:
        plt.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)

if __name__ == "_main_":
    beam_rotation_test = True
    if beam_rotation_test:
        x_lim_3D, y_lim_3D, z_lim_3D = np.linspace(-2, 2, 50), np.linspace(-2, 2, 50), np.linspace(-2, 2, 50)
        mesh_3D = np.meshgrid(x_lim_3D, y_lim_3D, z_lim_3D, indexing='ij')
        mesh_3D = rotate_meshgrid(*mesh_3D, np.radians(45), np.radians(30), np.radians(30))
        # Hopf Dennis  (0, 0) 2.63; (0, 1) −6.32; (0, 2) 4.21; (2, 0) −5.95).
        beam = (
                +2.63 * bp.LG_simple(*mesh_3D, l=0, p=0) +
                -6.31 * bp.LG_simple(*mesh_3D, l=0, p=1) +
                +4.21 * bp.LG_simple(*mesh_3D, l=0, p=2) +
                -5.95 * bp.LG_simple(*mesh_3D, l=2, p=0)
        )
        # plt.imshow(np.angle(beam[:, :, 25]))
        # plt.show()
        # exit()
        _, dots_init = sing.get_singularities(np.angle(beam), axesAll=True, returnDict=True)
        boundary_3D = [[0, 0, 0], [len(x_lim_3D), len(y_lim_3D), len(z_lim_3D)]]
        dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
        plt.show()
        exit()