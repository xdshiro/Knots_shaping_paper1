# %%
import numpy as np
import sys

sys.path.append("C:\\WORK\\CODES\\OAM_research")
import matplotlib.pyplot as plt
import knots_ML.dots_processing as dp
import my_functions.singularities as sing
import knots_ML.data_generation as dg
import my_functions.functions_general as fg
import knots_ML.center_beam_search as cbs
import my_functions.beams_and_pulses as bp
import my_functions.plotings as pl
import functools


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
                    a_cos_array=None, a_sin_array=None, braid_func=braid):
    ans = 1
    if theta_array is None:
        theta_array = [0] * len(angle_array)
    if a_cos_array is None:
        a_cos_array = [1] * len(angle_array)
    if a_sin_array is None:
        a_sin_array = [1] * len(angle_array)
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
def plot_field(field):
    if len(np.shape(field)) == 3:
        field2D = field[:, :, np.shape(field)[2] // 2]
    else:
        field2D = field
    plt.subplots(1, 2, figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(field2D))
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field2D), cmap='jet', interpolation='nearest')  # , cmap='twilight', interpolation='nearest'
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.tight_layout()


beam_rotation_test = False
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

# %%


# %%
if False:
    w_spec = 1
    res_xy = 111
    res_z = 3
    w = 2.5
    z_value_array = [0.0 * w ** 2]
    z_ind = res_z // 2
    z_ind = 1
    # "partial application" or "currying"

    x_lim, y_lim, z_lim = (-10, 10), (-10, 10), (-0.9, 0.9)
    x = np.linspace(*x_lim, res_xy)
    y = np.linspace(*y_lim, res_xy)
    z = np.linspace(*z_lim, res_z)
    mesh = np.meshgrid(x, y, z)
    width = res_xy / (x_lim[1] - x_lim[0])
    xyz_array = [(mesh[0] / 1.2 - 0 * 4, mesh[1] / 1.2, mesh[2] + 0 * 0.05 * w ** 2), (mesh[0] + 0 * 4, *mesh[1:])]  # 3
    angle_array = [0, 1 * np.pi]
    pow_cos_array = [1, 1]
    pow_sin_array = [1, 1]
    conj_array = [0, 0]
    a_cos_array = [1, 1]
    a_sin_array = [1, 1]
    field = field_of_braids(xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array, a_cos_array, a_sin_array)
    R = np.sqrt(mesh[0] ** 2 + mesh[1] ** 2)
    field_milnor = field * (1 + R ** 2) ** 2
    field_gauss = field_milnor * bp.LG_simple(*mesh[:2], 0, l=0, p=0, width=1 * w, k0=1, x0=0, y0=0, z0=0)
    field_norm = dg.normalization_field(field_gauss)
    plot_field(field_norm[:, :, z_ind])
    plt.show()
    moments = {'p': (0, 9), 'l': (-4, 4)}
    moment0 = moments['l'][0]
    # mesh2D_ind = fg.create_mesh_XY(xRes=res_xy, yRes=res_xy)
    mesh2D_ind = np.meshgrid(x, y, indexing='ij')
    # test = bp.LG_simple(*mesh2D_ind, z=1, l=1, p=1, width= w, k0=1)
    # plot_field(test)
    # plt.show()
    values_total = 0
    for z_value in z_value_array:
        new_function = functools.partial(bp.LG_simple, z=z_value)
        values = cbs.LG_spectrum(field_norm[:, :, z_ind], **moments, mesh=mesh2D_ind, plot=True, width=w * w_spec, k0=1,
                                 # width=width * w
                                 functions=new_function)
        values_total += values
    l1, l2, p1, p2 = moments['l'][0], moments['l'][1], moments['p'][0], moments['p'][1]

    pl.plot_2D(np.abs(values_total), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
               interpolation='none', grid=True, xname='l', yname='p', show=False)
    plt.yticks(np.arange(p1, p2 + 1))
    plt.xticks(np.arange(l1, l2 + 1))
    plt.show()
    values = values_total
    field_new = np.zeros((res_xy, res_xy)).astype(np.complex128)
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            # print(f'l={l + moment0}, p={p}: {np.abs(value)}')
            field_new += value * bp.LG_simple(*mesh2D_ind, z=0, l=l + moment0, p=p, width=w * w_spec, k0=1, x0=0, y0=0,
                                              z0=0)
    plot_field(field_new)
    plt.show()

    res_xy_3D = 71
    res_z_3D = 71
    x_lim_3D, y_lim_3D, z_lim_3D = (-1.3 * w, 1.3 * w), (-1.3 * w, 1.3 * w), (-0.25 * w ** 2, 0.25 * w ** 2)
    x_3D = np.linspace(*x_lim_3D, res_xy_3D)
    y_3D = np.linspace(*y_lim_3D, res_xy_3D)
    z_3D = np.linspace(*z_lim_3D, res_z_3D)
    mesh_3D = np.meshgrid(x_3D, y_3D, z_3D)
    boundary_3D = [[0, 0, 0], [res_xy_3D, res_xy_3D, res_z_3D]]
    field_new_3D = np.zeros((res_xy_3D, res_xy_3D, res_z_3D)).astype(np.complex128)
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p, width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
        # if np.abs(value) > 0.01:
        # print(f'l={l+ moment0}, p={p}: {np.abs(value)}')

    _, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
    plt.show()

# %%

w = 2.5
w_spec = 1
z_value_array = [0.0 * w ** 2]
res_xy_3D = 71
res_z_3D = 71
z_ind = res_z_3D // 2
x_lim_3D, y_lim_3D, z_lim_3D = (-11, 11), (-11, 11), (-1.0, 1.0)
# x_lim_3D, y_lim_3D, z_lim_3D = (-3, 3), (-3, 3), (-3.0, 3.0)
x_3D = np.linspace(*x_lim_3D, res_xy_3D)
y_3D = np.linspace(*y_lim_3D, res_xy_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
mesh_3D_rotated = rotate_meshgrid(*mesh_3D, np.radians(20), np.radians(20), np.radians(0))
mesh_3D_rotated2 = rotate_meshgrid(*mesh_3D, np.radians(20), np.radians(20), np.radians(0))
mesh2D_ind = np.meshgrid(x_3D, y_3D, indexing='ij')
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_xy_3D, res_xy_3D, res_z_3D]]
# xyz_array = [(mesh_3D[0] / 1, mesh_3D[1] / 1, mesh_3D[2])]  # , (mesh_3D[0] + 0*3, *mesh_3D[1:])]
xyz_array = [
    (mesh_3D_rotated[0], mesh_3D_rotated[1], mesh_3D_rotated[2]),
    (mesh_3D_rotated2[0], mesh_3D_rotated2[1], mesh_3D_rotated2[2])]
angle_array = [0, 1 * np.pi]
pow_cos_array = [1, 1]
pow_sin_array = [1, 1]
conj_array = [0, 0]
theta_array = [0.0 * np.pi, 0 * np.pi]
a_cos_array = [1, 1]
a_sin_array = [1, 1]
field = field_of_braids(xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array,
                        theta_array=theta_array, a_cos_array=a_cos_array, a_sin_array=a_sin_array,
                        )  # braid_func=braid_before_trans)
# field = rotate_3d_field_90(field, axis='x')
field_milnor = field * (1 + R ** 2) ** 2
field_gauss = field_milnor * bp.LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
field_norm = dg.normalization_field(field_gauss)
plot_field(field_gauss)
plt.show()
_, dots_init = sing.get_singularities(np.angle(field_gauss), axesAll=True, returnDict=True)
dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
plt.show()
# exit()
moments = {'p': (0, 10), 'l': (-3, 3)}
moment0 = moments['l'][0]

values_total = 0
for z_value in z_value_array:
    new_function = functools.partial(bp.LG_simple, z=z_value)
    values = cbs.LG_spectrum(
        field_norm[:, :, z_ind], **moments, mesh=mesh2D_ind, plot=True, width=w * w_spec, k0=1,
        functions=new_function
    )
    values_total += values
l1, l2, p1, p2 = moments['l'][0], moments['l'][1], moments['p'][0], moments['p'][1]

pl.plot_2D(np.abs(values_total), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
           interpolation='none', grid=True, xname='l', yname='p', show=False)
plt.yticks(np.arange(p1, p2 + 1))
plt.xticks(np.arange(l1, l2 + 1))
plt.show()
values = values_total
field_new = np.zeros((res_xy_3D, res_xy_3D)).astype(np.complex128)
for l, p_array in enumerate(values):
    for p, value in enumerate(p_array):
        # print(f'l={l + moment0}, p={p}: {np.abs(value)}')
        field_new += value * bp.LG_simple(*mesh2D_ind, z=0, l=l + moment0, p=p,
                                          width=w * w_spec, k0=1, x0=0, y0=0,
                                          z0=0)
plot_field(field_new)
plt.show()

res_xy_3D = 71
res_z_3D = 71
x_lim_3D, y_lim_3D, z_lim_3D = (-1.3 * w, 1.3 * w), (-1.3 * w, 1.3 * w), (-0.25 * w ** 2, 0.25 * w ** 2)
x_3D = np.linspace(*x_lim_3D, res_xy_3D)
y_3D = np.linspace(*y_lim_3D, res_xy_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D)
boundary_3D = [[0, 0, 0], [res_xy_3D, res_xy_3D, res_z_3D]]
field_new_3D = np.zeros((res_xy_3D, res_xy_3D, res_z_3D)).astype(np.complex128)
for l, p_array in enumerate(values):
    for p, value in enumerate(p_array):
        field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
                                             width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
    # if np.abs(value) > 0.01:
    # print(f'l={l+ moment0}, p={p}: {np.abs(value)}')

_, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=True, returnDict=True)
dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
plt.show()
exit()
res_xy = 51
res_z = 51
z_ind = 1
x_lim, y_lim, z_lim = (-1.0, 1.0), (-1.0, 1.0), (0, 2 * np.pi)
x = np.linspace(*x_lim, res_xy)
y = np.linspace(*y_lim, res_xy)
z = np.linspace(*z_lim, res_z)
mesh = [np.meshgrid(x, y, z)]  # , np.meshgrid(x, y, z)]
boundary_3D = [[0, 0, 0], [res_xy, res_xy, res_z]]
braid = field_of_braids(mesh, angle_array, pow_cos_array, pow_sin_array, conj_array,
                        theta_array=theta_array, a_cos_array=a_cos_array, a_sin_array=a_sin_array,
                        braid_func=braid_before_trans)

_, dots_init = sing.get_singularities(np.angle(braid), axesAll=False, returnDict=True)
dp.plotDots(dots_init, boundary_3D, color='blue', show=True, size=7)
plt.show()
