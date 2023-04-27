from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
def LG_simple_xz(x, z, y=0, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0):
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
    return E
"""used modules"""
plot_milnor_field = 1
plot_milnor_lines = False
plot_braids = False
plot_real_field = True
plot_real_lines = False
"""beam parameters"""
w = 1.2

# LG spectrum
moments = {'p': (0, 9), 'l': (-4, 4)}
"""mesh parameters"""
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1.2, 1.2)
res_x_3D, res_y_3D, res_z_3D = 71, 71, 71
x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')  #
mesh_2D_xz = np.meshgrid(x_3D, z_3D, indexing='ij')  #
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
"""creating the field"""
# mesh for each brade (in "Milnor" space)
xyz_array = [
    (mesh_3D[0], mesh_3D[1], mesh_3D[2]),
    (mesh_3D[0], mesh_3D[1], mesh_3D[2])
]
y_ind = res_y_3D // 2 + 0
# starting angle for each braid
angle_array = [0, 1. * np.pi]
# powers in cos in sin
pow_cos_array = [1.5, 1.5]
pow_sin_array = [1.5, 1.5]
# conjugating the braid (in "Milnor" space)
conj_array = [0, 0]
# moving x+iy (same as in the paper)
theta_array = [0.0 * np.pi, 0 * np.pi]
# braid scaling
a_cos_array = [1, 1]
a_sin_array = [1, 1]
field = field_of_braids(
    xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array,
    theta_array=theta_array, a_cos_array=a_cos_array, a_sin_array=a_sin_array
)
"""field transformations"""
# cone transformation
field_milnor = field * (1 + R ** 2) ** 3
field_gauss = field_milnor * bp.LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
field_norm = dg.normalization_field(field_gauss)
# pl.plot_3D_density(np.abs(field_norm))
# plt.show()
if plot_milnor_field:
    plot_field(field_gauss[:, y_ind, :])
    plt.show()
if plot_milnor_lines:
    _, dots_init = sing.get_singularities(np.angle(field_norm), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='blue', show=True, size=7)
    plt.show()
if plot_braids:
    braid = field_of_braids(
        xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array,
        theta_array=theta_array, a_cos_array=a_cos_array, a_sin_array=a_sin_array,
        braid_func=braid_before_trans, scale=[0.1, 0.1, np.pi]
    )
    plot_field(braid)
    plt.show()
    _, dots_init = sing.get_singularities(np.angle(braid), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='red', show=True, size=7)
    plt.show()

# building 'LG' field
#################################################################################
moment0 = moments['l'][0]
values_total = 0
y_value = 0
w_spec = 1
new_function = functools.partial(LG_simple_xz, y=y_value, width=w * w_spec)
plot_field(new_function(*mesh_2D_xz))
plt.show()
exit()
values = cbs.LG_spectrum(
    field_norm[:, y_ind, :], **moments, mesh=mesh_2D_xz, plot=True, width=w * w_spec, k0=1,
    functions=new_function
)
field_new_3D = np.zeros((res_x_3D, res_y_3D, res_z_3D)).astype(np.complex128)
for l, p_array in enumerate(values):
    for p, value in enumerate(p_array):
        field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
                                             width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
if plot_real_field:
    plot_field(field_new_3D)
    plt.show()

if plot_real_lines:
    _, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
    plt.show()
###################################################################




