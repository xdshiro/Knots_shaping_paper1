from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
def gauss_z(x, y, z, width):
    return np.exp(-z ** 2 / width ** 2)

def LG_simple_xz(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0, width_gauss=None):
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

def LG_spectre_coeff(field, l, p, xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, functions=bp.LG_simple):
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
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # print(123, xArray)
    # shape = np.shape(field)
    # xyMesh = fg.create_mesh_XY_old(xMax=xM[1], yMax=yM[1], xRes=shape[0], yRes=shape[1], xMin=xM[0], yMin=yM[0])
    LGlp = functions(x=mesh[0], y=0, z=mesh[1], l=l, p=p, width=width, k0=k0)
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
            value = LG_spectre_coeff(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
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

"""used modules"""
plot_milnor_field = 0
plot_milnor_lines = False
plot_braids = False
plot_real_field = True
plot_real_lines = False
"""beam parameters"""
w = 1.2

# LG spectrum
moments = {'p': (0, 9), 'l': (-7, 7)}
"""mesh parameters"""
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1.2, 1.2)
res_x_3D, res_y_3D, res_z_3D = 51, 51, 51
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
    plot_field(field_gauss)
    plt.show()
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
width_gauss = 0.75

new_function = functools.partial(LG_simple_xz, y=y_value, width_gauss=width_gauss)#, width=w * w_spec)

field_norm = np.load('trefoil3d.npy') * gauss_z(x=mesh_2D_xz[0], y=0, z=mesh_2D_xz[1], width=width_gauss)
# plot_field(new_function(*mesh_2D_xz, l=1, p=1))
# plot_field(np.load('trefoil3d.npy'))
# plt.show()
# exit()
# plot_field(field_norm[:, y_ind, :])
# plot_field(field_norm)
# plt.show()
# exit()
values = LG_spectrum(
    field_norm[:, y_ind, :], **moments, mesh=mesh_2D_xz, plot=True, width=w * w_spec, k0=1,
    functions=new_function
)
# !!!!!!!!!!!
# values = cbs.LG_spectrum(
#     field_norm[:, :, res_z_3D // 2], **moments, mesh=mesh_2D, plot=True, width=w * w_spec, k0=1,
# )

field_new_3D = np.zeros((res_x_3D, res_y_3D, res_z_3D)).astype(np.complex128)
for l, p_array in enumerate(values):
    for p, value in enumerate(p_array):
        field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
                                             width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
      
      

if plot_real_field:
    plot_field(field_new_3D)
    plt.show()
    plot_field(field_new_3D[:, y_ind, :])
    plt.show()

if plot_real_lines:
    _, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
    plt.show()
###################################################################




