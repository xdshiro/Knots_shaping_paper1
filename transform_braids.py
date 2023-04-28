from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg
"""used modules"""
plot_milnor_field = 1
plot_milnor_lines = False
plot_braids = 0
plot_real_field = 1
plot_real_lines = 1
"""beam parameters"""
w = 1.2

# LG spectrum
moments = {'p': (0, 9), 'l': (-5, 5)}
"""mesh parameters"""
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1.0, 1.0)
res_x_3D, res_y_3D, res_z_3D = 91, 91, 71
x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')  #
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
"""creating the field"""
# mesh for each brade (in "Milnor" space)
xyz_array = [
    (mesh_3D[0], mesh_3D[1], mesh_3D[2]),
    (mesh_3D[0], mesh_3D[1], mesh_3D[2])
]
z_ind = res_z_3D // 2
# starting angle for each braid
angle_array = [0, 1 * np.pi]
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
    plot_field(field_gauss[:, :, z_ind])
    plt.show()
if plot_milnor_lines:
    _, dots_init = sing.get_singularities(np.angle(field_norm), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='blue', show=True, size=7)
    plt.show()
if plot_braids:
    braid = field_of_braids(
        xyz_array, angle_array, pow_cos_array, pow_sin_array, conj_array,
        theta_array=theta_array, a_cos_array=a_cos_array, a_sin_array=a_sin_array,
        braid_func=braid_before_trans, scale=[0.3, 0.3, np.pi]
    )
    # plot_field(braid)
    # plt.show()
    _, dots_init = sing.get_singularities(np.angle(braid), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='red', show=True, size=7)
    plt.show()

# building 'LG' field
#################################################################################
moment0 = moments['l'][0]
values_total = 0
z_value = 0

w_spec = 1
new_function = functools.partial(bp.LG_simple, z=z_value)
values = cbs.LG_spectrum(
    field_norm[:, :, z_ind], **moments, mesh=mesh_2D, plot=True, width=w * w_spec, k0=1,
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
    _, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=7)
    plt.show()
###################################################################




