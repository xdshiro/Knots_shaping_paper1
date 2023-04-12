from functions_based import *

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
x_lim_3D, y_lim_3D, z_lim_3D = (-11, 11), (-11, 11), (-1.0, 1.0)
res_x_3D, res_y_3D, res_z_3D = 71, 71, 71
x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_xy_3D, res_xy_3D, res_z_3D]]
"""creating the field"""
# mesh for each brade (in "Mylnor" space)
xyz_array = [
    (mesh_3D_rotated[0], mesh_3D_rotated[1], mesh_3D_rotated[2]),
    (mesh_3D_rotated2[0], mesh_3D_rotated2[1], mesh_3D_rotated2[2])
]
# starting angle for each braid
angle_array = [0, 1 * np.pi]
# powers in cos in sin
pow_cos_array = [1, 1]
pow_sin_array = [1, 1]
# conjugating the braid (in "Mylnor" space)
conj_array = [0, 0]
# moving x+iy (same as in the paper)
theta_array = [0.0 * np.pi, 0 * np.pi]

a_cos_array = [1, 1]
a_sin_array = [1, 1]
####################################################################
exit()
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
