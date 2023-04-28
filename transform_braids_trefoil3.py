from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
	def cos_v(x, y, z, power=1):
		return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2
	
	def sin_v(x, y, z, power=1):
		return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j
	
	if angle == 0:
		return u(x, y, z) * np.exp(1j * theta) - (
				cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)
	else:
		print(z)
		exit()
		if z < 2 / 3 * np.pi:
			return u(x, y, z) * np.exp(1j * theta) - (
					cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)
		else:
			angle = 0.25 * np.pi
			return u(x, y, z) * np.exp(1j * theta) - (
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


"""used modules"""
plot_milnor_field = 1
plot_milnor_lines = 1
plot_braids = 1
plot_real_field = 1
plot_real_lines = 1
"""beam parameters"""
w = 1.2

# LG spectrum
moments = {'p': (0, 9), 'l': (-5, 5)}
"""mesh parameters"""
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1.0, 1.0)
res_x_3D, res_y_3D, res_z_3D = 31, 31, 31
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
	(mesh_3D[0], mesh_3D[1], mesh_3D[2]),
]
z_ind = res_z_3D // 2
# starting angle for each braid
angle_array = [0, np.pi * 1]
# powers in cos in sin
pow_cos_array = [1, 1]
pow_sin_array = [1, 1]
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
