from functions_based import *
import my_functions.beams_and_pulses as bp
import my_functions.plotings as pl

x_lim_3D, y_lim_3D, z_lim_3D = (-5.5, 5.5), (-5.5, 5.5), (-0.9, 0.9)
x_lim_3D, y_lim_3D, z_lim_3D = (-2.5, 2.5), (-2.5, 2.5), (-0.7, 0.7)
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1, 1)
x_lim_3D, y_lim_3D, z_lim_3D = (-2.4, 2.4), (-2.4, 2.4), (-0.75, 0.75)
x_lim_3D, y_lim_3D, z_lim_3D = (-2.5, 2.5), (-2.5, 2.5), (-0.75, 0.75)
x_lim_3D, y_lim_3D, z_lim_3D = (-4*1.6, 4*1.6), (-4*1.6, 4*1.6), (-0.75, 0.75)
# x_lim_3D, y_lim_3D, z_lim_3D = (-2*1.6, 2*1.6), (-2*1.6, 2*1.6), (-1.0, 1.0)
# x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.2*1.6, 2.2*1.6), (-2.2*1.6, 2.2*1.6), (-1.5, 1.5)
# x_lim_3D, y_lim_3D, z_lim_3D = (-2.2*1.6, 2.2*1.6), (-2.2*1.6, 2.2*1.6), (-3, 3)
res_x_3D, res_y_3D, res_z_3D = 551, 551, 3  # 2D
res_x_3D, res_y_3D, res_z_3D = 80, 80, 80
# res_x_3D, res_y_3D, res_z_3D = 51, 51, 51
x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')  #
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #
mesh_3D_res = np.meshgrid(np.arange(res_x_3D), np.arange(res_y_3D), np.arange(res_z_3D), indexing='ij')
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
cmapF = 'hsv'
cmapE = 'hot'

def x_iy(x, y):
    return x + 1j * y


def screw(x, y, z):
    return (x + 1j * y) * np.exp(1j * z)


def edge(x, y):
    return (x + 1j * y) * np.exp(1j * y)


# x_iy
if False:
    field_x_iy = x_iy(*mesh_2D)
    plot_field(field_x_iy, axes=False)
    plt.show()
# edge
if False:
    field_edge = edge(*mesh_2D)
    plot_field(field_edge, axes=False)
    plt.show()
# LG
if 0:
    field = bp.LG_simple(*mesh_2D, l=3, p=3, width=1)
    plot_field(field/field.max(), axes=False)
    # plot_field(field/field.max(), axes=False, cmap='viridis')
    plt.show()
    exit()
    field = bp.LG_simple(*mesh_2D, l=1, p=1)
    plot_field(field, axes=False)
    plt.show()
    field = bp.LG_simple(*mesh_2D, l=2, p=1)
    plot_field(field, axes=False)
    plt.show()
    field = bp.LG_simple(*mesh_2D, l=-1, p=1)
    plot_field(field, axes=False)
    plt.show()
    field = bp.LG_simple(*mesh_2D, l=-2, p=1)
    plot_field(field, axes=False)
    plt.show()
    exit()

# LG 3D
if 0:
    field = bp.LG_simple(*mesh_3D, l=1, p=0) / bp.LG_simple(*mesh_3D, l=1, p=0).max()
    # plot_field(field, axes=False)
    # plt.show()
    Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
                             opacityscale='max', colorscale='Jet')
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=10, fig=Fig)
    plt.show()
    exit()
    
# LG 3D turb
if 0:
    import scipy.fft as fft
    # field = bp.LG_simple(*mesh_3D, l=1, p=0) / bp.LG_simple(*mesh_3D, l=1, p=0).max()
    field = (bp.LG_simple(*mesh_3D, l=1, p=0) +
             bp.LG_simple(*mesh_3D, l=2, p=0) / 20 +
             bp.LG_simple(*mesh_3D, l=0, p=4) / 20 -
             bp.LG_simple(*mesh_3D, l=0, p=3) / 20 -
             bp.LG_simple(*mesh_3D, l=0, p=1) / 20
    )
    # plot_field(field, axes=False)
    # plt.show()
    # exit()
    # field = fft.fft(field) * np.exp(1j * (np.random.rand(*np.shape(field)) - 0.5) / 5)
    # field = fft.fft(field) * np.exp(1j * (mesh_3D) / 5)
    #
    # field = fft.ifft(field)

    Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
                             opacityscale='max', colorscale='Jet')
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=10, fig=Fig)
    plt.show()
    exit()

# LGs 3D
if 0:
    field = bp.LG_simple(*mesh_3D, l=1, p=0) + bp.LG_simple(*mesh_3D, l=0, p=1)
    field = field / field.max()
    # plot_field(field, axes=False)
    # plt.show()
    Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
                             opacityscale='max', colorscale='Jet')
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=10, fig=Fig)
    plt.show()

# trefoil 3D
if 1:
    # # Denis
    C00 = 1.51
    C01 = -5.06
    C02 = 7.23
    C03 = -2.04
    C30 = -3.97
    # (0.0011924249760862221 + 1.1372720865198616e-05j), (-0.002822503524696713 + 8.535015090975148e-06j), (
    #             0.0074027513552254 + 5.475152609562589e-06j), (-0.0037869189890120283 + 8.990311302510449e-06j), (
    #             -0.0043335243263204586 + 8.720849197446181e-07j)]}
    # C00 = 0.0011924249760862221
    # C01 = -0.002822503524696713
    # C02 = 0.0074027513552254
    # C03 = -0.0037869189890120283
    # C30 = -0.0043335243263204586
    # normal
    # C00 = 1.71
    # C01 = -5.66
    # C02 = 6.38
    # C03 = -2.3
    # C30 = -4.36  # * np.exp(1j * (np.pi / 2 + np.pi / 6))
    # # our
    # C00 = 1.55
    # C01 = -5.11
    # C02 = 8.29
    # C03 = -2.37
    # C30 = -5.36
    width = 1.28
    width = 1.6
    field = (
            C00 * bp.LG_simple(*mesh_3D, l=0, p=0, width=width) +
            C01 * bp.LG_simple(*mesh_3D, l=0, p=1, width=width) +
            C02 * bp.LG_simple(*mesh_3D, l=0, p=2, width=width) +
            C03 * bp.LG_simple(*mesh_3D, l=0, p=3, width=width) +
            C30 * bp.LG_simple(*mesh_3D, l=3, p=0, width=width)
    )
    field = field / field.max()
    # plot_field(field, axes=True)
    plot_field(field, titles=('', ''), intensity=False, cmapF=cmapF, cmapE=cmapE, axes=False)
    plt.show()
    # plt.show()
    # exit()
    # Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
    #                          opacityscale='max', colorscale='Jet')
    # Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False,
    #                          surface_count=20, resDecrease=(4, 4, 4),
    #                             opacity=1, colorscale='Jet',
    #                             opacityscale=[[0, 0.1], [0.15, 0.20], [1, 0.40]])
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)
    fig = dp.plotDots(dots_init, boundary_3D, color='red', show=False, size=12)#, fig=Fig)
    file_name = (
            f'trefoil_math_w={str(width).replace(".", "d")}_x={str(x_3D.max()).replace(".", "d")}' +
            f'_resXY={res_x_3D}_resZ={res_z_3D}'
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))

    # np.save(file_name, np.array(dots_init))
    fig.show()

# trefoil in Milnor space
if 0:
    def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
        def cos_v(x, y, z, power=1):
            return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2
        
        def sin_v(x, y, z, power=1):
            return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

        return u(x, y, z) * np.exp(1j * theta) - (
                cos_v(x, y, z, pow_cos) / a_cos + 1j
                * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)
        # cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle_3D)

# trefoil polynomials
if 0:
    C00 = 1.71
    C01 = -5.66
    C02 = 6.38
    C03 = -2.3
    C30 = -4.36
    field = (
            C00 * bp.LG_simple(*mesh_3D, l=0, p=0) +
            C01 * bp.LG_simple(*mesh_3D, l=0, p=1) +
            C02 * bp.LG_simple(*mesh_3D, l=0, p=2) +
            C03 * bp.LG_simple(*mesh_3D, l=0, p=3) +
            C30 * bp.LG_simple(*mesh_3D, l=3, p=0)
    ) / bp.LG_simple(*mesh_3D, l=0, p=0) / ((1 + mesh_3D[0] ** 2 + mesh_3D[1] ** 2) ** 3)
    field = field / field.max()
    # field = (mesh_3D[0] ** 2 + mesh_3D[1] ** 2) ** 1.5
    plot_field(field, axes=False)
    plt.show()
    exit()
    Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
                             opacityscale='max', colorscale='Jet')
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=10, fig=Fig)
    plt.show()
    
# trefoil polynomials modifications
if 0:
    C00 = 1.71
    C01 = -5.66
    C02 = 6.38
    C03 = -2.3
    C30 = -4.36
    field = (
            C00 * bp.LG_simple(*mesh_3D, l=0, p=0) +
            C01 * bp.LG_simple(*mesh_3D, l=0, p=1) +
            C02 * bp.LG_simple(*mesh_3D, l=0, p=2) +
            C03 * bp.LG_simple(*mesh_3D, l=0, p=3) +
            C30 * bp.LG_simple(*mesh_3D, l=3, p=0)
    ) / ((1 + mesh_3D[0] ** 2 + mesh_3D[1] ** 2) ** 3)
    
    field = field / field.max()
    # field = (mesh_3D[0] ** 2 + mesh_3D[1] ** 2) ** 1.5
    plot_field(field, axes=False)
    plt.show()
    exit()
    Fig = pl.plot_3D_density(np.abs(field), mesh=mesh_3D_res, show=False, opacity=0.15,
                             opacityscale='max', colorscale='Jet')
    _, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='black', show=True, size=10, fig=Fig)
    plt.show()