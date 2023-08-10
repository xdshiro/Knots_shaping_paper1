"""
Including all the ploting functions, 2D, 3D, dots
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import my_functions.functions_general as fg

# standard values for fonts
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def plot_2D(field, x=None, y=None, xname='', yname='', map='jet', vmin=None, vmax=None, title='',
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize, grid=False,
            axis_equal=False, xlim=None, ylim=None, ax=None, show=True, ijToXY=True, origin='lower',
            interpolation='bilinear',
            **kwargs) -> object:
    fieldToPlot = field
    if ijToXY:
        origin = 'lower'
        fieldToPlot = np.copy(field).transpose()
    if x is None:
        x = range(np.shape(fieldToPlot)[0])
    if y is None:
        y = range(np.shape(fieldToPlot)[1])
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(fieldToPlot,
                       interpolation=interpolation, cmap=map,
                       origin=origin, aspect='auto',  # aspect ration of the axes
                       extent=[x[0], x[-1], y[0], y[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd', **kwargs)
    cbr = plt.colorbar(image, ax=ax, shrink=0.8, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.grid(grid)
    if show:
        plt.show()
    return ax


def plot_scatter_2D(x, y, xname='', yname='', title='',
                    ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
                    axis_equal=False, xlim=None, ylim=None, ax=None, show=True,
                    size=plt.rcParams['lines.markersize'] ** 2, color=None,
                    **kwargs):
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x, y, s=size, color=color, **kwargs)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_plane_go(z, mesh, fig=None, opacity=0.6, show=False,
                  colorscale=([0, '#aa9ce2'], [1, '#aa9ce2']), **kwargs):
    """
    plotting the cross-section XY plane in 3d go figure
    :param z: z coordinate of the plane
     :param colorscale: need values for 0 and 1 (the same), or something like 'RdBu'
    :param kwargs: for go.Surface
    :return: fig
    """
    xyz = fg.arrays_from_mesh(mesh)

    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Surface(x=xyz[0], y=xyz[1], z=z,
                             opacity=opacity, colorscale=colorscale, showscale=False, **kwargs))
    if show:
        fig.show()
    return fig


def plot_3D_dots_go(dots, mode='markers', marker=None, fig=None, show=False, **kwargs):
    """
    plotting dots in the interactive window in browser using plotly.graph_objects
    :param dots: [[x,y,z],...]
    :param show: True if you want to show it instantly
    :return: fig
    """
    if marker is None:
        marker = {'size': 8, 'color': 'black'}
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=dots[:, 0], y=dots[:, 1], z=dots[:, 2],
                               mode=mode, marker=marker, **kwargs))
    if show:
        fig.show()
    return fig


def plot_3D_density(E, resDecrease=(1, 1, 1), mesh=None,
                    xMinMax=None, yMinMax=None, zMinMax=None,
                    surface_count=20, show=True,
                    opacity=0.5, colorscale='RdBu',
                    opacityscale=None, fig=None, **kwargs):
    """
    Function plots 3d density in the browser
    :param E: anything in real number to plot
    :param resDecrease: [a, b, c] steps in each direction
    :param xMinMax: values along x [xMinMax[0], xMinMax[1]] (boundaries)
    :param yMinMax: values along y [yMinMax[0], yMinMax[1]] (boundaries)
    :param zMinMax: values along z [zMinMax[0], zMinMax[1]] (boundaries)
    :param surface_count: numbers of layers to show. more layers - better resolution
                          but harder to plot it. High number can lead to an error
    :param opacity: needs to be small to see through all surfaces
    :param opacityscale: custom opacity scale [...] (google)
    :param kwargs: extra params for go.Figure
    :return: nothing since it's in the browser (not ax or fig)
    """
    if mesh is None:
        shape = np.array(np.shape(E))
        if resDecrease is not None:
            shape = (shape // resDecrease)
        if zMinMax is None:
            zMinMax = [0, shape[0]]
        if yMinMax is None:
            yMinMax = [0, shape[1]]
        if xMinMax is None:
            xMinMax = [0, shape[2]]

        X, Y, Z = np.mgrid[
                  xMinMax[0]:xMinMax[1]:shape[0] * 1j,
                  yMinMax[0]:yMinMax[1]:shape[1] * 1j,
                  zMinMax[0]:zMinMax[1]:shape[2] * 1j
                  ]
    else:
        X, Y, Z = mesh
    values = E[::resDecrease[0], ::resDecrease[1], ::resDecrease[2]]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),  # collapsed into 1 dimension
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=opacity,  # needs to be small to see through all surfaces
        opacityscale=opacityscale,
        surface_count=surface_count,  # needs to be a large number for good volume rendering
        colorscale=colorscale,
        **kwargs
    ))
    if show:
        fig.show()
    return fig


def plot_scatter_3D(X, Y, Z, ax=None, size=plt.rcParams['lines.markersize'] ** 2, color=None,
                    viewAngles=(70, 0), show=True, **kwargs):
    """
    ploting dots using plt.scatter
    :param ax: if you want multiple plots in one ax
    :param size: dots size. Use >100 for a better look
    :param color: color of the dots. Default for a single plot is blue
    :param viewAngles: (70, 0) (phi, theta)
    :param kwargs: extra parameters for plt.scatter
    :return: ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=size, color=color, **kwargs)  # plot the point (2,3,4) on the figure
    ax.view_init(*viewAngles)
    if show:
        plt.show()
    return ax


def box_set_go(fig, xyzMinMax=(-1, 1, -1, 1, -1, 1), width=3, perBox=0, mesh=None, autoDots=None,
               return_boundaries=False, aspects=(2, 2, 2), lines=True):
    """
    Function remove the standard layout and put the plot into the box of black lines
    :param fig: which fig should be updated
    :param xyzMinMax: boundaries for the box if there is no mesh
    :param width: width of the lines for the box
    :param perBox: percentage to make the box bigger
    :param mesh: if the mesh is given, boundaries are getting automatically
    :param autoDots: if autoDots == dots => finding the boundaries automatically from dots
    :return: fig
    """
    if autoDots is not None:
        dots = autoDots
        xMin, xMax = 1e10, 0
        yMin, yMax = 1e10, 0
        zMin, zMax = 1e10, 0
        for dot in dots:
            if dot[0] < xMin:
                xMin = dot[0]
            if dot[0] > xMax:
                xMax = dot[0]
            if dot[1] < yMin:
                yMin = dot[1]
            if dot[1] > yMax:
                yMax = dot[1]
            if dot[2] < zMin:
                zMin = dot[2]
            if dot[2] > zMax:
                zMax = dot[2]
    elif mesh is not None:
        xyz = fg.arrays_from_mesh(mesh)
        xMin, xMax = xyz[0][0], xyz[0][-1]
        yMin, yMax = xyz[1][0], xyz[1][-1]
        zMin, zMax = xyz[2][0], xyz[2][-1]
    else:
        xMin, xMax = xyzMinMax[0], xyzMinMax[1]
        yMin, yMax = xyzMinMax[2], xyzMinMax[3]
        zMin, zMax = xyzMinMax[4], xyzMinMax[5]
    if perBox != 0:
        xMin, xMax = xMin - (xMax - xMin) * perBox, xMax + (xMax - xMin) * perBox
        yMin, yMax = yMin - (yMax - yMin) * perBox, yMax + (yMax - yMin) * perBox
        zMin, zMax = zMin - (zMax - zMin) * perBox, zMax + (zMax - zMin) * perBox
    if lines:
        lineX = np.array([[xMin, yMin, zMin], [xMax, yMin, zMin], [xMax, yMax, zMin],
                          [xMin, yMax, zMin], [xMin, yMin, zMin]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMin, zMax], [xMax, yMin, zMax], [xMax, yMax, zMax],
                          [xMin, yMax, zMax], [xMin, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMin, zMin], [xMin, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMax, yMin, zMin], [xMax, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMax, yMax, zMin], [xMax, yMax, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMax, zMin], [xMin, yMax, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
    per = 0.01
    boundaries = [xMin - (xMax - xMin) * per, xMax + (xMax - xMin) * per,
                  yMin - (yMax - yMin) * per, yMax + (yMax - yMin) * per,
                  zMin - (zMax - zMin) * per, zMax + (zMax - zMin) * per]
    fig.update_layout(font_size=24, font_family="Times New Roman", font_color='black',
                      legend_font_size=20,
                      showlegend=False,

                      scene=dict(
                          # annotations=[dict(x=[ 2], y=[-2], z=[-.5], ax=-2, ay=-2), dict(align='left'),
                          #              dict(align='left')],
                          xaxis_title=dict(text='x', font=dict(size=45)),
                          yaxis_title=dict(text='y', font=dict(size=45)),
                          zaxis_title=dict(text='z', font=dict(size=45)),

                          aspectratio_x=aspects[0], aspectratio_y=aspects[1], aspectratio_z=aspects[2],

                          xaxis=dict(range=[xMin - (xMax - xMin) * per, xMax + (xMax - xMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False  # backgroundcolor='white',
                                     ),
                          yaxis=dict(range=[yMin - (yMax - yMin) * per, yMax + (yMax - yMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False
                                     ),
                          zaxis=dict(range=[zMin - (zMax - zMin) * per, zMax + (zMax - zMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False
                                     ), ),  # range=[-0.5, 0.5],
                      )
    if return_boundaries:
        return fig, boundaries
    return fig


if __name__ == '__main__':
    k1 = 80
    k2 = 70
    colors = np.array([[255, 255, k1], [255, 127, k1], [255, 0, k1],
                       [255, k2, 255], [127, k2, 255], [0, k2, 255]]) / 255
    plt.scatter([0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1], s=80000, marker='s', c=colors)
    plt.xlim(-1.1, 1.75)
    # plt.ylim(-0, 0.8)
    plt.show()
