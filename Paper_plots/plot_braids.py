from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg
import plotly.graph_objects as go
import numpy as np
from vispy import app, gloo, visuals, scene
# from mayavi import mlab
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io

hopf_braid1 = np.load('hopf_braid1_w=1d1_x=5d5_scale=0d25.npy')
sorted_indices = np.argsort(hopf_braid1[:, -1])[::-1]
hopf_braid1_sorted = hopf_braid1[sorted_indices]
x = hopf_braid1_sorted[:, 0]
y = hopf_braid1_sorted[:, 1]
z = hopf_braid1_sorted[:, 2]
# boundary_3D = [[0, 0, 0], [100, 100, 100]]
# dp.plotDots(hopf_braid1, boundary_3D, color='red', show=True, size=7)
# plt.show()
print(x, z)
# plotly
if 0:
	fig = go.Figure(data=[go.Scatter3d(
	    x=x,
	    y=y,
	    z=z,
	    mode='lines',
	)])
	fig.show()

# plotly figure_factory
if 0:
	import plotly.figure_factory as ff
	
	fig = ff.create_3d_line(x, y, z)
	
	# Show the figure
	fig.show()

# canvas
if 0:
	canvas = scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()
	line = scene.Line(np.column_stack((x, y, z)), color='red', method='agg', width=5, parent=view.scene)
	axis = scene.visuals.XYZAxis(parent=view.scene)
	view.camera = 'turntable'
	view.camera.set_range()
	app.run()
	
# canvas 2
if 1:
	canvas = scene.SceneCanvas(keys='interactive', show=True)  #, bgcolor='white')
	view = canvas.central_widget.add_view()
	points = np.column_stack((x, y, z))
	tube = scene.visuals.Tube(points, color='red', shading='smooth',
	                          parent=view.scene)
	
	# Auto-scale to see the whole line
	view.camera = 'turntable'
	view.camera.set_range()
	
	app.run()

# fig = go.Figure(data=[go.Scatter3d(x=hopf_braid1[:, 0], y=hopf_braid1[:, 1], z=hopf_braid1[:, 2],
#                                    mode='markers')])
# fig.show()
# mlab.plot3d(x, y, z, tube_radius=0.1)
# mlab.show()

