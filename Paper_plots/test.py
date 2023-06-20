import numpy as np
import plotly.graph_objects as go

# Cylinder parameters
radius = 1
height = 2
center = [0, 0, 0]

# Number of segments in the circular cross section of the cylinder
segments = 100

# Generate the vertices of the cylinder
theta = np.linspace(0, 2*np.pi, segments)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
z_bottom = np.full_like(theta, center[2] - height/2)
z_top = np.full_like(theta, center[2] + height/2)

# Create Scatter3d traces for the top and bottom circles of the cylinder
trace_bottom = go.Scatter3d(
    x=x, y=y, z=z_bottom,
    mode='lines',
    line=dict(color='blue', width=2),
    name='bottom circle'
)
trace_top = go.Scatter3d(
    x=x, y=y, z=z_top,
    mode='lines',
    line=dict(color='blue', width=2),
    name='top circle'
)

# Create Scatter3d traces for the vertical lines of the cylinder
traces_vertical = []
for i in range(segments):
    traces_vertical.append(go.Scatter3d(
        x=[x[i], x[i]], y=[y[i], y[i]], z=[z_bottom[i], z_top[i]],
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False
    ))

# Generate a helical line that fits inside the cylinder
theta_line = np.linspace(0, 10*np.pi, 1000)  # make the line spiral around 5 times
x_line = center[0] + (radius * 1) * np.cos(theta_line)  # make the line a bit smaller than the cylinder
y_line = center[1] + (radius * 1) * np.sin(theta_line)
z_line = np.linspace(center[2] - height/2, center[2] + height/2, len(theta_line))

# Create a Scatter3d trace for the line
trace_line = go.Scatter3d(
    x=x_line, y=y_line, z=z_line,
    mode='lines',
    line=dict(color='red', width=25),
    name='line'
)

# Create the 3D plot
fig = go.Figure(data=[trace_bottom, trace_top, *traces_vertical, trace_line])
fig.show()
