cd ~/Desktop/RF-Simulation
cd ~/Desktop/super-fans
source ~/venv/bin/activate
python3 Display3d.py


import os
import time
import panel as pn
import pyvista as pv
import numpy as np
import matplotlib.colors as mcolors
from stl import mesh
from scipy.stats import multivariate_normal
import functools


# ✅ Global Variables
stl_data = None
outline = None
walls = None
gas_points = None
transmitters = []  # Store multiple transmitters as dictionaries

MAX_TRANSMITTERS = 5  # Limit to 5 transmitters for performance


# ✅ Load STL File
def load_stl(filename):
    stl_mesh = mesh.Mesh.from_file(filename)
    points = np.array(stl_mesh.vectors.reshape(-1, 3))
    faces = np.hstack([[3, i, i + 1, i + 2] for i in range(0, len(points), 3)])
    return pv.PolyData(points, faces)


# ✅ Intersection Count Function
def count_wall_intersections(tx, point, walls):
    start = np.array(tx)
    end = np.array(point)
    points, intersection_ids = walls.ray_trace(start, end, first_point=False)
    return len(intersection_ids)


# ✅ RF Signal Strength Using Motley-Keenan Model
def motley_keenan_signal(tx, points, walls, frequency, power_dbm=30, wall_loss_db=5):
    """Computes received power using Motley-Keenan Model for one transmitter."""
    distance = np.linalg.norm(points - tx, axis=1) + 0.1
    path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
    wall_losses = np.array([count_wall_intersections(tx, p, walls) * wall_loss_db for p in points])
    received_power_dbm = power_dbm - (path_loss_db + wall_losses)
    return received_power_dbm, distance


# ✅ Combined Signal from Multiple Transmitters
def combined_signal(points):
    """Combine signals from all transmitters using Power Summation (dBm) and compute average distances."""
    if len(transmitters) == 0:
        return np.full(len(points), -100)  # No transmitters, return very low signal

    combined_power_mw = np.zeros(len(points))
    avg_distances = []

    for tx in transmitters:
        tx_position = tx['position']
        tx_power = tx['power']
        tx_frequency = tx['frequency']

        # ✅ Use dynamic frequency
        signal_dbm, distances = motley_keenan_signal(tx_position, points, walls, frequency=tx_frequency, power_dbm=tx_power)

        print(f"📡 Transmitter at {tx_position} | Power: {tx_power} dBm | Frequency: {tx_frequency / 1e9:.2f} GHz")
        
        avg_distances.append(np.mean(distances))

        power_mw = 10 ** (signal_dbm / 10)
        combined_power_mw += power_mw

    combined_signal_dbm = 10 * np.log10(combined_power_mw)
    return combined_signal_dbm


# ✅ Update Plot Based on Transmitters
import uuid  # ✅ Unique identifier for preventing caching

def update_plot():
    """Update the plot and ensure it reflects transmitter removal."""
    global stl_data, outline, walls, gas_points, transmitters

    print("Updating Plot with Transmitters:", transmitters)  # ✅ Debugging step

    if stl_data is None:
        return "<span style='color:red; font-weight:bold;'>⚠️ Please upload an STL file first.</span>"
    if len(transmitters) == 0:
        return "<span style='color:red; font-weight:bold;'>⚠️ No transmitters left. Add a new one.</span>"

    plotter = pv.Plotter()
    plotter.add_mesh(outline, color="black", line_width=1, opacity=0.3)
    plotter.add_mesh(walls, color="gray", opacity=0.2)

    for tx in transmitters:
        print("Adding Transmitter at:", tx['position'])  # ✅ Ensure positions are correct
        plotter.add_mesh(pv.Sphere(radius=1.0, center=tx['position']), color="red", label="5G Transmitter")

    # ✅ Compute signal strength at all gas points
    signal_strength = combined_signal(gas_points)
    
    # ✅ Debugging signal range before filtering
    print(f"📡 Signal Strength Range Before Filtering: Min={np.min(signal_strength):.2f} dBm, Max={np.max(signal_strength):.2f} dBm")

    # ✅ Adaptive thresholding: Ensures weak signals are visible
    threshold = max(np.min(signal_strength) + 5, -50)  # Keep at least weak signals visible

    # ✅ Apply threshold: Remove weak signals
    valid_indices = signal_strength >= threshold
    gas_points_filtered = gas_points[valid_indices]
    signal_strength_filtered = signal_strength[valid_indices]

    print(f"✅ Filtering complete. {len(gas_points) - len(gas_points_filtered)} points removed due to weak signal.")

    if len(gas_points_filtered) == 0:
        return "<span style='color:red; font-weight:bold;'>⚠️ No areas meet the signal threshold.</span>"

    # ✅ Normalize signal for coloring
    max_value = np.max(signal_strength_filtered)
    min_value = np.min(signal_strength_filtered)
    norm_signal = np.clip((signal_strength_filtered - min_value) / (max_value - min_value), 0, 1)

    sizes = norm_signal * 0.3 + 0.05
    opacity = norm_signal * 0.5 + 0.2

    # ✅ Define custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "RealHeatMap", ["darkblue", "blue", "cyan", "lime", "yellow", "orange", "red"]
    )

    # ✅ Limit the number of points to render (for performance)
    max_points = min(len(gas_points_filtered), 3000)
    selected_indices = np.random.choice(len(gas_points_filtered), max_points, replace=False)

    for idx in selected_indices:
        sphere = pv.Sphere(radius=sizes[idx], center=gas_points_filtered[idx])
        plotter.add_mesh(sphere, color=cmap(norm_signal[idx]), opacity=opacity[idx])

    plotter.show_axes()
    plotter.enable_3_lights()
    plotter.enable_parallel_projection()

    # ✅ Save the updated visualization
    html_dir = os.path.join(os.path.dirname(__file__), "custom_static")
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, "signal_visualization.html")
    plotter.export_html(html_path)

    unique_id = uuid.uuid4()

    return f'''
<div style="display: flex; flex-direction: row; justify-content: center; margin-top: 20px; position: relative;">
  
  <!-- 📡 Heatmap iframe -->
  <iframe id="signal_frame" src="custom_static/signal_visualization.html?{unique_id}" 
          width="1200px" height="800px" 
          style="border: 2px solid #ccc; border-radius: 10px;" allowfullscreen></iframe>

  <!-- 🎨 Legend next to iframe -->
  <div style="margin-left: 20px; display: flex; flex-direction: column; align-items: center;">
    <div style="font-weight: bold; margin-bottom: 5px;">Signal Strength (dBm)</div>
    <div style="position: relative; height: 500px; width: 30px; 
                background: linear-gradient(to top, darkblue, blue, cyan, lime, yellow, orange, red); 
                border: 1px solid black; border-radius: 5px;">
      <div style="position: absolute; top: 0; left: 35px;">-10</div>
      <div style="position: absolute; top: 25%; left: 35px;">-30</div>
      <div style="position: absolute; top: 50%; left: 35px;">-50</div>
      <div style="position: absolute; top: 75%; left: 35px;">-70</div>
      <div style="position: absolute; bottom: 0; left: 35px;">-90</div>
    </div>
  </div>

</div>

<!-- ✅ Refresh iframe -->
<script>
  setTimeout(() => {{
    document.getElementById("signal_frame").src = "custom_static/signal_visualization.html?{unique_id}";
  }}, 500);
</script>
'''


# ✅ Handle File Upload
file_input = pn.widgets.FileInput(name="Choose STL File", accept=".stl")

def handle_file_upload(event):
    """Load the STL file and initialize measurement points."""
    global stl_data, outline, walls, gas_points, transmitters

    if event.new:
        filename = os.path.join(os.getcwd(), "uploaded_file.stl")
        with open(filename, "wb") as f:
            f.write(event.new)

        stl_data = load_stl(filename)
        outline = stl_data.extract_feature_edges()
        walls = stl_data.extract_geometry().triangulate()

        num_points = 3000
        x = np.random.uniform(stl_data.bounds[0], stl_data.bounds[1], num_points)
        y = np.random.uniform(stl_data.bounds[2], stl_data.bounds[3], num_points)
        z = np.random.uniform(stl_data.bounds[4], stl_data.bounds[5], num_points)
        gas_points = np.column_stack((x, y, z))

        # Reset sliders
        slider_x.start, slider_x.end = stl_data.bounds[0], stl_data.bounds[1]
        slider_y.start, slider_y.end = stl_data.bounds[2], stl_data.bounds[3]
        slider_z.start, slider_z.end = stl_data.bounds[4], stl_data.bounds[5]

        # Clear transmitters
        transmitters.clear()

        result.object = "✅ STL file uploaded successfully. Add transmitters and click 'Show Signal Simulation'."


file_input.param.watch(handle_file_upload, 'value')


# ✅ Transmitter Management
def add_transmitter(event):
    """Add a new transmitter without triggering simulation."""
    if len(transmitters) < MAX_TRANSMITTERS:
        tx_position = [slider_x.value, slider_y.value, slider_z.value]
        tx_power = slider_power.value
        tx_frequency = slider_frequency.value * 1e9  # GHz to Hz

        transmitters.append({
            "position": tx_position,
            "power": tx_power,
            "frequency": tx_frequency
        })
        update_transmitter_table()
        result.object = "<span style='color:green;'>✅ Transmitter added. Click 'Show Signal Simulation' to update heatmap.</span>"



def remove_transmitter(index, event=None):
    """Remove a transmitter without refreshing the heatmap."""
    if 0 <= index < len(transmitters):
        del transmitters[index]
        update_transmitter_table()
        result.object = "<span style='color:orange;'>⚠️ Transmitter removed. Click 'Show Signal Simulation' to update heatmap.</span>"


def update_position(event, axis, idx):
    """Update transmitter position and refresh the heatmap in real-time."""
    transmitters[idx]['position'][axis] = event.new
    result.object = update_plot()


def update_transmitter_table():
    """Update the transmitter table with real-time controls (no auto heatmap refresh)."""
    transmitter_table.clear()

    transmitter_rows = []

    for i, tx in enumerate(transmitters):
        # Create sliders
        power_slider = pn.widgets.FloatSlider(
            name=f"Power {i+1} (dBm)", start=10, end=50, step=1, value=tx["power"], width=150
        )

        frequency_slider = pn.widgets.FloatSlider(
            name=f"Frequency {i+1} (GHz)", start=2.0, end=6.0, step=0.1, value=tx["frequency"] / 1e9, width=150
        )

        position_sliders = [
            pn.widgets.FloatSlider(name="X Position", start=0, end=10, step=0.1, value=tx['position'][0], width=100),
            pn.widgets.FloatSlider(name="Y Position", start=0, end=10, step=0.1, value=tx['position'][1], width=100),
            pn.widgets.FloatSlider(name="Z Position", start=0, end=10, step=0.1, value=tx['position'][2], width=100),
        ]

        # Define update callbacks INSIDE the loop so sliders are in scope
        def update_power(event, idx=i):
            transmitters[idx]["power"] = event.new
            result.object = "<span style='color:blue;'>ℹ️ Power updated. Click 'Show Signal Simulation' to refresh heatmap.</span>"

        def update_frequency(event, idx=i):
            transmitters[idx]["frequency"] = event.new * 1e9
            result.object = "<span style='color:blue;'>ℹ️ Frequency updated. Click 'Show Signal Simulation' to refresh heatmap.</span>"

        def update_position(event, axis, idx=i):
            transmitters[idx]['position'][axis] = event.new
            result.object = "<span style='color:blue;'>ℹ️ Position updated. Click 'Show Signal Simulation' to refresh heatmap.</span>"

        # Watch for slider changes
        power_slider.param.watch(update_power, "value")
        frequency_slider.param.watch(update_frequency, "value")

        for axis_idx, slider in enumerate(position_sliders):
            slider.param.watch(lambda event, axis=axis_idx: update_position(event, axis, i), "value")

        remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=80)
        remove_btn.on_click(lambda event, idx=i: remove_transmitter(idx))

        transmitter_rows.append(
            pn.Row(
                f"Transmitter {i+1}",
                power_slider,
                frequency_slider,
                *position_sliders,
                remove_btn
            )
        )

    transmitter_table.extend(transmitter_rows)

def simulate_signal(event):
    global stl_data, transmitters

    if stl_data is not None and len(transmitters) > 0:
        result.object = update_plot()

        # ✅ Create the summary HTML
        summary_html = "<h4 style='color:green;'>✅ Heatmap updated!</h4><ul>"
        for i, tx in enumerate(transmitters):
            summary_html += (
                f"<li><b>Transmitter {i+1}</b>: "
                f"Position = {np.round(tx['position'], 2).tolist()}, "
                f"Power = {tx['power']} dBm, "
                f"Frequency = {tx['frequency']/1e9:.2f} GHz</li>"
            )
        summary_html += "</ul>"

        # ✅ Create Close button
        close_btn = pn.widgets.Button(name="Close", button_type="primary", width=100)

        # ✅ Attach callback BEFORE appending to modal
        def close_modal(event):
            modal_popup.visible = False

        close_btn.on_click(close_modal)

        # ✅ Clear and show the modal
        modal_popup.clear()
        modal_popup.append(pn.pane.HTML(summary_html))
        modal_popup.append(pn.Row(close_btn, align='center', margin=(10, 0, 0, 0)))
        modal_popup.visible = True

    else:
        result.object = "<span style='color:red;'>⚠️ Please upload an STL file and add transmitters.</span>"
        modal_popup.visible = False






simulate_button = pn.widgets.Button(name="Show Signal Simulation", button_type="primary")
simulate_button.on_click(simulate_signal)



# ✅ Panel Configuration
pn.config.static_dirs = {'custom_static': os.path.abspath('custom_static')}

# ✅ Sliders
pn.extension('vtk', 'plotly')

slider_x = pn.widgets.FloatSlider(name='X Position', start=0, end=10, step=0.1)
slider_y = pn.widgets.FloatSlider(name='Y Position', start=0, end=10, step=0.1)
slider_z = pn.widgets.FloatSlider(name='Z Position', start=0, end=10, step=0.1)

# ✅ Define Power Selection Slider (Before It Is Used!)
slider_power = pn.widgets.FloatSlider(name="Transmission Power (dBm)", start=10, end=50, step=1, value=30)

add_button = pn.widgets.Button(name="Add Transmitter range", button_type="success")
result = pn.pane.HTML()
modal_popup = pn.Column(visible=False, styles={
    'position': 'fixed',
    'top': '50%',
    'left': '50%',
    'transform': 'translate(-50%, -50%)',
    'z-index': '1000',
    'background': 'white',
    'border': '2px solid green',
    'padding': '20px',
    'border-radius': '10px',
    'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.2)',
})
transmitter_table = pn.Column()

add_button.on_click(add_transmitter)
# ✅ Frequency Input Slider (GHz)
slider_frequency = pn.widgets.FloatSlider(name="Frequency (GHz)", start=2.0, end=6.0, step=0.1, value=3.5)

# ✅ Dashboard Layout (After Defining slider_power)
dashboard = pn.Column(
    "# 📡 Interactive 5G Signal Simulation (Multiple Transmitters)",
    "### Upload your STL file, add multiple transmitters with custom power and frequency, and visualize the heatmap.",
    file_input,
    slider_x,
    slider_y,
    slider_z,
    slider_power,
    slider_frequency,
    add_button,
    transmitter_table,
    simulate_button,
    result,
    modal_popup  # ✅ Add this here
)



# ✅ Serve Panel Server
if __name__ == "__main__":
    html_dir = os.path.join(os.path.dirname(__file__), "custom_static")
    os.makedirs(html_dir, exist_ok=True)
    pn.serve(dashboard, port=58506, show=True, static_dirs={'custom_static': html_dir})