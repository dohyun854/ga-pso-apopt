from flask import Flask, render_template, request, redirect, url_for
import os
from pso import pso_optimization
from image import extract_wall_and_internal_coordinates
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Get AP count from the form
    ap_count = int(request.form.get('ap_count', 3))

    # Extract internal coordinates (room layout) from the image
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    # Run PSO optimization for AP placement
    best_positions = pso_optimization(internal_coords, num_routers=ap_count, frequency=2400, coverage_radius=50)

    # Generate the output image with optimal AP placements
    output_image_path = generate_output_image(internal_coords, best_positions, ap_count)

    # Pass the relative output path for rendering
    relative_path = os.path.relpath(output_image_path, 'static')
    return render_template('result.html', output_image=relative_path)

def generate_output_image(internal_coords, best_positions, ap_count):
    """
    Generate an output image showing the optimal router placements.
    """
    fig, ax = plt.subplots()
    
    # Draw the internal coordinates
    x, y = zip(*internal_coords)
    ax.scatter(x, y, s=1, label="Internal Area", color="gray")

    # Draw the best AP positions
    router_x, router_y = zip(*best_positions)
    ax.scatter(router_x, router_y, color="red", label="Router Positions")

    # Draw coverage circles
    coverage_radius = 50
    for router in best_positions:
        circle = plt.Circle(router, coverage_radius, color='blue', alpha=0.3)
        ax.add_artist(circle)

    ax.legend()
    ax.axis('equal')

    # Save the output image
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'optimized_ap_placement.png')
    plt.savefig(output_image_path)
    plt.close()

    return output_image_path

if __name__ == '__main__':
    app.run(debug=True)
