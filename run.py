from flask import Flask, render_template, request, redirect, url_for
import os
from pso import pso_optimization
from ga import ga_optimization  # 유전 알고리즘 추가
from image import extract_wall_and_internal_coordinates
from PIL import Image, ImageDraw
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

    # Get AP count and selected algorithm from the form
    ap_count = int(request.form.get('ap_count', 3))
    algorithm = request.form.get('algorithm', 'pso')  # 'pso' or 'ga'

    # Extract internal coordinates (room layout) from the image
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    # Run the selected algorithm (PSO or GA)
    if algorithm == 'pso':
        best_positions = pso_optimization(internal_coords, num_routers=ap_count, frequency=2400, coverage_radius=50)
    elif algorithm == 'ga':
        best_positions = ga_optimization(internal_coords, num_routers=ap_count, frequency=2400, coverage_radius=50)

    # Generate the output image with optimal AP placements
    output_image_path = generate_output_image(internal_coords, best_positions, ap_count)

    # Pass the relative output path for rendering
    relative_path = os.path.relpath(output_image_path, 'static')
    return render_template('result.html', output_image=relative_path)

def generate_output_image(internal_coords, best_positions, ap_count):
    """
    Generate an output image showing the optimal router placements.
    """
    # Create a blank image with a white background
    img = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(img)

    # Draw the internal coordinates (representing the room layout)
    for coord in internal_coords:
        draw.point(coord, fill='gray')

    # Draw the best AP positions
    for router in best_positions:
        draw.point(router, fill='red')

        # Draw coverage circles (just for illustration)
        coverage_radius = 50
        # For the coverage circle, approximate a rough circle with points
        for angle in range(0, 360, 10):
            x_offset = int(coverage_radius * np.cos(np.radians(angle)))
            y_offset = int(coverage_radius * np.sin(np.radians(angle)))
            draw.point((router[0] + x_offset, router[1] + y_offset), fill='blue')

    # Save the output image
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'optimized_ap_placement.png')
    img.save(output_image_path)

    return output_image_path

if __name__ == '__main__':
    app.run(debug=True)
