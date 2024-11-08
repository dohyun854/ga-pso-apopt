from flask import Flask, render_template, request, redirect, url_for
from image_processing import process_image_for_ap_placement
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload and output directories if they don't exist
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

    # Get AP count from form
    ap_count = int(request.form.get('ap_count', 3))

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Process the image and get the output path
    output_path = process_image_for_ap_placement(image_path, OUTPUT_FOLDER, ap_count=ap_count)

    return render_template('result.html', output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
