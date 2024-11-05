from flask import Flask, render_template, request, redirect, url_for
from image_processing import process_image_for_ap_placement
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Upload 및 결과 디렉터리 생성
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

    # AP 개수는 폼에서 받음
    ap_count = int(request.form.get('ap_count', 3))

    # 이미지 저장 및 경로 설정
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # 이미지 처리
    output_path = process_image_for_ap_placement(image_path, OUTPUT_FOLDER, ap_count=ap_count)

    return render_template('result.html', output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
