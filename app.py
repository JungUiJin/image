from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# 이미지 분석 모델 로드
model = keras.models.load_model('image_classification_model.h5')

# 모델의 클래스 레이블
class_labels = ['Class 1', 'Class 2', 'Class 3', '...']

# 이미지 결과를 저장할 디렉토리
result_directory = 'static/results'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'result': '이미지를 업로드하세요.'})

    # 이미지를 가져옴
    image = request.files['file'].read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # 모델에 맞는 크기로 조정

    # 이미지를 모델에 입력하기 위해 사전 처리
    image = np.array(image) / 255.0
    image = image.reshape(1, 224, 224, 3)  # 모델에 맞는 형태로 재구성

    # 이미지 분석
    prediction = model.predict(image)

    # 예측 결과에서 가장 높은 확률의 클래스 선택
    predicted_class = class_labels[np.argmax(prediction)]

    # 이미지 저장
    result_image_path = os.path.join(result_directory, 'result_image.jpg')
    image.save(result_image_path)

    return jsonify({'result': predicted_class, 'image_url': result_image_path})

if __name__ == '__main__':
    app.run(debug=True)