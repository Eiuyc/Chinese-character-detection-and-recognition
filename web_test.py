from flask import Flask, render_template, request
import os, cv2, json, base64, numpy as np

from single_word.cnn_net import Net
from model_ctl import Young

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
def index():
    d=''
    if request.method == 'POST':
        f = request.files['file']
        img_path = f'static/tmp/{f.filename}'
        # f.save(img_path)

        np_data = np.frombuffer(f.read(), np.uint8)
        img_cv = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        cv2.imwrite(img_path,img_cv)
        
        pred, result = model.evaluate(img_cv)
        img_path = f'{img_path}_predict.jpg'
        cv2.imwrite(img_path, result)
        _, img = cv2.imencode('.jpg', result) # ndarray -> image
        img_s = str(base64.b64encode(img), encoding='utf-8')
        d = {
            'prediction': pred,
            'image': img_s,
            'image_path': img_path
        }
    return json.dumps(d)

if __name__ == '__main__':
    model = Young()
    app.run(host='127.0.0.1', port='80', debug=1)
