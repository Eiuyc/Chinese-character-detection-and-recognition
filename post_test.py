import requests as rq
import json, base64, cv2, numpy as np
url = 'http://localhost'
files = {'file':('AI.jpg', open('./static/imgs/ai.jpg', 'rb'), 'image/jpg', {})}
r = rq.post(url, files=files)

d = json.loads(r.text)

print('image_path:', d['image_path'])
img_s = d['image']
img_b = base64.b64decode(img_s)
np_data = np.frombuffer(img_b, np.uint8)
img_cv = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
# cv2.imwrite('a.jpg',img_cv)
print('image shape:', img_cv.shape)

pred = d['prediction']
print(f'{len(pred)} words detected')

for i, p in enumerate(pred):
    print(f'===={i}====')
    print('location:', p[0])
    print('confidence:', p[1])
    print('word:', p[2])
    print('score:', p[3])
    print('=========\n')


