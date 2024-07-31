import requests
import os
from search import (
    clean_cache,
    search_image_by_image,
    search_image_by_text,
)



payload = {
        "positive": "",
        "negative": "",
        "top_n": "6",
        "search_type": 0,
        "positive_threshold": 10,
        "negative_threshold": 10,
        "image_threshold": 85,
        "img_id": -1,
        "path": "",
    }

# 以图搜图
with requests.session() as sess:
    upload_file = '/home/colin/projects/assist_photo_taking/src/InternVL-main/internvl_chat/examples/image1.jpg'
    # 测试上传图片
    files = {'file': ('input.png', open(upload_file, 'rb'), 'image/png')}
    response = sess.post('http://127.0.0.1:8085/api/upload', files=files)
    assert response.status_code == 200
    # 测试以图搜图
    payload["search_type"] = 1
    response = sess.post('http://127.0.0.1:8085/api/match', json=payload)
    results = response.json()

print(results)