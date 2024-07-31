import requests
import os
from search import (
    clean_cache,
    search_image_by_image,
    search_image_by_text,
)


payload = {
        "positive": "red_panda",
        "negative": "",
        "top_n": "6",
        "search_type": 0,
        "positive_threshold": 10,
        "negative_threshold": 10,
        "image_threshold": 85,
        "img_id": -1,
        "path": "",
    }

response = requests.post('http://127.0.0.1:8085/api/match', json=payload)
results = response.json()
#results = search_image_by_text(payload["positive"], payload["negative"], payload['positive_threshold'], payload['negative_threshold'])
print(results)