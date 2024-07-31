import os, json
import requests
from utils import load_image,extract_exif_data
import torch

"""
工具函数

- 首先要在 tools 中添加工具的描述信息
- 然后在 tools 中添加工具的具体实现

- https://serper.dev/dashboard
"""

class Tools:
    def __init__(self,model) -> None:
        self.toolConfig = self._tools()
        self.model = model
    
    def _tools(self):
        tools = [
            {
                'name_for_human': 'composition_advice',
                'name_for_model': 'composition_advice',
                'description_for_model': 'provide composition advice based on reference image',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': 'input query key word, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    },    
                    {
                        'name': 'search_query_image',
                        'description': 'input query image path, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ]
            },
            {
                'name_for_human': 'pose_advice',
                'name_for_model': 'pose_advice',
                'description_for_model': 'provide pose advice based on reference image',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': 'input query key word, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    },    
                    {
                        'name': 'search_query_image',
                        'description': 'input query image path, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human': 'ISO_advice',
                'name_for_model': 'ISO_advice',
                'description_for_model': 'provide ISO setting advice based on reference image',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': 'input query key word, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'search_query_image',
                        'description': 'input query image path, if not exsist, pass blank string',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
        ]

        return tools
    
    def composition_advice(self,search_query,search_query_image):
        if search_query_image != '':
            reference_image_path = self.search_image_by_image(search_query_image)
            pixel_values = load_image(reference_image_path, max_num=6).to(torch.bfloat16).cuda()
        else:
            reference_image_path = self.search_image_by_text(search_query)
            pixel_values = load_image(reference_image_path, max_num=6).to(torch.bfloat16).cuda()
        agent_prompt = '<image>\nYou are an professional photography assistant, based on the reference image, describe the composition techniques used'
        response, his = self.model.chat(agent_prompt, pixel_values, [], )

        print(reference_image_path)
        print('Agent 输出',response)
        print(response)
        return reference_image_path,response
    
    def pose_advice(self,search_query,search_query_image):
        if search_query_image != '':
            reference_image_path = self.search_image_by_image(search_query_image)
            pixel_values = load_image(reference_image_path, max_num=6).to(torch.bfloat16).cuda()
        else:
            reference_image_path = self.search_image_by_text(search_query)
            pixel_values = load_image(reference_image_path, max_num=6).to(torch.bfloat16).cuda()

        agent_prompt = '<image>\nYou are an professional photography assistant, based on the reference image, describe the pose of people in the image, you should including the position of person, Body Orientation, Head and Neck Pose, Hands and Arms Pose, Legs and Feet Pose, Overall Composition. If there is no person in the reference image, provide your idea for posing and do not mention that you did not find person'
        response, his = self.model.chat(agent_prompt, pixel_values, [], )

        print(reference_image_path)
        print('Agent 输出',response)
        return reference_image_path,response
    
    def ISO_advice(self,search_query,search_query_image):
        if search_query_image != '':
            reference_image_path = self.search_image_by_image(search_query_image)
        else:
            reference_image_path = self.search_image_by_text(search_query)

        exif_info = extract_exif_data(reference_image_path)
        if len(exif_info) != 0:
            agent_prompt = '<image>\nYou are an professional photography assistant, based on the reference EXIF information, give explanation on how these camera settings works'
            response, his = self.model.chat(agent_prompt+exif_info, None, [], )
        else:
            pixel_values = load_image(reference_image_path, max_num=6).to(torch.bfloat16).cuda()
            agent_prompt = '<image>\nYou are an professional photography assistant, based on the reference image, give advice on ISO,Exposure time,FNumber and other camera settings '
            response, his = self.model.chat(agent_prompt, pixel_values, [], )
        print(reference_image_path)
        print('Agent 输出',response)
        return reference_image_path,response
    
    def search_image_by_text(self, search_query: str):
        payload = {
        "positive": "",
        "negative": "",
        "top_n": "1",
        "search_type": 0,
        "positive_threshold": 10,
        "negative_threshold": 10,
        "image_threshold": 85,
        "img_id": -1,
        "path": "",
        }

        payload['positive'] = search_query
        response = requests.post('http://127.0.0.1:8085/api/match', json=payload)
        results = response.json()

        return results[0]['path']
    
    def search_image_by_image(self, search_query: str):
        payload = {
        "positive": "",
        "negative": "",
        "top_n": "1",
        "search_type": 0,
        "positive_threshold": 10,
        "negative_threshold": 10,
        "image_threshold": 85,
        "img_id": -1,
        "path": "",
        }

        # 以图搜图
        with requests.session() as sess:
            upload_file = search_query
            # 测试上传图片
            files = {'file': ('input.png', open(upload_file, 'rb'), 'image/png')}
            response = sess.post('http://127.0.0.1:8085/api/upload', files=files)
            assert response.status_code == 200
            # 测试以图搜图
            payload["search_type"] = 1
            response = sess.post('http://127.0.0.1:8085/api/match', json=payload)
            results = response.json()

        return results[0]['path']
