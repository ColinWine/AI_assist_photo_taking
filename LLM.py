from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModel


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

class InternVL2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
                    self.path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True).eval().cuda()

        self.generation_config = dict(
                                num_beams=1,
                                max_new_tokens=1024,
                                do_sample=False,
                            )
        print('================ Model loaded ================')

    def chat(self, question: str, pixel_values, history: List[dict]) -> str:
        #response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
        if pixel_values is not None:
            print(pixel_values.shape)
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, history=history, return_history=True)
        return response, history
    
if __name__ == '__main__':
    model = InternVL2Chat('/home/colin/projects/assist_photo_taking/src/InternVL-main/InternVL2-8B')
    print(model.chat('Hello', None, []))