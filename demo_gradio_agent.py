import gradio as gr
import random
import time
from typing import Dict, List, Optional, Tuple, Union
import json5
from PIL import Image, ExifTags
from LLM import InternVL2Chat
from tool import Tools


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

Begin!
"""

class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        self.model = InternVL2Chat(path)
        self.tool = Tools(self.model)
        self.system_prompt = self.build_system_input()

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt
    
    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:') : j].strip()
            plugin_args = text[j + len('\nAction Input:') : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text
    
    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'composition_advice':
            ref_image_path, plugin_response = self.tool.composition_advice(**plugin_args)
            return ref_image_path,'\nObservation:' + plugin_response
        elif plugin_name == 'pose_advice':
            ref_image_path, plugin_response = self.tool.pose_advice(**plugin_args)
            return ref_image_path,'\nObservation:' + plugin_response
        elif plugin_name == 'ISO_advice':
            ref_image_path, plugin_response = self.tool.ISO_advice(**plugin_args)
            return ref_image_path,'\nObservation:' + plugin_response

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        response, his = self.model.chat(self.system_prompt + text, None, history, )
        print(response)

        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            ref_image_path, final_response = self.call_plugin(plugin_name, plugin_args)
            return response,ref_image_path,final_response 
        else:
            final_response , his = self.model.chat(text, None, history)
            return response,None,final_response

def respond(message,chat_history):
    content = ""
    if message["text"] is not None:
        content = content + 'Query question: ' + message["text"] 
    for x in message["files"]:
        content = content + '\nQuery image path: ' + x
    chat_history.append(gr.ChatMessage(role="user", content=content))
    return gr.MultimodalTextbox(value=None, interactive=True), chat_history

def draw_agent(agent_name, response, chat_history):
    chat_history.append(gr.ChatMessage(role="assistant", content=response,
                        metadata={"title": f"🛠️ Used tool {agent_name}"}))
    
    return chat_history

def draw_bot_response(agent_response, ref_image_path, final_response, chat_history):
    chat_history.append(gr.ChatMessage(role="assistant", content=agent_response))
    chat_history.append(gr.ChatMessage(role="assistant", content="I have found a reference photo for you: "+ ref_image_path))
    chat_history.append(gr.ChatMessage(role="assistant", content=final_response))
    return chat_history

def call_agent(chat_history):
    if isinstance(chat_history[-1]['content'],str):
        message = chat_history[-1]
    else:
        message = chat_history[-2]
    agent_response,ref_image_path,final_response = agent.text_completion(message['content'], [])

    return draw_bot_response(agent_response, ref_image_path, final_response, chat_history)
    
agent = Agent('/home/colin/projects/assist_photo_taking/src/InternVL-main/InternVL2-8B')
prompt = agent.build_system_input()
#print(prompt)
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Agent",
                         type="messages",
                         height=768,
                         avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"))
    chat_input = gr.MultimodalTextbox(interactive=True,
                                file_count="multiple",
                                placeholder="Enter message or upload file...", show_label=False)
    #Find a local photo of red panda
    clear = gr.ClearButton([chat_input, chatbot])


    chat_msg = chat_input.submit(respond, [chat_input, chatbot], [chat_input, chatbot])
    bot_msg = chat_msg.then(call_agent, [chatbot], [chatbot], api_name="bot_response")

demo.launch()
