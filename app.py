import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.01,model_format="awq")
# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1_8b-4bit'
os.system(f'git clone https://code.openxlab.org.cn/volta/internlm2-chat-1_8b-4bit.git {base_path}')

os.system(f'cd {base_path} && git lfs pull')

# tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

# def chat(message,history):
#     for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
#         yield response

# gr.ChatInterface(chat,
#                  title="xtuner_interml_1.8b",
#                 description="""
# 微调1.8b模型
#                  """,
#                  ).queue(1).launch()
# 初始化 pipeline
pipe = pipeline(
    base_path=base_path,
    backend_config=backend_config
)

# 设置生成配置
gen_config = GenerationConfig(
    top_p=0.8,
    top_k=40,
    temperature=0.8,
    max_new_tokens=1024
)

# 定义聊天功能
def chat(message, history):
    response = pipe(
        message,
        gen_config=gen_config
    )
    return response.text

# 设置 gradio 聊天界面
gr.ChatInterface(
    fn=chat,
    title="internlm2-chat-1_8b-4bit",
    description="""InternLM is mainly developed by Shanghai AI Laboratory."""
).queue(1).launch()
