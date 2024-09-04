import torch # 导入torch
from transformers import GPT2Tokenizer # 导入GPT2分词器
from transformers import GPT2LMHeadModel # 导入GPT2语言模型
model_name = "gpt2"  # 也可以选择其他模型，如"gpt2-medium"、"gpt2-large"等
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # 加载分词器
device = "cuda" if torch.cuda.is_available() else "cpu" # 判断是否有可用GPU
model = GPT2LMHeadModel.from_pretrained(model_name).to(device) # 将模型加载到设备上（CPU或GPU）
vocab = tokenizer.get_vocab() # 获取词汇表

print("模型信息：", model)
print("分词器信息：",tokenizer)
print("词汇表大小:", len(vocab))
print("部分词汇示例:", (list(vocab.keys())[8000:8005]))