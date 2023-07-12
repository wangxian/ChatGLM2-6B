from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

import torch, os

CHECKPOINT_PATH = "./output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-300"

# 载入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

# 加载 ptuning 的 checkpoint
config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", config=config, trust_remote_code=True)

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
print(prefix_state_dict)

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.half().cuda()

model = model.eval()

response, history = model.chat(tokenizer, "你好，睡眠不好怎么办？", history=[])

print(response)
