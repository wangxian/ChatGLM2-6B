set PRE_SEQ_LEN=128
set CUDA_VISIBLE_DEVICES=0 

python web_demo.py ^
    --model_name_or_path THUDM/chatglm2-6b ^
    --ptuning_checkpoint output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-300 ^
    --pre_seq_len %PRE_SEQ_LEN%

