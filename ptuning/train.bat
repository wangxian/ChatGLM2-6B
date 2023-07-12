@REM @echo off

@REM IF NOT EXIST venv (
@REM python -m venv venv
@REM ) ELSE (
@REM echo venv folder already exists, skipping creation...
@REM )
@REM call .\venv\Scripts\activate.bat

set PRE_SEQ_LEN=128
set LR=2e-2
set NUM_GPUS=1

python main.py ^
    --do_train ^
    --train_file AdvertiseGen/train.json ^
    --validation_file AdvertiseGen/dev.json ^
    --preprocessing_num_workers 10 ^
    --prompt_column content ^
    --response_column summary ^
    --overwrite_cache ^
    --model_name_or_path THUDM/chatglm2-6b ^
    --output_dir output/adgen-chatglm2-6b-pt-%PRE_SEQ_LEN%-%LR% ^
    --overwrite_output_dir ^
    --max_source_length 64 ^
    --max_target_length 128 ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps 300 ^
    --logging_steps 10 ^
    --save_steps 100 ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN% ^
