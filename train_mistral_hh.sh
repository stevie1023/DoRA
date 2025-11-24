export MODEL_PATH="/mistralai/Mistral-7B-v0.1"
export SFT_PATH="hh-sft-mistral7b-60k/LATEST"  ####SFT checkpoint path
export Method='dora'
export SAVE_PATH="results"
export DATA_PATH="train_data.json" ####data path, you can customize the data. In our settings, we ensure every line contains a key "score" which is computed by the classifier/
export MASTER_ADDR="localhost"
export MASTER_PORT="23385"


CUDA_VISIBLE_DEVICES=5,6,4,7 \
python3.10 -m torch.distributed.launch  --nproc_per_node=4 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --use_env train_mistral_prompt.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --sft_checkpoint $SFT_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --train_method $Method \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000000 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_total_limit 3 \
    --deepspeed "./default_offload_opt_param.json"\
    --tf32 True --model_max_length 512 --train_sample_num 4  
