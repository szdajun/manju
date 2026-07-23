@echo off
REM Lihui_real (ep01 男角) LoRA v1 launcher (2026-07-24).
REM Output: ../ComfyUI/models/loras/lihui_real_lora{,-step*}.safetensors
REM Usage: just run.

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

set TRAIN_DATA=../manju/output/lihui_real_lora/train_data
set CKPT_DIR=../ComfyUI/models/loras
set OUTPUT_NAME=lihui_real_lora
set LOG_DIR=../manju/output/lihui_real_lora/train_logs

if not exist %TRAIN_DATA%/10_lihui_real (
  echo Error: dataset not found at %TRAIN_DATA%/10_lihui_real
  echo Run scripts/aug_lihui_real_dataset.py first.
  exit /b 1
)

echo [Training lihui_real LoRA v1] 7 inputs x 7 variants = 49 images
echo [Dataset] %TRAIN_DATA%

"..\kohya_ss\.venv\Scripts\python.exe" ^
  "..\kohya_ss\sd-scripts\train_network.py" ^
  --pretrained_model_name_or_path "../ComfyUI/models/checkpoints/DreamShaper_8_pruned.safetensors" ^
  --train_data_dir "%TRAIN_DATA%" ^
  --output_dir "%CKPT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --network_module networks.lora ^
  --network_dim 8 ^
  --network_alpha 4 ^
  --resolution 384 ^
  --train_batch_size 1 ^
  --max_train_steps 1200 ^
  --learning_rate 1e-4 ^
  --unet_lr 1e-4 ^
  --text_encoder_lr 1e-5 ^
  --lr_scheduler cosine ^
  --lr_scheduler_num_cycles 1 ^
  --optimizer_type AdamW8bit ^
  --mixed_precision fp16 ^
  --gradient_checkpointing ^
  --network_train_unet_only ^
  --sdpa ^
  --save_precision fp16 ^
  --save_every_n_steps 400 ^
  --caption_extension .txt ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --max_data_loader_n_workers 0 ^
  --seed 42 ^
  --logging_dir "%LOG_DIR%"
