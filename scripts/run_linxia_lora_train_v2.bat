@echo off
REM Linxia LoRA v2 fresh training launcher (2026-07-24).
REM   Differs from run_linxia_lora_train.bat:
REM     - Output name: linxia_lora_v2 (avoid clobber v1)
REM     - Cosine LR with restarts (smoother 1200-step decay than constant 1e-4)
REM     - Caption set has logo_zoom + pants_zoom variants (closeup emphasis)
REM
REM Usage: just run this batch (always trains linxia only, v2 purpose).
REM Output: ../ComfyUI/models/loras/linxia_lora_v2-step{NNNN}.safetensors
REM         + linxia_lora_v2.safetensors (final)

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

set CHAR=linxia
set TRAIN_DATA=../manju/output/linxia_lora/train_data
set CKPT_DIR=../ComfyUI/models/loras
set OUTPUT_NAME=linxia_lora_v2
set LOG_DIR=../manju/output/linxia_lora/train_logs_v2

echo [Training v2] %CHAR% with extended dataset (54 train+val, includes logo_zoom + pants_zoom variants)
echo [Dataset] %TRAIN_DATA%
echo [Output] %CKPT_DIR%/%OUTPUT_NAME%-step00001200.safetensors

if not exist %TRAIN_DATA%/10_linxia (
  echo Error: dataset not found at %TRAIN_DATA%/10_linxia
  echo Run scripts/aug_linxia_dataset.py first.
  exit /b 1
)

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
