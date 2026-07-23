@echo off
REM Linxia LoRA v3 launcher (2026-07-24).
REM Dataset: 203 images (stage A: 28 source x 7 variants each + canon).
REM   Differs from v2: more data, simpler emphasis (less aggressive),
REM   longer steps (1500) for the larger 203-sample dataset.
REM Output: ../ComfyUI/models/loras/linxia_lora_v3-{stepNNNNNN|final}.safetensors

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

set CHAR=linxia
set TRAIN_DATA=../manju/output/linxia_lora/train_data
set CKPT_DIR=../ComfyUI/models/loras
set OUTPUT_NAME=linxia_lora_v3
set LOG_DIR=../manju/output/linxia_lora/train_logs_v3

echo [Training v3] %CHAR% with 203 images (28 source x 7 variants)
echo [Dataset] %TRAIN_DATA%
echo [Output] %CKPT_DIR%/%OUTPUT_NAME%-step001500.safetensors

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
  --max_train_steps 1500 ^
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
  --save_every_n_steps 500 ^
  --caption_extension .txt ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --max_data_loader_n_workers 0 ^
  --seed 42 ^
  --logging_dir "%LOG_DIR%"
