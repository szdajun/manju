@echo off
REM LoRA continue-train general launcher (2026-07-24).
REM   Resumes from step-1200 ckpt to step-2400 using --network_weights.
REM   Cosine-with-restarts LR scheduler 5e-5 -> 5e-6 (finetune stage).
REM
REM Usage:
REM   run_lora_continue_train.bat xiaoya      continue-training xiaoya
REM   run_lora_continue_train.bat linxia      continue-training linxia
REM Output: ../ComfyUI/models/loras/{char}_lora_continue-step002400.safetensors

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

if "%~1"=="" (
  echo Usage: %~nx0 xiaoya^|linxia
  exit /b 1
)

set CHAR=%~1
if not "%CHAR%"=="xiaoya" if not "%CHAR%"=="linxia" (
  echo Error: character must be xiaoya or linxia, got %CHAR%
  exit /b 1
)

set CKPT_DIR=../ComfyUI/models/loras
set MANJU_OUT=../manju/output/%CHAR%_lora
if "%CHAR%"=="xiaoya" set TRAIN_DATA=../manju/output/lihui_lora/train_data
if "%CHAR%"=="linxia" set TRAIN_DATA=../manju/output/linxia_lora/train_data
set NETWORK_WEIGHTS=%CKPT_DIR%/%CHAR%_lora-step00001200.safetensors
set OUTPUT_NAME=%CHAR%_lora_continue
set LOG_DIR=%MANJU_OUT%/train_logs_continue

if not exist %NETWORK_WEIGHTS% (
  echo Error: continue ckpt not found %NETWORK_WEIGHTS%
  echo Need stage-4 step-1200 ckpt first.
  exit /b 1
)

echo [Continue] %CHAR% from step 1200 to 2400
echo [Base weights] %NETWORK_WEIGHTS%
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
  --max_train_steps 2400 ^
  --learning_rate 5e-5 ^
  --unet_lr 5e-5 ^
  --text_encoder_lr 5e-6 ^
  --lr_scheduler cosine_with_restarts ^
  --lr_scheduler_num_cycles 1 ^
  --optimizer_type AdamW8bit ^
  --mixed_precision fp16 ^
  --gradient_checkpointing ^
  --network_train_unet_only ^
  --sdpa ^
  --save_precision fp16 ^
  --save_every_n_steps 200 ^
  --caption_extension .txt ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --max_data_loader_n_workers 0 ^
  --seed 42 ^
  --logging_dir "%LOG_DIR%" ^
  --network_weights "%NETWORK_WEIGHTS%"
