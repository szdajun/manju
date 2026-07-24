@echo off
REM ep05 kid_dance LoRA v1 launcher (2026-07-24).
REM Target: SDXL + cartoon child dance character, 14 image dataset.
REM Output: ../ComfyUI/models/loras/kid_dance_lora{,-step*}.safetensors
REM
REM Windows kohya 已知坑:
REM   --max_data_loader_n_workers 0 避 spawn 子进程 MemoryError
REM   (diffusers/autoencoders/vae.py OOM on import, 2026-07-23 HANDOFF 撞过)
REM
REM SDXL 关键点 (12GB VRAM):
REM   - 必须用 sdxl_train_network.py (train_network.py 不识 SDXL 双文本编码器)
REM   - --full_bf16 让 text encoder 2 走 bf16 (~850MB) 而非 fp32 (~1.7GB),
REM     否则 PyTorch CPU caching allocator 碎片致 6.5MB 都给不出 (12GB 实测 OOM)
REM   - 不开 --cache_text_encoder_outputs (无 --lowram 时 text encoder 留 CPU, 12GB 余量够)
REM   - HF online 一次, 拉 laion/CLIP-ViT-bigG-14 tokenizer (SDXL text encoder 2)
REM   - python.exe 直调 (kohya venv accelerate.exe 缺 psutil)
REM
REM Usage: just run.

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

REM CPU 分配器: 避 PyTorch CPU caching allocator 碎片 (Windows 不支持, 但不报错)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set TRAIN_DATA=../manju/assets/characters/kid_dance/train_data
set CKPT_DIR=../manju/output/ep05_lora_v1/checkpoints
set COMFY_LORAS=../ComfyUI/models/loras
set OUTPUT_NAME=kid_dance_lora
set LOG_DIR=../manju/output/ep05_lora_v1/train_logs

if not exist %TRAIN_DATA%/10_kid_dance (
  echo Error: dataset not found at %TRAIN_DATA%/10_kid_dance
  echo Run scripts/ep05/build_character.py first.
  exit /b 1
)
if not exist %CKPT_DIR% mkdir %CKPT_DIR%
if not exist %LOG_DIR% mkdir %LOG_DIR%

echo [Training kid_dance LoRA v1] 14 images (cartoon child dance, SDXL)
echo [Dataset] %TRAIN_DATA%
echo [Output]  %CKPT_DIR%
echo [Mirror]  %COMFY_LORAS%\%OUTPUT_NAME%.safetensors

REM 沿用 lihui launcher 范式: 走 kohya venv 的 python 直调
"..\kohya_ss\.venv\Scripts\python.exe" ^
  "..\kohya_ss\sd-scripts\sdxl_train_network.py" ^
  --pretrained_model_name_or_path "../ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors" ^
  --train_data_dir "%TRAIN_DATA%" ^
  --output_dir "%CKPT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --network_module networks.lora ^
  --network_dim 32 ^
  --network_alpha 16 ^
  --resolution 512 ^
  --train_batch_size 1 ^
  --max_train_steps 1200 ^
  --learning_rate 1e-4 ^
  --unet_lr 1e-4 ^
  --text_encoder_lr 1e-5 ^
  --lr_scheduler cosine ^
  --lr_scheduler_num_cycles 1 ^
  --optimizer_type AdamW8bit ^
  --mixed_precision bf16 ^
  --full_bf16 ^
  --gradient_checkpointing ^
  --network_train_unet_only ^
  --sdpa ^
  --save_precision bf16 ^
  --save_every_n_steps 400 ^
  --caption_extension .txt ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --max_data_loader_n_workers 0 ^
  --seed 42 ^
  --logging_dir "%LOG_DIR%"

REM Mirror final step -> ComfyUI/models/loras (T3 LoraLoader 喂这个文件名)
if exist "%CKPT_DIR%\%OUTPUT_NAME%-step001200.safetensors" (
  copy /Y "%CKPT_DIR%\%OUTPUT_NAME%-step001200.safetensors" "%COMFY_LORAS%\%OUTPUT_NAME%.safetensors"
  echo [Mirror OK] %COMFY_LORAS%\%OUTPUT_NAME%.safetensors
) else (
  echo [WARN] step001200 missing, manual mirror needed
)