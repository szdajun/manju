@echo off
REM Xiaoya character LoRA training launcher (kohya_ss sd-scripts)
REM MemoryError fix (2026-07-23): spawn subprocess diffusers import -> set --max_data_loader_n_workers 0
REM Same args as previously dead run except: --max_data_loader_n_workers 0 (was 2)
REM Resume: steps 0/1200 (no checkpoint saved, run died before step 1)
REM Dataset layout (kohya 期望 class-by-class): train_data/10_xiaoya/*.png (+ .txt caption)

setlocal
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost

"..\kohya_ss\.venv\Scripts\python.exe" ^
  "..\kohya_ss\sd-scripts\train_network.py" ^
  --pretrained_model_name_or_path "../ComfyUI/models/checkpoints/DreamShaper_8_pruned.safetensors" ^
  --train_data_dir "../manju/output/lihui_lora/train_data" ^
  --output_dir "../ComfyUI/models/loras" ^
  --output_name xiaoya_lora ^
  --network_module networks.lora ^
  --network_dim 8 ^
  --network_alpha 4 ^
  --resolution 384 ^
  --train_batch_size 1 ^
  --max_train_steps 1200 ^
  --learning_rate 1e-4 ^
  --unet_lr 1e-4 ^
  --text_encoder_lr 1e-5 ^
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
  --logging_dir "../manju/output/lihui_lora/train_logs"
