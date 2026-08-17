"""Wan2.2-Animate 动作克隆 — 正式档 (参数化, 带进度轮询).

基于已验证的 ViTPose 预处理链 (scripts/_debug_wan_posefix.py 根因修复版):
  LoadVideo → OnnxDetectionModelLoader(yolov10m+vitpose) → PoseAndFaceDetection
            → [pose_data, face_images]
  DrawViTPose(pose_data) → pose_video (ViTPose 黑底骨架, 驱动身体)
  face_images            → face_video (驱动表情)
  WanAnimateToVideo(pose_video, face_video, reference_image) → 原生采样链 → SaveVideo

用法:
  uv run python scripts/gen_wan_animate.py                       # 默认 576×1024
  uv run python scripts/gen_wan_animate.py --width 512 --height 768 --steps 25
  uv run python scripts/gen_wan_animate.py --drive other.mp4 --coach client_a --length 49
"""
import sys, os, time, argparse, glob, traceback, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.comfyui_client import ComfyUIClient

SERVER = "http://127.0.0.1:8188"
COMFY_INPUT = "../ComfyUI/input"
DEFAULT_REF_GLOB = "../manju/assets/characters/client_a/*正面*.png"
FFMPEG = r"~/ffmpeg/ffmpeg.exe"
FFPROBE = r"~/ffmpeg/ffprobe.exe"


def _probe_dims(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path]).decode()
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def prep_zoom_drive(drive_name, W, H, zoom, vshift=0.0, hshift=0.0):
    """驱动视频放大裁切 → scale 回输出尺寸 → 中近景构图 (解决"人物远/脸小", 且不变形).

    根因: Wan-Animate 输出构图由 pose_video 主导 (正面半身参考照也出全身 = 证据),
    所以"拉近"必须裁驱动视频, 不是换参考照. 裁驱动同时让 pose_video (骨架) 和
    face_video (表情裁剪) 一起变近 → 输出变近 → 脸像素预算变大 (face_swap 救不了
    全身 50px 小脸, 见 memory client_k-v2v-pivot-face-budget).

    zoom>1: crop_h = DH/zoom (主体放大). vshift -1..1 纵向移裁框 (负=上移, 给全身
    舞者头顶留 headroom; 全身居中驱动用 -0.5). 按 W/H 裁 (保宽高比) **再 scale 回 W×H**
    (驱动源尺寸必须 = 输出尺寸, 否则骨架坐标被拉伸 → 脸/身材/大腿变形; 见 memory
    wan-animate-framing-pose-dominates 变形根因). 偶数尺寸 (yuv420p). 已存在则复用.
    验证 (448×672): zoom=2.0 vshift=-0.5 → crop=224:336:112:84 → scale 448:672 →
    半身特写/脸大/无变形/无鬼影 (2026-08-11 用户动片终判).
    """
    if zoom <= 1.001:
        return drive_name
    src = os.path.join(COMFY_INPUT, drive_name)
    DW, DH = _probe_dims(src)
    a = W / H
    crop_h = int(round(DH / zoom))
    crop_w = int(round(crop_h * a))
    crop_h -= crop_h % 2
    crop_w -= crop_w % 2
    y = int(round((DH - crop_h) / 2 * (1 + vshift)))
    x = int(round((DW - crop_w) / 2 * (1 + hshift)))
    x = max(0, min(x, DW - crop_w))
    y = max(0, min(y, DH - crop_h))
    # tag zs = zoom+scale (旧 z 版裁后未 scale 会变形, 废弃其缓存)
    tag = f"zs{zoom}_v{vshift}".replace(".", "p").replace("-", "m")
    out_name = f"{os.path.splitext(drive_name)[0]}_{tag}.mp4"
    out_path = os.path.join(COMFY_INPUT, out_name)
    if not os.path.exists(out_path):
        cmd = [FFMPEG, "-y", "-i", src,
               "-vf", f"crop={crop_w}:{crop_h}:{x}:{y},scale={W}:{H}:flags=lanczos",
               "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out_path]
        print(f"[zoom] crop={crop_w}:{crop_h}:{x}:{y} + scale={W}:{H} (zoom={zoom} vshift={vshift}) -> {out_name}")
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        print(f"[zoom] 复用已存在 {out_name}")
    return out_name


def build_workflow(drive_file, ref_image, W, H, LENGTH, FPS, STEPS, SEED, POS, NEG, prefix):
    return {
        "vid":   {"class_type": "LoadVideo", "inputs": {"file": drive_file}},
        "vc":    {"class_type": "GetVideoComponents", "inputs": {"video": ["vid", 0]}},
        "loader":{"class_type": "OnnxDetectionModelLoader", "inputs": {
            "vitpose_model": "vitpose-l-wholebody.onnx",
            "yolo_model": "yolov10m.onnx",
            "onnx_device": "CPUExecutionProvider"}},
        "detect":{"class_type": "PoseAndFaceDetection", "inputs": {
            "model": ["loader", 0], "images": ["vc", 0], "width": W, "height": H}},
        "pose":  {"class_type": "DrawViTPose", "inputs": {
            "pose_data": ["detect", 0], "width": W, "height": H,
            "body_stick_width": -1, "hand_stick_width": -1, "draw_head": True, "retarget_padding": 0}},
        "ref":   {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "unet":  {"class_type": "UnetLoaderGGUFDynamicVRAM", "inputs": {"unet_name": "Wan2.2-Animate-14B-Q4_K_M.gguf"}},
        "clip":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "clipv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}},
        "vae":   {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "pos":   {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": POS}},
        "neg":   {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEG}},
        "cve":   {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["clipv", 0], "image": ["ref", 0], "crop": "center"}},
        "wa":    {"class_type": "WanAnimateToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "width": W, "height": H, "length": LENGTH, "batch_size": 1,
            "continue_motion_max_frames": 1, "video_frame_offset": 0,
            "clip_vision_output": ["cve", 0], "reference_image": ["ref", 0],
            "face_video": ["detect", 1], "pose_video": ["pose", 0]}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["unet", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
        "guider":{"class_type": "BasicGuider", "inputs": {"model": ["unet", 0], "conditioning": ["wa", 0]}},
        "samp":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "sampler":{"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["samp", 0],
            "sigmas": ["sched", 0], "latent_image": ["wa", 2]}},
        "dec":   {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "cv":    {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": FPS}},
        # ComfyUI 0.30.0 SaveVideo.codec = COMFY_DYNAMICCOMBO_V3.
        # build_nested_inputs 把 live 值包成 {"codec": <livevalue>}; execute 读 codec["codec"].
        # 所以 live 值 = 选项 key 字符串 "h264" (不是 dict).
        "save":  {"class_type": "SaveVideo", "inputs": {"video": ["cv", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--length", type=int, default=33, help="输出帧数 (16fps 时 33≈2s)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drive", default="drive_dance.mp4", help="ComfyUI/input 下驱动视频文件名")
    ap.add_argument("--ref", default=None, help="参考照本地路径 (缺省 glob client_a正面)")
    ap.add_argument("--coach", default="client_a")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", default="../manju/output/wan_animate")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="驱动视频放大裁切 >1=拉近 (如 1.4 中近景). pose 主导构图, 故裁驱动非参考")
    ap.add_argument("--vshift", type=float, default=0.0,
                    help="裁框纵向偏移 -1..1 (负=上移留 headroom; 全身居中驱动用 -0.5)")
    ap.add_argument("--hshift", type=float, default=0.0, help="裁框横向偏移 -1..1")
    args = ap.parse_args()

    client = ComfyUIClient(SERVER)

    ref_local = args.ref
    if not ref_local:
        refs = sorted(glob.glob(f"../manju/assets/characters/{args.coach}/*正面*.png"))
        if not refs:
            print(f"ERROR: 找不到参考照 ({args.coach} 正面)", file=sys.stderr); sys.exit(2)
        ref_local = refs[-1]
    ref_server = client.upload_image(ref_local)
    prefix = args.prefix or f"wan_{args.coach}_{args.width}x{args.height}"
    if args.zoom > 1.001 and not args.prefix:
        prefix += f"_z{args.zoom}"

    POS = (f"a Chinese woman in her 30s with shoulder-length black hair, wearing a form-fitting emerald-green "
           f"sports bra crop top and very short black athletic shorts, bare toned midriff and bare athletic legs, "
           f"performing smooth fitness dance moves, graceful flowing arm movements, rhythmic stepping, "
           f"photorealistic, high detail")
    NEG = "distorted, deformed, blurry, low quality, bad anatomy"

    drive_name = prep_zoom_drive(args.drive, args.width, args.height, args.zoom, args.vshift, args.hshift)
    wf = build_workflow(drive_name, ref_server, args.width, args.height, args.length,
                        args.fps, args.steps, args.seed, POS, NEG, prefix)

    print(f"[提交] {args.width}×{args.height} {args.length}帧 {args.steps}步 seed={args.seed} zoom={args.zoom} drive={drive_name} ref={os.path.basename(ref_local)}")
    pid = client.queue_prompt(wf)
    print(f"[prompt_id] {pid}")

    # 带进度轮询的等待 (每 15s 打印队列/历史状态)
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            entry = client.get_history(pid)
        except Exception as e:
            print(f"[warn] get_history 失败: {e} (可能服务在跑/OOM, 继续 poll)")
            time.sleep(15); continue
        if entry is not None:
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                msgs = (entry.get("status") or {}).get("messages", [])
                print(f"[FAIL] ComfyUI error status. messages: {str(msgs)[:1500]}", file=sys.stderr)
                sys.exit(3)
            saved = client.fetch_outputs(entry, args.out)
            dt = time.time() - t0
            print(f"[完成] {dt:.0f}s -> {saved}")
            for s in saved:
                print(f"OUTPUT\t{s}")
            return
        # 还在跑
        try:
            q = client._get_json("/queue")
            r = len(q.get("queue_running", [])); p = len(q.get("queue_pending", []))
            print(f"[运行 {time.time()-t0:.0f}s] running={r} pending={p}")
        except Exception:
            print(f"[运行 {time.time()-t0:.0f}s] (队列查询失败)")
        time.sleep(15)
    print(f"[TIMEOUT] {args.timeout}s 未完成 (可能 OOM 卡死)", file=sys.stderr); sys.exit(4)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
