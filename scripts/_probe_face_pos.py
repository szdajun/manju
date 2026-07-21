"""探立绘脸在图的位置 vs IPAdapter center-crop 窗口。
ComfyUI log: 竖图立绘被 CLIPImageProcessor center-crop 成正方形。
若脸偏上(全身/半身竖图通病), crop 切掉脸 → IPAdapter 锁身体不锁脸 → 关键帧脸随机。
解释为何 ep01 李辉(脸可能居中)过, ep02 小雅(脸偏上)挂。"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
P = {
    "xiaoya#1(写实)": "../manju/output/xiaoya_real/小雅立绘写实.png",
    "lihui#3(写实)": "../manju/output/lihui_real/主角立绘写实.png",
}
for tag, p in P.items():
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    fs = app.get(img)
    if not fs:
        print(f"{tag}: {w}x{h} NO FACE")
        continue
    fs.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    f = fs[0]
    bx = [int(x) for x in f.bbox]
    cy = (bx[1] + bx[3]) / 2
    # IPAdapter center-crop: 短边 resize 到 224 再 center-crop 224, 映射回原图 ≈ y[(h-短边)/2 ...]
    # 竖图短边=宽, 等比 crop 高度 ≈ w → y 窗口 [(h-w)//2, (h-w)//2 + w]
    cy0 = (h - w) // 2
    cy1 = cy0 + w
    face_top_in = bx[1] >= cy0
    face_bot_in = bx[3] <= cy1
    fully = face_top_in and face_bot_in
    print(f"{tag}: {w}x{h} face_bbox={bx} cy={cy:.0f}({cy/h*100:.0f}%from top) "
          f"crop_y=[{cy0},{cy1}] face_fully_in_crop={fully}")
