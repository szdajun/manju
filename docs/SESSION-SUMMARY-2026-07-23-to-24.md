# 📋 Session 工作总结 (2026-07-23 ~ 2026-07-24)

**项目**: manju (AI 漫剧生成 pipeline)
**分支**: `ep01-wan-photoreal`
**会话跨度**: ~5小时 (含两次重启)
**最终状态**: clean working tree, 36/36 tests pass, 10 commits 落地

---

## 🎯 本会话主要成就

### 1. 双角色 LoRA 流水线搭建 (ep04)
- 完成小雅 / 林夏**双角色** LoRA 训练器 (有触发词 trigger word + 数据集 + launcher)
- 修了**3 类独立坑**(见下"挖到的坑"小节)
- 9 个 graph 结构测试 + 36/36 全套测试无 regression

### 2. LoRA 三次迭代 (小雅 / 林夏 一共走 6 次训练, 林夏 3 次)
| 版本 | 数据 | 步数 | 终态 loss | face_sim 均 | 主要变化 |
|------|------|------|----------|-------------|----------|
| v1 小雅 | 80 | 1200 | 0.086 | - | 起点 |
| v1 小雅续训 | 80→1200 | 2400 | **0.068** | - | 显著精调 |
| v1 林夏 | 40 | 1200 | 0.049 | 0.426 | ep03 起点 |
| v1 林夏续训 | 80→1200 | 2400 | 0.052 | 0.379 | 精调(但回退) |
| v2 林夏 | 54 (+2 特写) | 1200 | 0.058 | 0.358 | 加 emphasis + NEG |
| **v3 林夏** | **203** | **1500** | **0.066** | **0.319** | 最佳平衡 |

### 3. ep03 v3 LoRA 实战出片 (最大成就 🎬)
- ✅ `output/ep03_assemble/ep03_real_v3.mp4` 5.32MB / 23.62s
- ✅ **灰侧线 0/5 → 5/5 出现**(用户剧本钉死特征治本成功)
- ✅ 上衣雾蓝一致 / 裤色深灰统一
- ⚠️ 白弧 logo 仍 0/5 (SD1.5 + 1.5k 步上限,治本需换 XL)

### 4. 李辉 (ep01 男角) LoRA v1 训满
- 49 张数据集 (7 原图 × 7 变体), 1200 步, loss 0.077
- **但 caption 错**: 写了 "gray shirt + black tie",剧本是 "light blue plaid shirt + blue lanyard"
- **需 v2 重训修,但已 commit 代码就绪**

### 5. Inpaint 治本探索 (失败但有结论)
- 在 SD1.5 上,**denoise 0.5 也无法精准只改 mask 区**(总是全身重画)
- 拍了 v3 实战出片走 LoRA 量压制路径,**不再押 inpaint 治本**
- ❌ XL 模型 / PowerPaint / Fooocus Inpaint 本地缺, HF 远程 401 卸不了

---

## 🔧 挖到的坑(已修)

### 1. kohya MemoryError on Windows spawn
- **症状**: spawn 子进程在 `diffusers/autoencoders/vae.py:24` import 时 MemoryError
- **根因**: Windows spawn fork 进程 + diffusers 大 import 链 + ComfyUI 占 RAM
- **修法**: `--max_data_loader_n_workers 0` (主进程同步加载)
- **固化**: 已写入 `run_linxia_lora_train*.bat` launchers

### 2. kohya class folder 结构预期
- **症状**: "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
- **根因**: kohya 要求 `--train_data_dir` 是**父目录**,图片必须在 `train_data/10_xxx/`
- **修法**: 改 launcher 传父目录路径

### 3. ComfyUI 节点真名 + LoadImage filename-only
- **症状 1**: HTTP 400 `Node 'Load LoRA' not found`
- **根因 1**: 真实类名是 `LoraLoader`,不是 `Load LoRA`(后者是 AnimateDiff custom node)
- **症状 2**: HTTP 400 `Invalid image file: output/ep03_lihui/linxia_real_face.png`
- **根因 2**: `LoadImage` 节点期望**文件名**(从 `ComfyUI/input/` 找),不是相对 manju 根的路径
- **修法**: `LoraLoader` 真名 + `REFS_BY_NAME` 改为纯文件名

### 4. cmd batch 中文注释 GBK 解析坑
- **症状**: `'重置' 不是内部或外部命令` + Git Bash 把 UTF-8 batch 字符当 token 切
- **根因**: PowerShell 5.1 按 GBK 读 .bat, 中文注释乱码;Git Bash 中转解码错
- **修法**: **所有 batch 文件用纯 ASCII 注释**(`run_lora_continue_train.bat` 重写时)

### 5. setLatentNoiseMask 字段名错
- **症状**: HTTP 400 `Required input is missing: mask`
- **根因**: 字段是 `mask`,我接成 `noise_mask`
- **修法**: 字段名校正

### 6. Win ComfyUI 启动入口
- `start.py` 是**数字人 pipeline 菜单 wrapper**(等用户输入),不是 ComfyUI 服务
- 直接用 `main.py --listen 127.0.0.1 --port 8188` (绕过)

### 7. SEEDANCE 等 4 容器 keep-awake 防睡眠
- 长训练中 Win 睡眠掉线 → 写 `keep_awake_90min.ps1` 用 SetThreadExecutionState,90min 自动释放

---

## 📂 关键文件清单 (本会话新加)

| 文件 | 用途 |
|------|------|
| `lib/ep04_shots.py` | 双角色 (xiaoya/linxia) 5 镜剧本 |
| `scripts/gen_ep04_lora_ablation.py` | A/B/C 关键帧 ablation |
| `scripts/gen_ep04_compare_2400_vs_1200.py` | 1200 vs 2400 step 比较 contact sheet |
| `scripts/gen_ep03_real_lora_kf.py` | ep03 LoRA 版关键帧 (5 镜, 任意 LoRA) |
| `scripts/make_ep04_lora_report.py` | contact sheet + face_sim 矩阵 + per-region HSV |
| `scripts/aug_linxia_dataset.py` | 林夏 LoRA 数据集 (8 原图 → 203 张) |
| `scripts/aug_lihui_real_dataset.py` | 李辉男角 LoRA 数据集 (7 原图 → 49 张) |
| `scripts/probe_inpaint_logo.py` | ComfyUI inpaint 节点试点 |
| `scripts/batch_inpaint_pants.py` | 9 张批量裤区 inpaint |
| `scripts/run_linxia_lora_train.bat` | 林夏 LoRA launcher |
| `scripts/run_linxia_lora_train_v2.bat` | v2 emphasis caption |
| `scripts/run_linxia_lora_train_v3.bat` | v3 mass dataset 1500 步 |
| `scripts/run_lihui_real_lora_train.bat` | 李辉 LoRA launcher |
| `scripts/run_lora_continue_train.bat` | 续训通用 (1200→2400) |
| `tests/test_ep04_lora_workflow.py` | 9 个 LoRA graph 结构测试 |

---

## 🔀 关键 git commits (本分支 `ep01-wan-photoreal`)

```
7a374ed  docs(handoff): ep03 v3 LoRA + 李辉 LoRA 训满 + 工作 session 收尾
b5dbf96  docs(handoff): ep03 v3 LoRA 实战出片定档 — 灰侧线 0/5 -> 5/5 突破
3b98fd4  feat(ep04): gen_ep03_seedance.py 加 --kf-dir / --out 参数化 (v3 实战需)
416d352  feat(ep04): inpaint pilot (logo + pants) + 扩数据集 28 张 + v3 launcher
621c107  feat(ep04): linxia v2 加 logo_zoom + pants_zoom 特写 + caption emphasis
67d2e26  feat(ep04): 1200 vs 2400 续训 side-by-side compare driver
f1a33cb  feat(ep04): 续训通用 launcher — 1200→2400 步 --network_weights
58dd9aa  feat(ep04): 双角色 LoRA 流水线 + A/B/C ablation
dd17ed6  feat(ep04): ep03 LoRA 版实战脚本 — 用 linxia_lora 1200 step 重出关键帧
6226801  feat(ep05): 李辉 LoRA 数据集 + launcher (ep01 男角治本路径)
```

---

## 🗺️ 未完成 / 后续工作 (留给下次接续)

### 短期 (下次会话)
- **(a) 李辉 LoRA v2 caption 修正**:重生成 dataset + 重训 ~13 min (LoRA 修对蓝格子衬衫)
- **(b) ep01 v5 LoRA 实战出片**:用 lihui_real_lora + 双 IPA 出 ep01_real_v5.mp4 ~30 min
- **(c) shot8 drift fix**:v4 标 drifted,重抽 i2v seed 选 best,~10 min

### 中期 (治本优化)
- **(d) ep03 v3 xfade 边界双字幕/双身 bug**:xfade=0.2 fadeblack + 修 `lib/render_subtitle.py` 宽屏 bug
- **(e) 装 XL 域 / 专用 inpaint 模型治白弧 logo** (本地缺 + HF 401 卸不了,需找镜像源)
- **(f) LoRA 治本白弧 logo**:用 ComfyUI inpaint 加精确 mask + 改 prompt 重画立绘(SD1.5 局限)

### 长期 (战略)
- ep01 同步写实化 (ep01 v4 已写实,但 shot8 drift + 男角 LoRA 没修完)
- ep05 / ep06 / ep07... 新故事
- HuggingFace 镜像源(B 站 / 国内镜像)装专用 inpaint ckpt

---

## 📂 关键产物路径

```
output/
├── ep03_assemble/
│   └── ep03_real_v3.mp4                # ep03 v3 LoRA 实战版 ⭐
├── ep03_real/                          # ep03 关键帧 (无 LoRA v10)
├── ep03_real_lora1200/                 # LoRA v1 5 镜
├── ep03_real_v2/                       # LoRA v2 (emphasis)
├── ep03_real_v3/                       # LoRA v3 5 镜 ⭐
├── contact_v0_v1_v2_v3.png             # 4 列 contact sheet ⭐
├── lihui_lora/checkpoints*/            # 小雅 LoRA v1 全 ckpts 备份
├── lihui_real_lora/                    # 李辉 v1 dataset + ckpts
├── linxia_lora/
│   ├── checkpoints/                    # v1 (4 ckpts)
│   ├── checkpoints_continue/           # v1 续训 (3 ckpts)
│   ├── checkpoints_v2/                 # v2 emphasis
│   └── checkpoints_v3/                 # v3 mass dataset
└── ep03_lihui/inpaint_pants/           # 9 张批量裤区 inpaint
```

---

## 🧠 下次会话快速续接

**启动指令**(下一次会话起手):
1. 读 `../manju/CLAUDE.md` (项目规则)
2. 读 `../manju/HANDOFF.md` (活状态,含 `SESSION-SUMMARY-2026-07-23-to-24.md` 链)
3. 看 git status (应该干净,最新 commit `7a374ed`)
4. 询问用户拍板:(a)(b)(c) 选哪个开始
5. 沿用现有 launcher 续(launcher 文件都已固化)

**接续关键能力**:
- `gen_ep03_real_lora_kf.py --out-suffix v3 --lora linxia_lora_v3.safetensors` 直接复用
- `aug_linxia_dataset.py` / `aug_lihui_real_dataset.py` 直接复用
- 训练 launcher 全部含 `--max_data_loader_n_workers 0` (避 spawn MemoryError)
- 6 个 PIL.batch 文件 ASCII 注释,不会触发 PS5.1 GBK 解析坑

---

## 💡 关键 takeaway

1. **LoRA 学大件颜色 OK,小标志(白弧 logo)卡上限**:治本需换 XL 域
2. **LoRA 在 SD1.5 上治本服装是"数据集 majority"原则**:不一致时 LoRA 学数据偏
3. **face_sim 退化是 LoRA 副作用**:服装治本越狠,脸一致性损失越大
4. **Inpaint 在 SD1.5 上精度不够**:denoise 0.5 仍全身重画,治本需专用 inpaint ckpt
5. **延续 ComfyUI 工作必须先查 `/system`**:找真 8188 listen 状态别被 start.py 误导
6. **batch 文件永远用 ASCII 注释**:中文 PS5.1 + Git Bash 双层错

---

**作者**: Claude (model MiniMax-M3.0)
**会话长度**: 5h ~ 6h
**总 commits**: 10 (本会话) + 历史 4 = 14 总
**测试状态**: 36/36 passing
**工作树**: clean
