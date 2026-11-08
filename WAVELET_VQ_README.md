# FeMaSR 两轮 DWT、双码本实验

此实现通过 `network_g.wavelet_vq: true` 启用；官方配置不启用时仍走原版单码本路径。所有新增代码用 `[WaveletVQ-*]` 标记，并在旁边说明用途。

为方便 GitHub 阅读，代码里的新逻辑统一用 `WAVELET-VQ CHANGE NN BEGIN/END` 圈出。可在仓库中搜索 `WAVELET-VQ CHANGE`：网络文件的 01–11 对应 Haar 变换、双码本、两轮量化/合成、索引解码和测试收集；训练模型文件的 12–16 对应检查点校验和验证图保存。新 YAML 文件顶部也有醒目标记。

## 数据流

1. FeMaSR 的 32×32、256 通道编码特征先经原有 1×1 卷积变为 512 通道。
2. 第一轮固定 Haar DWT：得到 LL1 与 D1=(LH1, HL1, HH1)，每个子带为 16×16。
3. 只对 LL1 再做一次 DWT：得到 LL2 与 D2=(LH2, HL2, HH2)，每个子带为 8×8。
4. LL2 用第一个码本；D2 和 D1 的六个方向子带分别查同一个高频码本。返回索引的顺序固定为 `LL2, LH2, HL2, HH2, LH1, HL1, HH1`。
5. 先 IDWT(LL2,D2) 得 LL1，再 IDWT(LL1,D1) 得 32×32、512 通道特征。之后的 3×3 通道映射、三级解码，以及 LQ 阶段 64×64/128×128 连续特征相加，都沿用 FeMaSR。

这里的小波子带来自**可学习特征**，不是 RGB 图像的小波系数。D1 和 D2 各有三个方向，不能直接相加代替 IDWT 所需的方向输入。

每张 D1 子带为 16×16=256 个位置，每张 D2 子带为 8×8=64 个位置，所以 D1 的位置数是 D2 的 4 倍。`VectorQuantizer` 已对每张子带的位置求均值，因此原先“避免 D1 仅因位置更多而主导训练”的注释不够准确。当前总量化损失明确采用 `LL2:D1:D2 = 1/2:1/4:1/4`；这是主动选择的尺度权重，三个方向在各自尺度内平均。

## 训练配置

- HQ: `options/train_FeMaSR_HQ_wavelet2.yml`。从头训练新的双码本 HQ 网络。
- LQ: `options/train_FeMaSR_LQ_wavelet2.yml`。加载上述 HQ 配置产出的 `net_g_200000.pth`；不能加载官方单码本 HQ 权重。
- 两个码本各有 512 个、512 维码字，embedding 参数总量等于原版一个 1024×512 码本。HQ 已于 2026-11-07 训练至 200,000 步（从 120,000 步断点续训）；LQ 阶段于 2026-11-08 从零启动，目标 200,000 步。每 5,000 步保存断点，实际停止点应结合验证曲线决定。
- HQ 教师在 LQ 训练时提供七张对应子带的索引图。LQ 自己仍按最近邻选码，教师索引仅参与监督。

### 当前云端数据来源与读取方式

- DIV2K：直接从数据集发布方下载 train/valid 的 HR 与 ×4 bicubic LR，分别为 800/100 对。详见 `scripts/prepare_official_div2k.sh`。
- Flickr2K：原作者下载主机从当前实例不可达，使用公开 DF2K 镜像中已有的 Flickr2K HR/×4 LR，2,650 对。镜像自称汇集原始数据；由于无法从原作者主机获取哈希，不能独立确认镜像字节与原发布文件完全相同。详见 `scripts/prepare_flickr2k_mirror.py`。
- 两个阶段均从**现成的 HR/LR 配对图**读取。训练只在图上作同位置裁块和翻转/旋转；没有自制 bicubic LR，也不会再随机改变原有 LR 退化。验证在官方 DIV2K 配对图上取固定中心 512×512 HR / 128×128 LR 块；HQ 阶段以 HR 输入检查重建，LQ 阶段以 LR 输入检查超分。
- 云端启动脚本为 `scripts/train_hq_cloud.sh`、`scripts/train_lq_cloud.sh`；前者会确认 3,450 组训练和 100 组验证文件完整，后者会先检查 HQ 权重存在。两阶段不要同时运行。

从本仓库根目录运行时，请把**本仓库绝对路径**放在 `PYTHONPATH` 最前面。本机发现另一个项目的 editable `basicsr` 会被误导入；验证脚本已自行固定当前仓库路径。

Linux 云端启动示例（在本仓库根目录；先核对 YAML 中的数据路径）：

```bash
PYTHONPATH="$PWD" python -m basicsr.train -opt options/train_FeMaSR_HQ_wavelet2.yml
# HQ 保存 net_g_200000.pth 后，再运行：
PYTHONPATH="$PWD" python -m basicsr.train -opt options/train_FeMaSR_LQ_wavelet2.yml
```

## 每轮验证的产物

两份配置均设置 `wavelet_vis_max_samples: 1`，每轮验证保存首张样本。目录为：

`experiments/<实验名>/visualization/wavelet_bands/<数据集名>/<步数>/<图片名>/`

其中 `LL2_D2_before_after.png` 是两排四列拼图：第一排量化前、第二排量化后；四列依次为 LL2、LH2、HL2、HH2。它显示 512 个通道的 RMS 能量，并在同一个子带的前后图之间共用强度尺度。`LL2_D2_signed_channel0.png` 则展示固定第 0 通道的带符号系数，红色为正、蓝色为负。目录里还有八张单独的能量图。**这些是特征预览，不是 RGB 小波图。**

若需分析完整系数，把 `wavelet_vis_save_raw` 改为 `true`，会额外保存量化前后的四组 512 通道张量；长期训练默认关闭以节省空间。

## LQ 推理与评测口径

`inference_femasr.py` 会根据 LQ 权重中的两套码本自动启用小波模式；官方单码本权重仍走原版路径。推理需使用 LQ 阶段的 `net_g_*.pth`，HQ 的 `net_g_200000.pth` 是 HR tokenizer，编码器与 SR 网络不同，不能直接用于 LR→SR 推理。

```bash
python inference_femasr.py -s 4 -i /path/to/LR_images \
  -w experiments/015_FeMaSR_LQ_wavelet2/models/net_g_best_.pth \
  -o /path/to/SR_images
```

本实验的配对 LR 来自 DIV2K 官方 ×4 bicubic 文件及 Flickr2K 配对镜像；“配对”说明每张 LR 有对应 HR，不说明退化类型。FeMaSR 官方论文复现说明使用 BSRGAN 生成训练 LR、BSRGAN+ 生成合成测试 LR；当前 bicubic 训练和中心 512×512 裁剪验证与其协议不同，因此当前指标不能直接与论文表格比较。严格架构对照应让原版 FeMaSR 在同一配对数据、裁剪、训练步数和指标实现下重训。来源：[DIV2K 官方数据说明](https://data.vision.ee.ethz.ch/cvl/DIV2K/)、[FeMaSR 官方训练说明](https://github.com/chaofengc/FeMaSR#train-the-model)。

## 本地功能检查

`tests/test_wavelet_vq_smoke.py` 检查两轮 DWT/IDWT 的可逆性与梯度、HQ/LQ 前向和反向、七张教师索引、原版单码本入口，以及真实一次验证是否写出拼图。此检查不代表模型已训练收敛或超分质量已提高。
