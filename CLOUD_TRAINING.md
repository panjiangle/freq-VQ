# 新 4090 实例：自己在 VS Code 中启动训练

本机 VS Code 已配置 SSH 别名 `freq-vq-4090`。云端为 Ubuntu 16.04，当前 Microsoft Remote-SSH 的服务器要求更新的 glibc；请在 VS Code **集成终端**输入以下命令。需要在侧栏查看云端图片和文件时，可使用已配置的 **SSH FS** 扩展中的 `freq-vq-4090` 连接，它通过 SFTP 读取云端目录。

```bash
ssh freq-vq-4090
cd /root/autodl-tmp/freq-vq/FeMaSR
screen -S freqvq-hq
bash scripts/train_hq_cloud.sh
```

在 `screen` 中按 `Ctrl+A`、再按 `D` 可以断开显示，训练继续；重新连接后执行 `screen -r freqvq-hq`。另一终端可用 `tail -f cloud_logs/hq_train.log` 和 `nvidia-smi` 监控。

HQ 完成并保存 `experiments/008_FeMaSR_HQ_wavelet2/models/net_g_latest.pth` 后，再启动 LQ 阶段：

```bash
screen -S freqvq-lq
bash scripts/train_lq_cloud.sh
```

两阶段不可同时训练。训练日志分别是 `cloud_logs/hq_train.log` 与 `cloud_logs/lq_train.log`。中间频带图位于 `experiments/<实验名>/visualization/wavelet_bands/`；LQ 超分图位于 `experiments/015_FeMaSR_LQ_wavelet2/visualization/image_results/`。LQ 每轮验证保存前四张 SR 图，每 50,000 步保存全部 100 张；指标始终在全部 100 张上计算。

HQ 配置的 `total_iter` 为 **200,000 步**，已于 2026-11-07 跑满并保存 `200000.state`、`net_g_200000.pth` 和 `net_g_best_.pth`。LQ 配置目前仍以原版的 2,000,000 步为停止上限，尚未启动。两个阶段都每 5,000 步验证，同时保存 `latest` 和带步数的完整断点（含优化器状态）。每个阶段仅保留最近三个完整断点；启动脚本启用 `--auto_resume`，中断后重新运行同一脚本可从最近完整断点恢复。`latest` 权重本身不含优化器状态，不能单独用来精确续训。上限不代表保证收敛；请根据验证曲线与租机预算决定实际停点。若要改步数，**启动前**编辑相应配置文件中的 `train.total_iter`。

数据放在 `/root/autodl-tmp/freq-vq/datasets/DF2K_official`，训练集 3,450 对（DIV2K 发布方 800 对 + Flickr2K 公开镜像 2,650 对），验证集为 DIV2K 发布方 100 对。启动脚本会在开跑前检查全部配对文件和尺寸；不制作新的 LR 图。Flickr2K 原作者主机当前不可达，镜像来源与核验限制详见 `WAVELET_VQ_README.md`。
