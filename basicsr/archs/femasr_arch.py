"""FeMaSR 的网络结构入口。

阅读顺序：VectorQuantizer（特征量化）→ MultiScaleEncoder（提取多尺度特征）
→ FeMaSRNet（把编码、码本和逐级解码串起来）。这里的 HQ/LQ 分别指
高质量图像预训练阶段和低质量图像恢复阶段；这个文件定义网络，不负责训练循环。
BasicSR 通过 ARCH_REGISTRY 找到 FeMaSRNet；训练、HQ 权重加载与冻结逻辑
在 basicsr/models/femasr_model.py，具体参数在 options/ 的 YAML 中。
"""

import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

from .network_swinir import RSTB
from .fema_utils import ResBlock, CombineQuantBlock 
from .vgg_arch import VGGFeatureExtractor


# ==================== WAVELET-VQ CHANGE 01 BEGIN: 固定 Haar DWT / IDWT ====================
# 只重排和线性组合空间采样，不引入可训练参数；两轮变换见 _quantize_wavelet()。
def haar_dwt2d(x):
    """B,C,H,W -> LL 与 (LH, HL, HH)，每个子带的宽高减半。

    LH 表示行方向低通、列方向高通；HL 则相反，IDWT 使用同一约定。
    """
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f'Haar DWT requires even height and width, got {tuple(x.shape[-2:])}')
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh)


def haar_idwt2d(ll, details):
    """按小波位置合成 LL 与 LH/HL/HH；它们必须同形状。"""
    if len(details) != 3 or any(detail.shape != ll.shape for detail in details):
        raise ValueError('Haar IDWT requires LL, LH, HL and HH with identical shapes')
    lh, hl, hh = details
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    bsz, channels, height, width = ll.shape
    return F.pixel_shuffle(torch.stack((a, b, c, d), dim=2).reshape(
        bsz, channels * 4, height, width), 2)
# ==================== WAVELET-VQ CHANGE 01 END ===========================================


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________

    中文提示：每个空间位置的 e_dim 维向量选一个最近码字。这里的码本是
    nn.Embedding 的可训练参数；它量化编码器特征，不直接量化 RGB/小波系数。
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def dist(self, x, y):
        # 返回每个输入向量到全部码字的平方欧氏距离；只省略了共同的平方根。
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        # 比较两个特征的通道相关性，用于 LQ 阶段的纹理约束。
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: 可选的 HQ 教师索引图，只用于 LQ 训练损失；不会强制替换本分支所选码字。
        """
        # 每个像素位置对应一个待量化的特征向量：B,C,H,W → B*H*W,C。
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        
        # 为每个位置选最近码字，并用 one-hot 表示索引。
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            # LQ 训练时传入 HQ 教师的索引；它用于下面的监督损失。
            # 当前前向量化结果 z_q 仍由本分支自己计算的最近索引决定。
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # detach 控制梯度去向：编码器靠 e_latent_loss 靠近码字，
        # 码字靠 q_latent_loss 靠近编码器输出。
        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            # LQ 阶段让预测特征接近 HQ 教师选出的码字，并匹配通道相关性。
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean() 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # Straight-through：前向值等于离散码字，反向梯度按恒等映射回传给 z。
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    
    def get_codebook_entry(self, indices):
        # 已知离散索引时直接查表得到 B,C,H,W 特征，不再运行最近邻搜索。
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

class SwinLayers(nn.Module):
    """把特征图展成 token 序列，连续通过四个 RSTB，再还原为特征图。"""
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        # RSTB 接受 B,HW,C；外部网络其余部分使用 B,C,H,W。
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x


class MultiScaleEncoder(nn.Module):
    """共用的多尺度编码器：HQ 逐级下采样；LQ 在低分辨率处加 Swin 和上采样。"""
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        # 入口卷积把 RGB 转成特征通道；下方每个 block 将空间尺寸约减半。
        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage: 
            # LQ 输入缺少细节：在最深层用 Swin 建模上下文，再生成较高尺度特征。
            self.blocks.append(SwinLayers(**swin_opts))
            upsampler = nn.ModuleList()
            for i in range(2):
                in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
                upsampler.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    )
                )
                res = res * 2
        
            self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        # outputs 按实际运行顺序存储各尺度结果，HQ/LQ 的列表顺序因此不同。
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class DecoderBlock(nn.Module):
    """单级特征解码：上采样 2 倍、卷积，再经两个残差块。"""

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


@ARCH_REGISTRY.register()
class FeMaSRNet(nn.Module):
    """BasicSR 注册的 FeMaSR 主网络。

    HQ 阶段输入高质量图像，学习码本和解码器；LQ 阶段输入低质量图像，
    学习预测可被 HQ 码本/解码器使用的特征。两阶段共用本类，但走不同编码路径。
    """
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=4,
                 use_semantic_loss=False,
                 use_residual=True,
                 # >>> WAVELET-VQ CHANGE 02: 新增可选模式和高频码本容量
                 wavelet_vq=False,
                 detail_codebook_size=None,
                 # <<< WAVELET-VQ CHANGE 02
                 **ignore_kwargs):
        super().__init__()

        # 官方路径每行是 [特征分辨率, 码字数量, 码字维度]，例如 [32, 1024, 512]。
        # 小波路径只用首行指定 32x32 解码起点、LL2 码本容量和码字维度；
        # 高频码本容量由 detail_codebook_size 单独指定，不把 8x8/16x16 误作解码尺度。
        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        # HQ 阶段输入/输出同尺寸；LQ 阶段输出尺寸为输入的 scale_factor 倍。
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual
        # >>> WAVELET-VQ CHANGE 02 BEGIN: 校验小波模式配置；关闭时仍走官方路径。
        self.wavelet_vq = wavelet_vq
        if self.wavelet_vq:
            if codebook_params.shape != (1, 3) or int(self.codebook_scale[0]) != 32:
                raise ValueError('Wavelet VQ expects one 32x32 decoder scale in codebook_params')
            if int(codebook_emb_dim[0]) != 512:
                raise ValueError('Wavelet VQ expects 512-dimensional codes for the existing decoder')
            if detail_codebook_size is None or int(detail_codebook_size) < 1:
                raise ValueError('Wavelet VQ requires a positive detail_codebook_size')
        # <<< WAVELET-VQ CHANGE 02 END

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # max_depth 是从首个码本尺度逐级上采样回 GT 尺寸所需的级数。
        # LQ 的实际编码输入更小，因此 encode_depth 按输入分辨率另算。
        # build encoder 
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, LQ_stage
                            )


        # 解码器逐级把低分辨率特征放大到目标图像尺寸。
        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # 对每个指定尺度分别建立：量化前通道映射、码本、量化后融合模块。
        # 多码本时，后一级还会接收前一级的量化/解码信息。
        # build multi-scale vector quantizers 
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        if self.wavelet_vq:
            # >>> WAVELET-VQ CHANGE 03 BEGIN: 两个码本按频带分工，解码起点仍是 32x32。
            # 第 0 个量化 LL2，第 1 个在 D2 与 D1 的六个方向子带间共享。
            code_dim = int(codebook_emb_dim[0])
            self.quantize_group.append(VectorQuantizer(
                int(codebook_emb_num[0]), code_dim, LQ_stage=self.LQ_stage))
            self.quantize_group.append(VectorQuantizer(
                int(detail_codebook_size), code_dim, LQ_stage=self.LQ_stage))
            self.before_quant_group.append(nn.Conv2d(channel_query_dict[32], code_dim, 1))
            self.after_quant_group.append(CombineQuantBlock(code_dim, 0, channel_query_dict[32]))
            # <<< WAVELET-VQ CHANGE 03 END
        else:
            # 官方实现：每个配置行对应一个需要量化的解码尺度。
            for scale in range(0, codebook_params.shape[0]):
                quantize = VectorQuantizer(
                    codebook_emb_num[scale],
                    codebook_emb_dim[scale],
                    LQ_stage=self.LQ_stage,
                )
                self.quantize_group.append(quantize)

                scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
                if scale == 0:
                    quant_conv_in_ch = scale_in_ch
                    comb_quant_in_ch1 = codebook_emb_dim[scale]
                    comb_quant_in_ch2 = 0
                else:
                    quant_conv_in_ch = scale_in_ch * 2
                    comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                    comb_quant_in_ch2 = codebook_emb_dim[scale]

                self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
                self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # 可选 VGG 语义监督，只在启用 use_semantic_loss 时构造。
        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
                )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer]) 

        # >>> WAVELET-VQ CHANGE 04 BEGIN: 验证阶段才开启的中间子带收集开关。
        self._record_wavelet_visuals = False
        self._last_wavelet_visuals = None
        # <<< WAVELET-VQ CHANGE 04 END

    # ==================== WAVELET-VQ CHANGE 05 BEGIN: 两轮分解、分频量化、两轮合成 ====================
    def _quantize_wavelet(self, feat_to_quant, gt_indices=None):
        """32x32 特征分解、分频带量化、再合成回 32x32。"""
        if gt_indices is not None and len(gt_indices) != 7:
            raise ValueError('Wavelet VQ teacher must provide 7 index maps: LL2, D2(3), D1(3)')

        # 第一轮 DWT 得到 LL1、D1；第二轮只分解 LL1，得到 LL2、D2。
        ll1, d1 = haar_dwt2d(feat_to_quant)
        ll2, d2 = haar_dwt2d(ll1)
        bands = (ll2, *d2, *d1)  # 顺序固定为 LL2, LH2, HL2, HH2, LH1, HL1, HH1。
        quantized, losses, indices = [], [], []
        for band_idx, band in enumerate(bands):
            # LL2 使用第 0 个码本；六个高频方向/尺度共用第 1 个码本。
            quantizer = self.quantize_group[0 if band_idx == 0 else 1]
            teacher_index = None if gt_indices is None else gt_indices[band_idx]
            quant_band, band_loss, band_indices = quantizer(band, teacher_index)
            quantized.append(quant_band)
            losses.append(band_loss)
            # 索引图只存码字编号：B x 1 x H x W，每个空间位置对应一个整数。
            indices.append(band_indices)

        if self._record_wavelet_visuals:
            # 只在验证时截取首张样本并移到 CPU，避免训练期间持有整批特征图。
            self._last_wavelet_visuals = {
                name: (bands[i][:1].detach().cpu(), quantized[i][:1].detach().cpu())
                for i, name in enumerate(('LL2', 'LH2', 'HL2', 'HH2'))
            }

        # D1 每张子带有 16x16=256 个位置，D2 每张有 8x8=64 个，故位置数是 4 倍。
        # 但 VectorQuantizer 已对各子带全部位置求均值，位置多不会自动增大 band_loss。
        # 这里主动设定 LL2:D1:D2 = 1/2:1/4:1/4；每级的三个方向再取平均。
        d2_loss = sum(losses[1:4]) / 3
        d1_loss = sum(losses[4:7]) / 3
        codebook_loss = (losses[0] + (d2_loss + d1_loss) * 0.5) * 0.5

        # 与官方 use_quantize=False 消融一致：仍计算 VQ 损失，但解码使用连续子带。
        reconstructed_bands = bands if not self.use_quantize else quantized
        reconstructed_ll1 = haar_idwt2d(reconstructed_bands[0], reconstructed_bands[1:4])
        reconstructed_feat = haar_idwt2d(reconstructed_ll1, reconstructed_bands[4:7])
        return reconstructed_feat, codebook_loss, indices
    # ==================== WAVELET-VQ CHANGE 05 END ============================================

    def encode_and_decode(self, input, gt_indices=None, current_iter=None):
        """主数据流：图像→多尺度特征→指定尺度量化→逐级解码→输出图像。"""
        # detach 只阻断对输入图像求梯度；编码器参数仍可训练。
        enc_feats = self.multiscale_encoder(input.detach())
        if self.LQ_stage:
            # LQ 编码器含 Swin 与上采样，取末尾三层供解码器使用。
            enc_feats = enc_feats[-3:]
        else:
            # HQ 编码器依次下采样；翻转后从最粗尺度开始解码。
            enc_feats = enc_feats[::-1]

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        x = enc_feats[0]
        for i in range(self.max_depth):
            # 每轮处理一个解码尺度；只在 codebook_scale 指定的尺度做 VQ。
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    # 当前编码特征与上一尺度的解码结果共同决定待量化特征。
                    before_quant_feat = torch.cat((enc_feats[i], prev_dec_feat), dim=1)
                else:
                    before_quant_feat = enc_feats[i]
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                # >>> WAVELET-VQ CHANGE 06 BEGIN: 在原 32x32 量化入口切换到小波双码本。
                if self.wavelet_vq:
                    # 返回的是两轮 IDWT 后的 32x32 特征，后续解码器无需改变输入尺寸。
                    z_quant, codebook_loss, indices = self._quantize_wavelet(feat_to_quant, gt_indices)
                else:
                    if gt_indices is not None:
                        # LQ 训练可以传入 HQ 阶段产生的索引，作为码本损失的监督。
                        z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx])
                    else:
                        z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant)
                # <<< WAVELET-VQ CHANGE 06 END

                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)
                
                # >>> WAVELET-VQ CHANGE 07 BEGIN: 小波消融已在 _quantize_wavelet() 内选择原始子带。
                if not self.use_quantize and not self.wavelet_vq:
                    # 消融开关：仍计算码本及损失，但解码器接收连续特征。
                    z_quant = feat_to_quant
                # <<< WAVELET-VQ CHANGE 07 END

                # 将当前码字和前一级码字组合，映射到解码器需要的通道数。
                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                # >>> WAVELET-VQ CHANGE 08 BEGIN: HQ 教师向 LQ 返回七张子带索引图。
                if self.wavelet_vq:
                    indices_list.extend(indices)
                else:
                    indices_list.append(indices)
                # <<< WAVELET-VQ CHANGE 08 END

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat
            else:
                if self.LQ_stage and self.use_residual:
                    # 非量化尺度的 LQ 编码特征直接加到解码路径，补充连续信息。
                    x = x + enc_feats[i]
                else:
                    x = x 

            # 本级解码后空间分辨率翻倍，供下一尺度继续使用。
            x = self.decoder_group[i](x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        codebook_loss = sum(codebook_loss_list)
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        # 返回图像、量化损失、可选语义损失、每级码本索引。
        return out_img, codebook_loss, semantic_loss, indices_list
    
    def decode_indices(self, indices):
        """已知码字编号时直接查表并重建图像；正常前向仍由编码器选编号。

        小波模式的顺序为 LL2, LH2, HL2, HH2, LH1, HL1, HH1。
        256x256 训练图对应 8x8 的前四张索引图、16x16 的后三张索引图。
        """
        # >>> WAVELET-VQ CHANGE 09 BEGIN: 七张索引图按 LL2、D2、D1 顺序查两套码本。
        if self.wavelet_vq:
            if len(indices) != 7:
                raise ValueError('Wavelet VQ decoding requires 7 index maps')
            ll2 = self.quantize_group[0].get_codebook_entry(indices[0])
            details = [self.quantize_group[1].get_codebook_entry(idx) for idx in indices[1:]]
            ll1 = haar_idwt2d(ll2, details[:3])
            z_quant = haar_idwt2d(ll1, details[3:])
        else:
            assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'
            z_quant = self.quantize_group[0].get_codebook_entry(indices)
        # <<< WAVELET-VQ CHANGE 09 END；以下量化后卷积和三级解码为 FeMaSR 原路径。
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # 大图推理：切成带重叠边界的小块，分别调用 test，再拼回输出。
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input, collect_wavelet=False):
        # 推理时先镜像填充，使输入尺寸适配窗口/下采样约束；最后裁回原尺寸。
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False
        # >>> WAVELET-VQ CHANGE 10 BEGIN: 验证时才收集量化前后 LL2/D2。
        self._record_wavelet_visuals = bool(collect_wavelet and self.wavelet_vq)
        self._last_wavelet_visuals = None
        # <<< WAVELET-VQ CHANGE 10 END

        # padding to multiple of window_size * 8
        wsz = 8 // self.scale_factor * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        dec, _, _, _ = self.encode_and_decode(input)

        output = dec
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]

        self.use_semantic_loss = org_use_semantic_loss 
        # >>> WAVELET-VQ CHANGE 11 BEGIN: 保持普通 test() 返回图像；验证可额外取子带。
        self._record_wavelet_visuals = False
        if collect_wavelet:
            return output, self._last_wavelet_visuals
        # <<< WAVELET-VQ CHANGE 11 END
        return output 

    def forward(self, input, gt_indices=None):
        # BasicSR 训练调用的入口；gt_indices 通常只在 LQ 训练时由 HQ 教师提供。

        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, codebook_loss, semantic_loss, indices = self.encode_and_decode(input, gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, codebook_loss, semantic_loss, indices = self.encode_and_decode(input)

        return dec, codebook_loss, semantic_loss, indices
