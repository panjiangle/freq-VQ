from collections import OrderedDict
from os import path as osp
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa


@MODEL_REGISTRY.register()
class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items(): 
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            # >>> WAVELET-VQ CHANGE 12 BEGIN: 防止 LQ 阶段误加载官方单码本 HQ 权重。
            if hq_opt.get('wavelet_vq', False):
                # HQ/LQ 的语义模块和编码器不完全同构，仍采用原有非严格加载；
                # 但必须先确认新 HQ 权重确实包含两套正确形状的频带码本。
                if load_path.startswith('https://'):
                    raise ValueError('Wavelet VQ LQ stage requires a locally trained two-codebook HQ checkpoint')
                checkpoint = torch.load(load_path, map_location='cpu')
                state = checkpoint.get('params', checkpoint)
                state = {(key[7:] if key.startswith('module.') else key): value
                         for key, value in state.items()}
                expected = self.get_bare_model(self.net_hq).state_dict()
                for key in ('quantize_group.0.embedding.weight',
                            'quantize_group.1.embedding.weight'):
                    if key not in state or state[key].shape != expected[key].shape:
                        raise ValueError(f'HQ checkpoint has no compatible wavelet codebook: {key}')
                del checkpoint, state
            # <<< WAVELET-VQ CHANGE 12 END
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, gt_indices = self.net_hq(self.gt)

            self.output, l_codebook, l_semantic, _ = self.net_g(self.lq, gt_indices) 
        else:
            self.output, l_codebook, l_semantic, _ = self.net_g(self.gt) 

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantic *= train_opt['semantic_opt']['loss_weight'] 
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    # >>> WAVELET-VQ CHANGE 13 BEGIN: 测试入口可选返回中间子带，训练/普通推理沿用原调用。
    def test(self, collect_wavelet=False):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        # >>> WAVELET-VQ CHANGE 21 BEGIN: HQ validation reconstructs HR, LQ validation super-resolves LR.
        lq_input = self.gt if (not self.LQ_stage and net_g.wavelet_vq) else self.lq
        # <<< WAVELET-VQ CHANGE 21 END
        _, _, h, w = lq_input.shape
        self.wavelet_visuals = None
        if h*w < min_size:
            if collect_wavelet:
                self.output, self.wavelet_visuals = net_g.test(lq_input, collect_wavelet=True)
            else:
                self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()
    # <<< WAVELET-VQ CHANGE 13 END

    # ==================== WAVELET-VQ CHANGE 14 BEGIN: 验证子带可视化 ====================
    def _save_wavelet_visuals(self, visuals, dataset_name, current_iter, img_name):
        """保存 LL2/D2 量化前后的通道 RMS 能量图和固定通道带符号图。"""
        output_dir = osp.join(self.opt['path']['visualization'], 'wavelet_bands',
                              dataset_name, str(current_iter), img_name)
        os.makedirs(output_dir, exist_ok=True)
        rows = [[], []]
        signed_rows = [[], []]
        scale_lines = ['Each before/after pair uses one shared intensity scale.']
        for name in ('LL2', 'LH2', 'HL2', 'HH2'):
            before, after = visuals[name]
            before_energy = before[0].float().square().mean(dim=0).sqrt().numpy()
            after_energy = after[0].float().square().mean(dim=0).sqrt().numpy()
            pair_scale = max(float(before_energy.max()), float(after_energy.max()), 1e-8)
            scale_lines.append(f'{name}: channel-RMS maximum = {pair_scale:.8g}')
            signed_scale = max(float(before[0, 0].abs().max()),
                               float(after[0, 0].abs().max()), 1e-8)
            for row, (phase, energy, band) in enumerate((
                    ('before', before_energy, before), ('after', after_energy, after))):
                preview = np.clip(energy / pair_scale * 255.0, 0, 255).astype(np.uint8)
                preview = cv2.resize(preview, (256, 256), interpolation=cv2.INTER_NEAREST)
                preview = cv2.applyColorMap(preview, getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET))
                cv2.putText(preview, f'{name} {phase}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                imwrite(preview, osp.join(output_dir, f'{name}_{phase}.png'))
                rows[row].append(preview)
                # 固定第 0 通道的带符号系数：红=正、蓝=负；同一对量化前后共用强度尺度。
                signed = np.clip(band[0, 0].float().numpy() / signed_scale, -1, 1)
                positive = np.maximum(signed, 0)
                negative = np.maximum(-signed, 0)
                signed_preview = np.stack((negative, np.zeros_like(signed), positive), axis=-1)
                signed_preview = (signed_preview * 255.0).astype(np.uint8)
                signed_preview = cv2.resize(signed_preview, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.putText(signed_preview, f'{name} {phase}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                signed_rows[row].append(signed_preview)
        # 第一排量化前、第二排量化后；D2 的三个方向分列展示。
        contact_sheet = np.concatenate((np.concatenate(rows[0], axis=1),
                                        np.concatenate(rows[1], axis=1)), axis=0)
        imwrite(contact_sheet, osp.join(output_dir, 'LL2_D2_before_after.png'))
        signed_sheet = np.concatenate((np.concatenate(signed_rows[0], axis=1),
                                       np.concatenate(signed_rows[1], axis=1)), axis=0)
        imwrite(signed_sheet, osp.join(output_dir, 'LL2_D2_signed_channel0.png'))
        with open(osp.join(output_dir, 'README.txt'), 'w', encoding='utf-8') as handle:
            handle.write('Feature energy previews, not RGB wavelet images.\n'
                         'Columns: LL2, LH2, HL2, HH2. Rows: before and after VQ.\n'
                         'Each pixel is the RMS across 512 feature channels.\n'
                         'The signed_channel0 sheet shows one fixed channel: red positive, blue negative.\n'
                         + '\n'.join(scale_lines) + '\n')
        if self.opt['val'].get('wavelet_vis_save_raw', False):
            torch.save(visuals, osp.join(output_dir, 'LL2_D2_raw.pt'))
    # ==================== WAVELET-VQ CHANGE 14 END =====================================
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            # >>> WAVELET-VQ CHANGE 15 BEGIN: 每轮验证仅保存前 N 张，控制长期训练磁盘占用。
            wavelet_sample_limit = int(self.opt['val'].get('wavelet_vis_max_samples', 0))
            collect_wavelet = (idx < wavelet_sample_limit and
                               getattr(self.get_bare_model(self.net_g), 'wavelet_vq', False))
            self.test(collect_wavelet=collect_wavelet)
            if collect_wavelet and self.wavelet_visuals is not None:
                self._save_wavelet_visuals(self.wavelet_visuals, dataset_name, current_iter, img_name)
                self.wavelet_visuals = None
            # <<< WAVELET-VQ CHANGE 15 END
            
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # >>> WAVELET-VQ CHANGE 25 BEGIN: keep long-run validation images within disk budget.
            image_limit = self.opt['val'].get('save_img_max_samples')
            full_freq = self.opt['val'].get('save_img_full_freq')
            save_this_img = save_img and (
                image_limit is None or idx < image_limit or
                (full_freq and current_iter % full_freq == 0))
            if save_this_img:
            # <<< WAVELET-VQ CHANGE 25 END
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item() 

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        # >>> WAVELET-VQ CHANGE 16 BEGIN: 原单索引可视化不适用于小波模式的七张索引图。
        if not self.LQ_stage and not getattr(self.get_bare_model(self.net_g), 'wavelet_vq', False):
            out_dict['codebook'] = self.vis_single_code()
        # <<< WAVELET-VQ CHANGE 16 END
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
        # >>> WAVELET-VQ CHANGE 24 BEGIN: retain a few resumable numbered checkpoints.
        # The upstream latest save has no optimizer state, so numbered saves are
        # required for --auto_resume. Prune only this experiment's old numbered files.
        keep = int(self.opt.get('logger', {}).get('keep_last_checkpoints', 0))
        if current_iter >= 0 and keep > 0 and getattr(self.get_bare_model(self.net_g), 'wavelet_vq', False):
            state_dir = self.opt['path']['training_states']
            iterations = sorted(int(name[:-6]) for name in os.listdir(state_dir)
                                if name.endswith('.state') and name[:-6].isdigit())
            for old_iter in iterations[:-keep]:
                for old_path in (osp.join(state_dir, f'{old_iter}.state'),
                                 osp.join(self.opt['path']['models'], f'net_g_{old_iter}.pth'),
                                 osp.join(self.opt['path']['models'], f'net_d_{old_iter}.pth')):
                    if osp.isfile(old_path):
                        os.remove(old_path)
        # <<< WAVELET-VQ CHANGE 24 END
