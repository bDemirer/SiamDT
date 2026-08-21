import init_paths
import argparse
import copy
import os
import os.path as osp
import time
import sys
from types import ModuleType

class MockAmp:
    @staticmethod
    def initialize(model, optimizer, opt_level=None):
        return model, optimizer
    @staticmethod
    def master_params(optimizer):
        return optimizer.param_groups
    @staticmethod
    def scale_loss(loss, optimizer):
        class ContextManager:
            def __enter__(self):
                return loss
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return ContextManager()
    @staticmethod
    def state_dict():
        return {}
    @staticmethod
    def load_state_dict(state_dict):
        pass
    
class MockApex(ModuleType):
    def __init__(self):
        super().__init__("apex")
        self.amp = MockAmp()

mock_apex_instance = MockApex()
sys.modules['apex'] = mock_apex_instance
sys.modules['apex.amp'] = mock_apex_instance.amp

import builtins
builtins.apex = mock_apex_instance

import torch
import mmcv.parallel._functions as mmcv_functions

import mmcv
from mmcv import Config, DictAction
from mmdet.utils import get_root_logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import libs.data as data
from trackers import *
import libs.ops as ops

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate a specific trained epoch using validation dataset on GPU')
    parser.add_argument(
        '--epoch',
        type=str,
        required=True,
        help='Epoch identifier, e.g., epoch_2 or 2')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Görsel kareleri video isimlerine göre ayrı klasörlerde kaydetmek için kullan')
    parser.add_argument(
        '--config',
        default='configs/siamdt_swin_tiny_sgd.py')
    parser.add_argument(
        '--work_dir',
        default='work_dirs/siamdt_swin_tiny_sgd')
    parser.add_argument(
        '--base_dataset',
        type=str,
        default='uavtir_train',
        help='dataset configuration name for validation'
    )
    parser.add_argument(
        '--base_transforms',
        type=str,
        default='extra_partial')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=[0],
        help='ids of gpus to use')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_ids[0] if args.gpu_ids else 0)

    if 'val' not in cfg.data:
        cfg.data.val = copy.deepcopy(cfg.data.train)

    if args.base_dataset is not None:
        cfg.data.val.base_dataset = args.base_dataset
    if args.base_transforms is not None:
        cfg.data.val.base_transforms = args.base_transforms

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.gpu_ids = args.gpu_ids

    # Gelen argümanı ayrıştırıyoruz (örn: epoch_2 veya 2)
    epoch_input = args.epoch.strip()
    if epoch_input.startswith('epoch_'):
        epoch_str = epoch_input
        epoch_idx = int(epoch_input.split('_')[1])
    else:
        epoch_idx = int(epoch_input)
        epoch_str = f"epoch_{epoch_idx}"

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'val_{epoch_str}_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Starting GPU validation for {epoch_str} in {args.work_dir}')

    ckp_file = osp.join(args.work_dir, f'{epoch_str}.pth')
    
    if not osp.exists(ckp_file):
        logger.error(f'Checkpoint bulunamadı: {ckp_file}')
        return

    logger.info(f'=== Validating {epoch_str} on GPU ({ckp_file}) ===')

    cfg.load_from = ckp_file
    name_suffix = osp.splitext(osp.basename(args.config))[0]
    visualize = True
    selected_seq = 'ALL'

    if args.save:
        base_save_dir = osp.join('validate_one_results', f'result_frames_{epoch_str}')
        os.makedirs(base_save_dir, exist_ok=True)
        logger.info(f'Görsel kayıt aktif: Kareler {base_save_dir}/[video_adi] klasörlerine kaydedilecek.')
        os.environ['ACTIVE_SAVE_DIR'] = base_save_dir
        os.environ['SAVE_BY_VIDEO'] = 'True'
        
    else:
        os.environ.pop('ACTIVE_SAVE_DIR', None)  # Değişkeni temizle
        os.environ['SAVE_BY_VIDEO'] = 'False'

    transforms = data.BasicPairTransforms(train=False)
    
    tracker = SiamDTTracker(
        args.config, ckp_file, transforms,
        name_suffix=name_suffix, visualize=visualize)
    
    custom_result_dir = osp.join('validate_one_results', f'result_{epoch_str}')
    custom_report_dir = osp.join('validate_one_results', 'reports')

    evaluators = [
        data.EvaluatorUAVtir(
            root_dir='dataset/Anti-UAV/data1', 
            subset='val',
            epoch=epoch_idx,
            result_dir=custom_result_dir,
            report_dir=custom_report_dir
        )
    ]

    for e in evaluators:
        e.run(tracker, selected_seq=selected_seq)
        
    logger.info(f'=== {epoch_str} Tamamlandı. Sonuçlar {custom_result_dir} klasöründe. ===')

if __name__ == '__main__':
    main()
