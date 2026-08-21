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

# Projeye özel importlar (Test/Değerlendirme için)
import libs.data as data
from trackers import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate all trained epochs using validation dataset on GPU')
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
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'val_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Starting GPU validation for all epochs in {args.work_dir}')

    for epoch_idx in range(1, 13):
        ckp_file = osp.join(args.work_dir, f'epoch_{epoch_idx}.pth')
        
        if not osp.exists(ckp_file):
            logger.warning(f'Checkpoint bulunamadı, atlanıyor: {ckp_file}')
            continue

        logger.info(f'=== Validating Epoch {epoch_idx} on GPU ({ckp_file}) ===')

        cfg.load_from = ckp_file
        name_suffix = osp.splitext(osp.basename(args.config))[0]
        visualize = True
        selected_seq = 'ALL'

        transforms = data.BasicPairTransforms(train=False)
        
        tracker = SiamDTTracker(
            args.config, ckp_file, transforms,
            name_suffix=name_suffix, visualize=visualize)

        evaluators = [
            data.EvaluatorUAVtir(
                root_dir='dataset/Anti-UAV/data1', 
                subset='val',
                epoch=epoch_idx,
                result_dir='validate_all_results',
                report_dir='validate_all_results'
            )
        ]

        for e in evaluators:
            e.run(tracker, selected_seq=selected_seq)
            
        logger.info(f'=== Epoch {epoch_idx} Tamamlandı ===')

if __name__ == '__main__':
    main()
