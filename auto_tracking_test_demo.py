import init_paths
import libs.data as data
from trackers import *
from trackers.auto_detect import AutoInitTracker

# python auto_tracking_test_demo.py
# Farki tracking_test_demo.py'den: init_bbox ground-truth'tan degil,
# free_detect() ile otomatik bulunuyor; hedef kaybolursa yeniden tetikleniyor.

if __name__ == '__main__':
    cfg_file = 'configs/siamdt_swin_tiny_sgd.py'
    ckp_file = 'checkpoints/siamdt_swin_tiny_sgd.pth'
    name_suffix = cfg_file[8:-3] + '_auto'
    selected_seq = 'ALL'

    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(
        cfg_file, ckp_file, transforms, name_suffix=name_suffix)
    auto_tracker = AutoInitTracker(
        tracker, tracker.model, transforms,
        lost_thr=0.8, patience=1)
    # Tracker.forward_test'in beklediği `name`/`is_deterministic` gibi
    # alanları AutoInitTracker taşımadığı için evaluator raporlamasında
    # sarılan tracker'ın adı kullanılır:
    auto_tracker.name = tracker.name
    auto_tracker.is_deterministic = tracker.is_deterministic

    evaluators = [
        data.EvaluatorUAVtir(
            root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/',
            subset='test')]

    for e in evaluators:
        e.run(auto_tracker, selected_seq=selected_seq)
