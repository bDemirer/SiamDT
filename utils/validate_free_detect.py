"""free_detect'in (trackers/auto_detect.py) Anti-UAV410 test setindeki ilk-kare
tespit kalitesini olcen bagimsiz bir dogrulama betigi. Egitim/agirlik
degistirmez; sadece mevcut checkpoint'in template-siz tespit kalitesini
raporlar. Kullanim: `python utils/validate_free_detect.py`.
"""
import init_paths
import libs.data as data
import libs.ops as ops
from trackers.auto_detect import default_preprocess, free_detect
from trackers.siamdt_tracking import SiamDTTracker


def compute_iou(box_a, box_b):
    xa0, ya0, xa1, ya1 = box_a
    xb0, yb0, xb1, yb1 = box_b
    ix0, iy0 = max(xa0, xb0), max(ya0, yb0)
    ix1, iy1 = min(xa1, xb1), min(ya1, yb1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (xa1 - xa0) * (ya1 - ya0)
    area_b = (xb1 - xb0) * (yb1 - yb0)
    return inter / (area_a + area_b - inter)


def main(cfg_file='configs/siamdt_swin_tiny_sgd.py',
         ckp_file='checkpoints/siamdt_swin_tiny_sgd.pth',
         root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/',
         num_sequences=10, iou_thr=0.5):
    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(cfg_file, ckp_file, transforms)
    dataset = data.UAVtir(root_dir=root_dir, subset='test')

    hits, total = 0, 0
    for i in range(min(num_sequences, len(dataset))):
        img_files, target = dataset[i]
        gt = target['anno'][0]  # ilk karenin gt_rect'i, [x1,y1,x2,y2]
        img = ops.read_image(img_files[0])
        img_tensor, img_metas = default_preprocess(transforms, img, tracker.device)
        candidates = free_detect(tracker.model, img_tensor, img_metas)

        total += 1
        top1_iou = compute_iou(candidates[0].bbox, gt) if candidates else 0.0
        if candidates and top1_iou >= iou_thr:
            hits += 1
        print(f'[{i}] candidates={len(candidates)} top1_iou={top1_iou:.3f}')

    recall = hits / total if total else 0.0
    print(f'\nRecall@IoU{iou_thr} (ilk kare, {total} sekans): {recall:.3f}')


if __name__ == '__main__':
    main()
