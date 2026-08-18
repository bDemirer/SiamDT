import math

import torch
import numpy as np

from libs import Tracker
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model
from copy import deepcopy

__all__ = ['SiamDTTracker']


class SiamDTTracker(Tracker):

    def __init__(self, cfg_file, ckp_file, transforms, name_suffix='', visualize=False):
        name = 'siamdt'
        if name_suffix:
            name = name_suffix
        super(SiamDTTracker, self).__init__(
            name=name, is_deterministic=True, visualize=visualize)
        self.transforms = transforms

        # build config
        cfg = Config.fromfile(cfg_file)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        self.cfg = cfg

        # build model
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(
            model, ckp_file, map_location='cpu')
        model.CLASSES = ('object',)

        # GPU usage
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = model.to(self.device)
        
        self.prev_bbox = None

    def _compute_iou(self, box1, box2):
        """box1, box2: [x1, y1, x2, y2] (ltrb).

        DUZELTME: onceki surum box1/box2'yi [x,y,w,h] sanip x1+w1 seklinde
        yeniden 'kose' hesapliyordu -- ama model._process_gallary zaten
        [x1,y1,x2,y2,score] (ltrb) donduruyor (bkz. debug_bbox_format.py
        dogrulamasi), bu yuzden o hesap kutuyu yanlislikla buyutuyordu.
        Ayrica eski kod 'min(y2, yb2) if "y2" in locals() else ...' seklinde
        anlamsiz bir kontrol iceriyordu (y2 her zaman locals()'ta oldugu
        icin bu dal hep calisiyordu, ustelik y2 burada box2'nin KENDI
        y-koordinatiydi, ya2 degil) -- bu satir da kaldirildi.
        """
        xa1, ya1, xa2, ya2 = box1
        xb1, yb1, xb2, yb2 = box2

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area2 = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @torch.no_grad()
    def init(self, img, bbox):
        self.model.eval()
        self.prev_bbox = np.copy(bbox)

        # prepare query data
        img_meta = {'ori_shape': img.shape}
        bboxes = np.expand_dims(bbox, axis=0)
        img, img_meta, bboxes = \
            self.transforms._process_query(img, img_meta, bboxes)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        # initialize the modulator
        self.model._process_query(img, [bboxes], [img_meta])

    @torch.no_grad()
    def update(self, img, **kwargs):
        self.model.eval()

        # prepare gallary data
        img_meta = {'ori_shape': img.shape}
        img, img_meta, _ = \
            self.transforms._process_gallary(img, img_meta, None)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)

        # get detections
        results, up_flag = self.model._process_gallary(
            img, [img_meta], rescale=True, **kwargs)

        if not kwargs.get('return_all', False):
            if self.prev_bbox is not None and len(results) > 0:
                if isinstance(results, torch.Tensor):
                    res_np = results.cpu().numpy()
                else:
                    res_np = np.array(results)
                boxes = res_np[:, :4]  # [x1, y1, x2, y2] (ltrb -- DUZELTILDI)
                scores = res_np[:, -1]  # Skorlar

                # DUZELTME: prev_bbox de ltrb -- genislik/yukseklik ve
                # merkez, kose koordinatlarindan CIKARILARAK hesaplaniyor
                # (eskiden dogrudan x2,y2'yi w,h sanip kullaniyordu).
                prev_x1, prev_y1, prev_x2, prev_y2 = self.prev_bbox
                prev_w = prev_x2 - prev_x1
                prev_h = prev_y2 - prev_y1
                prev_center = np.array([
                    (prev_x1 + prev_x2) / 2.0, (prev_y1 + prev_y2) / 2.0])

                scored_candidates = []
                for i, box in enumerate(boxes):
                    bx1, by1, bx2, by2 = box
                    bw = bx2 - bx1
                    bh = by2 - by1
                    curr_center = np.array([
                        (bx1 + bx2) / 2.0, (by1 + by2) / 2.0])

                    dist = np.linalg.norm(curr_center - prev_center)
                    max_allowable_dist = max(prev_w, prev_h) * 2.5

                    if dist > max_allowable_dist:
                        continue

                    scale_ratio_w = bw / prev_w
                    scale_ratio_h = bh / prev_h
                    if scale_ratio_w > 2.5 or scale_ratio_h > 2.5 or scale_ratio_w < 0.2 or scale_ratio_h < 0.2:
                        continue

                    iou = self._compute_iou(self.prev_bbox, box)
                    combined_score = scores[i] + (0.8 * iou)
                    scored_candidates.append((combined_score, i))

                if len(scored_candidates) > 0:
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_ind = scored_candidates[0][1]
                else:
                    if len(res_np) > 0:
                        max_ind = res_np[:, -1].argmax()
                        raw_score = res_np[max_ind, -1]
                        
                        if raw_score > 0.30:
                            best_ind = max_ind
                        else:
                            best_ind = res_np[:, -1].argmax()
                    else:
                        best_ind = 0

                self.model._last_selection_mode = 'reranked' if len(scored_candidates) > 0 else 'fallback'
                # --- Hafıza Güncellemesi İçin NumPy Koordinatları (ltrb) ---
                selected_box = np.copy(boxes[best_ind])  # [x1,y1,x2,y2]

                # DUZELTME: genislik/yukseklik kose koordinatlarindan
                # cikariliyor, kirpma sonrasi merkez SABIT tutularak
                # kose koordinatlarina geri cevriliyor (eskiden x2,y2
                # sutunlarina dogrudan genislik/yukseklik degeri
                # yaziliyordu, bu da kutuyu bozuyordu).
                curr_w = selected_box[2] - selected_box[0]
                curr_h = selected_box[3] - selected_box[1]
                max_w, min_w = prev_w * 1.5, prev_w * 0.50
                max_h, min_h = prev_h * 1.5, prev_h * 0.50

                clipped_w = np.clip(curr_w, min_w, max_w)
                clipped_h = np.clip(curr_h, min_h, max_h)

                cx = (selected_box[0] + selected_box[2]) / 2.0
                cy = (selected_box[1] + selected_box[3]) / 2.0
                selected_box[0] = cx - clipped_w / 2.0
                selected_box[2] = cx + clipped_w / 2.0
                selected_box[1] = cy - clipped_h / 2.0
                selected_box[3] = cy + clipped_h / 2.0

                self.prev_bbox = np.copy(selected_box)

                # --- DOĞRUDAN ORİJİNAL TENSÖR DİLİMİ DÖNDÜRÜLÜYOR ---
                return results[best_ind, :4], up_flag
            else:
                # İlk frame veya boş sonuç durumu için güvenli dönüş
                if len(results) > 0:
                    max_ind = results[:, -1].argmax()
                    self.prev_bbox = np.copy(results[max_ind, :4].cpu().numpy() if isinstance(results, torch.Tensor) else results[max_ind, :4])
                    return results[max_ind, :4], up_flag
                else:
                    return results, up_flag
        else:
            return results, up_flag
