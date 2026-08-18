"""Template gerektirmeyen (SiamDT'ye ozgu Siamese modullerden bagimsiz) tespit ve
otomatik-baslatma/yeniden-tespit orkestrasyonu.

Bu dosyadaki agir bagimliliklar (mmdet, torch, numpy, libs.ops) fonksiyon ici
(local) import edilir; boylece Candidate/_select_candidates/AutoInitTracker'in
saf mantik kismi bu paketler kurulu olmayan bir ortamda da import edilip test
edilebilir. Bu dosya kasitli olarak hicbir sibling dosyaya (trackers/*, libs/*)
top-level import ile bagli degil - baska bir kod tabanina kopyalanip
yapistirilabilmesi icin.
"""

from collections import namedtuple

Candidate = namedtuple('Candidate', ['bbox', 'score'])


def _select_candidates(det_bboxes, score_thr, max_candidates, min_area=1.0):
    """det_bboxes: her satiri [x1, y1, x2, y2, score] olan bir dizi (liste ya da
    .tolist() metoduna sahip bir nesne, ör. torch.Tensor). score_thr altinda
    kalanlar elenir. min_area'dan kucuk (dejenere - sifir/negatif genislik ya
    da yukseklik) kutular da elenir - RPN'in goruntu kenarlarinda urettigi
    anlamsiz artefaktlari filtrelemek icin (bkz. Adim 4 debug bulgusu: bazi
    sekanslarda score_thr=0.0'da bile donen 'adaylarin' tamami sifir alanli
    kenar kutulariydi). Geri kalan skora gore azalan sekilde siralanip ilk
    max_candidates tanesi dondurulur."""
    rows = det_bboxes.tolist() if hasattr(det_bboxes, 'tolist') else list(det_bboxes)
    candidates = []
    for row in rows:
        if row[4] < score_thr:
            continue
        x1, y1, x2, y2 = row[:4]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue
        candidates.append(Candidate(bbox=list(row[:4]), score=float(row[4])))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def free_detect(model, img_tensor, img_metas, score_thr=0.06, max_candidates=5,
                 min_area=1.0):
    """Template gerektirmeden, tum goruntude aday kutu arar. model.rpn_head'e
    x_corr eklenmeden ham ozellik verilir; model.roi_head.bbox_head'in zaten
    template'siz proposal'larla egitilmis "ham dali" kullanilir (bkz. tasarim
    belgesi Bolum 4, Yaklasim A)."""
    from mmdet.core import bbox2roi

    x = model.extract_feat(img_tensor)
    proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
    rois = bbox2roi(proposal_list)
    bbox_feats = model.roi_head.bbox_roi_extractor(
        x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
    cls_score, bbox_pred = model.roi_head.bbox_head(bbox_feats)
    det_bboxes, _det_labels = model.roi_head.bbox_head.get_bboxes(
        rois, cls_score, bbox_pred,
        img_metas[0]['img_shape'], img_metas[0]['scale_factor'],
        rescale=True, cfg=model.test_cfg.rcnn)
    return _select_candidates(det_bboxes, score_thr, max_candidates, min_area)


def default_preprocess(transforms, img, device):
    """SiamDTTracker.update()'teki (trackers/siamdt_tracking.py:65-78) on-isleme
    zinciriyle ayni: transforms._process_gallary + batch boyutu ekleme + cihaza
    tasima. free_detect'in bekledigi (img_tensor, img_metas) ciftini uretir."""
    img_meta = {'ori_shape': img.shape}
    img_tensor, img_meta, _ = transforms._process_gallary(img, img_meta, None)
    img_tensor = img_tensor.unsqueeze(0).contiguous().to(device, non_blocking=True)
    return img_tensor, [img_meta]


class AutoInitTracker:
    """SiamDTTracker'in (veya init(img,bbox)/update(img) arayuzune sahip
    herhangi bir tracker'in) etrafini saran, template-siz free_detect ile
    otomatik baslatma ve kayip-hedef sonrasi yeniden-tespit yapan orkestrator.
    Mevcut tracker/model siniflarinin ic detaylarina dokunmaz."""

    def __init__(self, tracker, model, transforms, preprocess_fn=None,
             lost_thr=0.8, patience=1, score_thr=0.06, max_candidates=5):
        self.tracker = tracker
        self.model = model
        self.transforms = transforms
        self._preprocess_fn = preprocess_fn
        self.lost_thr = lost_thr
        self.patience = patience
        self.score_thr = score_thr
        self.max_candidates = max_candidates

    def _preprocess(self, img):
        if self._preprocess_fn is not None:
            return self._preprocess_fn(img)
        device = next(self.model.parameters()).device
        return default_preprocess(self.transforms, img, device)

    def _try_detect(self, img):
        img_tensor, img_metas = self._preprocess(img)
        return free_detect(
            self.model, img_tensor, img_metas,
            score_thr=self.score_thr, max_candidates=self.max_candidates)

    def _run_loop(self, images, on_frame=None):
        """Saf orkestrasyon mantigi. images: onceden yuklenmis kare listesi
        (turu _preprocess_fn'in kabul ettigi turle ayni olmali). Donus:
        her karenin kutusunu iceren list[list[float]].

        on_frame: opsiyonel, imzasi on_frame(frame_idx, img, bbox, up_flag)
        olan bir callable. Her kare islendikten SONRA cagirilir (ör.
        gorsellestirme/kayit icin) - mevcut testler bu parametreyi hic
        kullanmadigindan (None birakildigindan) davranislarini etkilemez."""
        state = 'NEED_DETECT'
        low_score_streak = 0
        bboxes = []
        for f, img in enumerate(images):
            if state == 'NEED_DETECT':
                candidates = self._try_detect(img)
                if candidates:
                    self.tracker.init(img, candidates[0].bbox)
                    state = 'TRACKING'
                    low_score_streak = 0
                    bbox = candidates[0].bbox
                    up_flag = True  # yeni baslatma = "guncellendi"
                else:
                    bbox = bboxes[-1] if bboxes else [0.0, 0.0, 0.0, 0.0]
                    up_flag = False
            else:
                bbox, up_flag = self.tracker.update(img)
                bbox = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                score = getattr(self.model, '_last_score', 0.0)
                selection_mode = getattr(self.model, '_last_selection_mode', None)

                if selection_mode == 'fallback':
                    state = 'NEED_DETECT'
                    low_score_streak = 0
                elif score < self.lost_thr:
                    low_score_streak += 1
                    if low_score_streak >= self.patience:
                        state = 'NEED_DETECT'
                        low_score_streak = 0

                else:
                    low_score_streak = 0

            bboxes.append(bbox)
            if on_frame is not None:
                on_frame(f, img, bbox, up_flag)
        return bboxes

    def forward_test(self, img_files, init_bbox=None, visualize=False,
                      gt_bboxes=None, **kwargs):
        """libs/tracker.py:37'deki Tracker.forward_test ile ayni imza/donus
        sozlesmesi - EvaluatorUAVtir.run(auto_tracker, ...) bu sinifi
        degistirmeden kabul edebilir. init_bbox kasitli olarak yok sayilir:
        bu sinifin butun amaci init_bbox'a ihtiyac duymamak. gt_bboxes de
        ayni sekilde yok sayilir. **kwargs, ileride gelebilecek baska ozel
        argumanlarin da sessizce yutulup crash'e sebep olmamasi icin.

        Gorsellestirme/kayit: sarilan self.tracker'in KENDI visualize/viz
        durumunu (Tracker.__init__'te visualize=True ise zaten kurulu olan
        visdom baglantisi) yeniden kullanir - yeni bir Visdom baglantisi
        ACMAZ, farkli bir kayit mekanizmasi UYDURMAZ. ops.show_image, ayni
        Tracker.forward_test'in cagirdigi sekilde (fig parametresi
        verilmeden, yani varsayilan/sabit pencere adiyla) cagirilir - bu
        da ayni kayit/goruntuleme davranisini (ör. ACTIVE_SAVE_DIR tabanli
        kare kaydi, varsa) DEGISTIRMEDEN korur ve tek, surekli guncellenen
        bir pencere ('video oynatimi' gorunumu) saglar. gt_bboxes verilirse
        (ve img_files ile ayni uzunluktaysa), GT (yesil) ve tahmin
        (kirmizi) BIRLIKTE ciziliyor - GT sadece kayit icin degil, gorsel
        karsilastirma icin de kullaniliyor artik."""
        import time

        import numpy as np
        import libs.ops as ops

        images = [ops.read_image(f) for f in img_files]

        do_visualize = getattr(self.tracker, 'visualize', False)
        viz = getattr(self.tracker, 'viz', None)
        gt_ok = gt_bboxes is not None and len(gt_bboxes) == len(img_files)

        on_frame = None
        if do_visualize and viz is not None:
            def on_frame(f, img, bbox, up_flag):
                if gt_ok:
                    ops.show_image(
                        img, [gt_bboxes[f], bbox], f, up_flag, viz,
                        colors=[(0, 255, 0), (0, 0, 255)])  # yesil=GT, kirmizi=tahmin
                else:
                    ops.show_image(img, bbox, f, up_flag, viz)

        begin = time.time()
        bboxes_list = self._run_loop(images, on_frame=on_frame)
        elapsed = time.time() - begin

        frame_num = len(img_files)
        times = np.full(frame_num, elapsed / max(frame_num, 1))
        return np.array(bboxes_list, dtype=float), times
