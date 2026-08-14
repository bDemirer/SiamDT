# Otomatik Tespit (Auto-Detect) Entegrasyonu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SiamDT'nin ilk karede zorunlu tuttuğu `init_bbox`'ı otomatikleştirmek ve takip sırasında hedef kaybolduğunda kendi kendine yeniden tespit yapabilmesini sağlamak — mevcut Siamese takip mantığına dokunmadan.

**Architecture:** Tek yeni dosyada (`trackers/auto_detect.py`) template gerektirmeyen bir tespit fonksiyonu (`free_detect`, mevcut `rpn_head`/`roi_head`'in ham/template-siz dalını kullanır) + bunu `SiamDTTracker`'ın public `init()`/`update()` API'siyle saran bir durum makinesi (`AutoInitTracker`). Mevcut dosyalara tek bir satırlık kanca eklenir (güven skorunu dışa açmak için). Bkz. onaylanmış tasarım: `docs/superpowers/specs/2026-08-14-otomatik-tespit-entegrasyonu-design.md`.

**Tech Stack:** Python, PyTorch, mmdetection (vendored, `libs/swintransformer`), stdlib `unittest` (pure-mantık testleri için — repo hedefi taşınabilirlik olduğundan pytest/numpy'a bağımlı olmayan testler tercih edildi).

## Global Constraints

- Tüm yeni mantık **tek dosyada** toplanır: `trackers/auto_detect.py` (onaylanmış tasarım, Bölüm 2/3.1).
- Mevcut dosyalara değişiklik **tek satırla** sınırlı: `trackers/siamdt_rcnn.py`'deki `_process_gallary` içine 1 satır (tasarım Bölüm 3.2). `trackers/siamdt_tracking.py`'ye **hiç dokunulmaz** — `update()`'in dönüş imzası (`bbox, up_flag`) değişmez, bu planın yazımı sırasında `libs/tracker.py:55`'teki mevcut 2'li unpack ile uyumluluğu korumak için tasarımdaki 3'lü dönüş fikrinden vazgeçildi; skor, `model._last_score` attribute'u üzerinden okunur.
- **Sıfır yeni ağırlık/eğitim** — `free_detect`, mevcut checkpoint'in zaten var olan ham (template'siz) dalını kullanır (Yaklaşım A).
- Yeni dosyanın ağır bağımlılıkları (`mmdet`, `libs.ops`, `numpy`, `torch`) **fonksiyon içi (local) import** olarak yazılır — böylece dosyanın saf mantık kısımları (`_select_candidates`, `AutoInitTracker._run_loop`) bu ağır paketler kurulu olmayan bir ortamda da (bu değerlendirme ortamı dahil) import edilip test edilebilir.
- Varsayılan sabitler: `score_thr=0.3`, `lost_thr=0.8`, `patience=1`, `max_candidates=5` — hepsi `AutoInitTracker`/`free_detect` kurucu/parametrelerinde dışa açık.
- Bu değerlendirme ortamı bir git reposu değil (torch/numpy/pytest de kurulu değil) — Task'lardaki `git commit` adımları, planın gerçek/hedef ortamda (kullanıcının "asıl çalıştığım kod" dediği, git ile takip edilen kopya) çalıştırılması için yazılmıştır. Task 1 ve Task 3'teki testler stdlib-only olduğu için bu değerlendirme ortamında da çalıştırılabilir; Task 2, 4, 5, 6 gerçek model/checkpoint/veri gerektirdiğinden **manuel doğrulama** adımı olarak yazılmıştır.

---

### Task 1: `Candidate` ve `_select_candidates` — saf mantık, bağımlılıksız

**Files:**
- Create: `trackers/auto_detect.py`
- Test: `tests/test_auto_detect.py`

**Interfaces:**
- Produces: `Candidate` (namedtuple, alanlar: `bbox: list[float]` uzunluk 4, `score: float`), `_select_candidates(det_bboxes, score_thr: float, max_candidates: int) -> list[Candidate]`. `det_bboxes`, her satırı `[x1, y1, x2, y2, score]` olan bir dizi (plain `list[list[float]]` ya da `.tolist()` metoduna sahip herhangi bir nesne, ör. `torch.Tensor`).

- [ ] **Step 1: Testi yaz (henüz başarısız olacak)**

`tests/test_auto_detect.py` dosyasını oluştur:

```python
import unittest

from trackers.auto_detect import Candidate, _select_candidates


class TestSelectCandidates(unittest.TestCase):
    def test_filters_below_threshold(self):
        det_bboxes = [
            [0, 0, 10, 10, 0.9],
            [5, 5, 15, 15, 0.1],
        ]
        result = _select_candidates(det_bboxes, score_thr=0.3, max_candidates=5)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].score, 0.9)

    def test_sorts_descending_by_score(self):
        det_bboxes = [
            [0, 0, 10, 10, 0.4],
            [1, 1, 11, 11, 0.8],
            [2, 2, 12, 12, 0.6],
        ]
        result = _select_candidates(det_bboxes, score_thr=0.0, max_candidates=5)
        self.assertEqual([c.score for c in result], [0.8, 0.6, 0.4])

    def test_caps_at_max_candidates(self):
        det_bboxes = [[i, i, i + 1, i + 1, 1.0 - i * 0.01] for i in range(10)]
        result = _select_candidates(det_bboxes, score_thr=0.0, max_candidates=3)
        self.assertEqual(len(result), 3)

    def test_accepts_tensor_like_objects_via_tolist(self):
        class FakeTensor:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

        det_bboxes = FakeTensor([[0, 0, 10, 10, 0.7]])
        result = _select_candidates(det_bboxes, score_thr=0.3, max_candidates=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, [0, 0, 10, 10])

    def test_empty_input_returns_empty_list(self):
        result = _select_candidates([], score_thr=0.3, max_candidates=5)
        self.assertEqual(result, [])

    def test_candidate_is_namedtuple_with_bbox_and_score(self):
        c = Candidate(bbox=[1, 2, 3, 4], score=0.5)
        self.assertEqual(c.bbox, [1, 2, 3, 4])
        self.assertEqual(c.score, 0.5)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Testi çalıştırıp başarısız olduğunu doğrula**

Run (repo kökünden): `python -m unittest tests.test_auto_detect -v`
Expected: `ModuleNotFoundError: No module named 'trackers.auto_detect'` (dosya henüz yok) veya `ImportError` — FAIL.

- [ ] **Step 3: `trackers/auto_detect.py`'yi oluştur**

```python
"""Template gerektirmeyen (SiamDT'ye ozgu Siamese modullerden bagimsiz) tespit ve
otomatik-baslatma/yeniden-tespit orkestrasyonu.

Bu dosyadaki agir bagimliliklar (mmdet, torch, numpy, libs.ops) fonksiyon ici
(local) import edilir; boylece Candidate/_select_candidates/AutoInitTracker'in
saf mantik kismi bu paketler kurulu olmayan bir ortamda da import edilip test
edilebilir.
"""

from collections import namedtuple

Candidate = namedtuple('Candidate', ['bbox', 'score'])


def _select_candidates(det_bboxes, score_thr, max_candidates):
    """det_bboxes: her satiri [x1, y1, x2, y2, score] olan bir dizi (liste ya da
    .tolist() metoduna sahip bir nesne, ör. torch.Tensor). score_thr altinda
    kalanlar elenir, geri kalan skora gore azalan sekilde siralanip ilk
    max_candidates tanesi dondurulur."""
    rows = det_bboxes.tolist() if hasattr(det_bboxes, 'tolist') else list(det_bboxes)
    candidates = [
        Candidate(bbox=list(row[:4]), score=float(row[4]))
        for row in rows
        if row[4] >= score_thr
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]
```

- [ ] **Step 4: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: 6 test de PASS.

- [ ] **Step 5: Commit**

```bash
git add trackers/auto_detect.py tests/test_auto_detect.py
git commit -m "feat: add Candidate/_select_candidates for template-free detection"
```

---

### Task 2: `free_detect` — mevcut model üzerinden gerçek tespit çağrısı

**Files:**
- Modify: `trackers/auto_detect.py` (Task 1'in üzerine ekleme)

**Interfaces:**
- Consumes: `_select_candidates` (Task 1).
- Produces: `free_detect(model, img_tensor, img_metas, score_thr=0.3, max_candidates=5) -> list[Candidate]`, `default_preprocess(transforms, img, device) -> (img_tensor, img_metas)`. `model`: `SiamDTRCNN` örneği (`.extract_feat`, `.rpn_head`, `.roi_head`, `.test_cfg` özelliklerine sahip — bunlar mmdet `TwoStageDetector` temelinden gelir, SiamDT'ye özgü değildir). `img_metas`: tek elemanlı liste, `{'img_shape':..., 'scale_factor':...}` içeren dict.

**Neden bu fonksiyonlar için otomatik test yok:** `free_detect`, gerçek bir `SiamDTRCNN` örneği + gerçek Swin/FPN ağırlıkları gerektiriyor (torch/mmdet bu değerlendirme ortamında kurulu değil — Global Constraints). Doğrulaması Task 5'teki betikle gerçek checkpoint üzerinde yapılacak.

- [ ] **Step 1: `trackers/auto_detect.py`'nin sonuna ekle**

```python
def free_detect(model, img_tensor, img_metas, score_thr=0.3, max_candidates=5):
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
    return _select_candidates(det_bboxes, score_thr, max_candidates)


def default_preprocess(transforms, img, device):
    """SiamDTTracker.update()'teki (trackers/siamdt_tracking.py:65-78) on-isleme
    zinciriyle ayni: transforms._process_gallary + batch boyutu ekleme + cihaza
    tasima. free_detect'in bekledigi (img_tensor, img_metas) ciftini uretir."""
    img_meta = {'ori_shape': img.shape}
    img_tensor, img_meta, _ = transforms._process_gallary(img, img_meta, None)
    img_tensor = img_tensor.unsqueeze(0).contiguous().to(device, non_blocking=True)
    return img_tensor, [img_meta]
```

- [ ] **Step 2: Bu dosyanın hâlâ bağımlılıksız kısmının test edilebilir kaldığını doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: Task 1'deki 6 test hâlâ PASS (yeni eklenen `free_detect`/`default_preprocess` içindeki `from mmdet.core import bbox2roi` local import olduğu için, bu fonksiyonlar çağrılmadıkça modül import'u hata vermez).

- [ ] **Step 3: Manuel doğrulama (gerçek ortamda, torch/mmdet kurulu makinede)**

Gerçek checkpoint yüklü bir Python oturumunda:
```python
import init_paths
from trackers.siamdt_tracking import SiamDTTracker
from trackers.auto_detect import free_detect, default_preprocess
import libs.data as data
import libs.ops as ops

transforms = data.BasicPairTransforms(train=False)
tracker = SiamDTTracker('configs/siamdt_swin_tiny_sgd.py', 'checkpoints/siamdt_swin_tiny_sgd.pth', transforms)
img = ops.read_image('<Anti-UAV410 test sekansindan bir kare>.jpg')
img_tensor, img_metas = default_preprocess(transforms, img, tracker.device)
candidates = free_detect(tracker.model, img_tensor, img_metas)
print(candidates[:3])
```
Expected: Hata fırlatmadan çalışır, `Candidate(bbox=[...], score=...)` içeren bir liste döner (boş liste de geçerli bir sonuçtur, bu adımda "çöküyor mu" kontrol ediliyor — doğruluk Task 5'te ölçülecek).

- [ ] **Step 4: Commit**

```bash
git add trackers/auto_detect.py
git commit -m "feat: add free_detect template-free detection call"
```

---

### Task 3: `AutoInitTracker` — durum makinesi (NEED_DETECT / TRACKING)

**Files:**
- Modify: `trackers/auto_detect.py` (Task 2'nin üzerine ekleme)
- Modify: `tests/test_auto_detect.py` (Task 1'in üzerine ekleme)

**Interfaces:**
- Consumes: `free_detect` (Task 2, modül seviyesinde `trackers.auto_detect.free_detect` olarak mock'lanabilir), `Candidate` (Task 1), bir `tracker` nesnesi (`init(img, bbox)` ve `update(img) -> (bbox, up_flag)` metotlarına sahip — `SiamDTTracker` ile aynı arayüz), bir `model` nesnesi (`_last_score` attribute'una sahip olması beklenir, Task 4'te eklenecek).
- Produces: `AutoInitTracker(tracker, model, transforms, preprocess_fn=None, lost_thr=0.8, patience=1, score_thr=0.3, max_candidates=5)`, metotları `._run_loop(images) -> list[list[float]]` (saf mantık, test edilen kısım) ve `.forward_test(img_files, init_bbox=None, visualize=False) -> (bboxes: np.ndarray, times: np.ndarray)` (gerçek ortamda `EvaluatorUAVtir.run(tracker, ...)`'a doğrudan verilebilecek şekilde `libs/tracker.py:37`'deki `Tracker.forward_test` ile aynı imza).

- [ ] **Step 1: Testleri `tests/test_auto_detect.py`'ye ekle**

Dosyanın sonuna (importlara `patch`/`MagicMock` ekleyerek):

```python
from unittest.mock import patch

from trackers.auto_detect import AutoInitTracker, Candidate


class FakeTracker:
    def __init__(self, update_script):
        self.update_script = list(update_script)
        self.init_calls = []

    def init(self, img, bbox):
        self.init_calls.append(bbox)

    def update(self, img):
        return self.update_script.pop(0)


class FakeModel:
    def __init__(self, last_score=0.0):
        self._last_score = last_score


class TestAutoInitTrackerRunLoop(unittest.TestCase):
    @patch('trackers.auto_detect.free_detect')
    def test_calls_init_when_target_found(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[1, 2, 3, 4], score=0.9)]
        tracker = FakeTracker(update_script=[])
        model = FakeModel()
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None))

        bboxes = wrapper._run_loop(['frame0'])

        self.assertEqual(tracker.init_calls, [[1, 2, 3, 4]])
        self.assertEqual(bboxes, [[1, 2, 3, 4]])

    @patch('trackers.auto_detect.free_detect')
    def test_stays_in_need_detect_when_no_candidates(self, mock_detect):
        mock_detect.return_value = []
        tracker = FakeTracker(update_script=[])
        model = FakeModel()
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None))

        bboxes = wrapper._run_loop(['frame0', 'frame1'])

        self.assertEqual(tracker.init_calls, [])
        self.assertEqual(bboxes, [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    @patch('trackers.auto_detect.free_detect')
    def test_switches_to_need_detect_after_low_score_then_redetects(self, mock_detect):
        mock_detect.side_effect = [
            [Candidate(bbox=[0, 0, 1, 1], score=0.9)],
            [Candidate(bbox=[9, 9, 10, 10], score=0.9)],
        ]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)  # default lost_thr=0.8 altinda
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            patience=1)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        self.assertEqual(tracker.init_calls, [[0, 0, 1, 1], [9, 9, 10, 10]])

    @patch('trackers.auto_detect.free_detect')
    def test_patience_delays_switch_to_need_detect(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[0, 0, 1, 1], score=0.9)]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            patience=2)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        # patience=2: iki dusuk-skor karesi gerekiyor, bu yuzden 3 karelik
        # sekansta sadece frame0'daki ilk detect init() cagirir.
        self.assertEqual(len(tracker.init_calls), 1)

    @patch('trackers.auto_detect.free_detect')
    def test_high_score_resets_streak(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[0, 0, 1, 1], score=0.9)]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),  # dusuk skor -> streak=1
            ([0, 0, 1, 1], False),  # yuksek skor -> streak=0 (asagida ayarlanacak)
            ([0, 0, 1, 1], False),  # tekrar dusuk skor -> streak=1 (patience=2'yi tetiklemez)
        ])
        scores = iter([0.1, 0.9, 0.1])

        class ScriptedModel:
            @property
            def _last_score(self):
                return next(scores)

        wrapper = AutoInitTracker(
            tracker, ScriptedModel(), transforms=None,
            preprocess_fn=lambda img: (None, None), patience=2)

        wrapper._run_loop(['frame0', 'frame1', 'frame2', 'frame3'])

        self.assertEqual(len(tracker.init_calls), 1)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Testi çalıştırıp başarısız olduğunu doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: Yeni eklenen `TestAutoInitTrackerRunLoop` testleri `ImportError: cannot import name 'AutoInitTracker'` ile FAIL. Task 1'deki eski testler PASS olmaya devam eder.

- [ ] **Step 3: `AutoInitTracker`'ı `trackers/auto_detect.py`'nin sonuna ekle**

```python
class AutoInitTracker:
    """SiamDTTracker'in (veya init(img,bbox)/update(img) arayuzune sahip
    herhangi bir tracker'in) etrafini saran, template-siz free_detect ile
    otomatik baslatma ve kayip-hedef sonrasi yeniden-tespit yapan orkestrator.
    Mevcut tracker/model siniflarinin ic detaylarina dokunmaz."""

    def __init__(self, tracker, model, transforms, preprocess_fn=None,
                 lost_thr=0.8, patience=1, score_thr=0.3, max_candidates=5):
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

    def _run_loop(self, images):
        """Saf orkestrasyon mantigi. images: onceden yuklenmis kare listesi
        (turu _preprocess_fn'in kabul ettigi turle ayni olmali). Donus:
        her karenin kutusunu iceren list[list[float]]."""
        state = 'NEED_DETECT'
        low_score_streak = 0
        bboxes = []
        for img in images:
            if state == 'NEED_DETECT':
                candidates = self._try_detect(img)
                if candidates:
                    self.tracker.init(img, candidates[0].bbox)
                    state = 'TRACKING'
                    low_score_streak = 0
                    bboxes.append(candidates[0].bbox)
                else:
                    bboxes.append(bboxes[-1] if bboxes else [0.0, 0.0, 0.0, 0.0])
            else:
                bbox, _up_flag = self.tracker.update(img)
                # bbox, SiamDTTracker.update()'ten gelirken bir GPU torch.Tensor
                # olabilir (siamdt_tracking.py:46-47, CUDA varsa) - dogrudan
                # np.array()'e vermek CUDA tensor'de patlar, once duz listeye cevir.
                bboxes.append(bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
                score = getattr(self.model, '_last_score', 0.0)
                if score < self.lost_thr:
                    low_score_streak += 1
                    if low_score_streak >= self.patience:
                        state = 'NEED_DETECT'
                        low_score_streak = 0
                else:
                    low_score_streak = 0
        return bboxes

    def forward_test(self, img_files, init_bbox=None, visualize=False):
        """libs/tracker.py:37'deki Tracker.forward_test ile ayni imza/donus
        sozlesmesi - EvaluatorUAVtir.run(auto_tracker, ...) bu sinifi
        degistirmeden kabul edebilir. init_bbox kasitli olarak yok sayilir:
        bu sinifin butun amaci init_bbox'a ihtiyac duymamak."""
        import time

        import numpy as np
        import libs.ops as ops

        images = [ops.read_image(f) for f in img_files]
        begin = time.time()
        bboxes_list = self._run_loop(images)
        elapsed = time.time() - begin

        frame_num = len(img_files)
        times = np.full(frame_num, elapsed / max(frame_num, 1))
        return np.array(bboxes_list, dtype=float), times
```

- [ ] **Step 4: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: 11 test de PASS (Task 1'deki 6 + Task 3'teki 5).

- [ ] **Step 5: Commit**

```bash
git add trackers/auto_detect.py tests/test_auto_detect.py
git commit -m "feat: add AutoInitTracker state machine for auto-init and re-detection"
```

---

### Task 4: Güven skorunu dışa açan tek satırlık kanca

**Files:**
- Modify: `trackers/siamdt_rcnn.py` (`_process_gallary` metodu)

**Interfaces:**
- Produces: `SiamDTRCNN` örneklerinde `self._last_score: float` attribute'u — `AutoInitTracker._run_loop` (Task 3) `getattr(self.model, '_last_score', 0.0)` ile bunu okur.

**Not:** Bu değişiklik `trackers/siamdt_tracking.py`'ye **dokunmaz** — `SiamDTTracker.update()`'in dönüş imzası (`bbox, up_flag`) aynen kalır, `libs/tracker.py:55`'teki mevcut `current_box, up_flag = self.update(img)` unpack'i bozulmaz (Global Constraints'te açıklanan tasarım revizyonu).

- [ ] **Step 1: Anchor satırı bul**

Hedef koddaki `trackers/siamdt_rcnn.py` dosyasında şu metni ara (satır numarası farklı olabilir, metin araması yap):
```python
if tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9 and self.computeiou(det_bboxes[0, :-1], tra_bboxes[0, :-1])>0.8:
```
Bu koşul `_process_gallary` metodunun içinde, `up_flag` hesaplanmadan hemen önce yer alır.

- [ ] **Step 2: Bu satırın hemen üstüne skor saklama satırını ekle**

```python
        self._last_score = float(tra_bboxes[0, -1] + det_bboxes[0, -1])
        if tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9 and self.computeiou(det_bboxes[0, :-1], tra_bboxes[0, :-1])>0.8:
```

- [ ] **Step 3: Manuel doğrulama (gerçek ortamda)**

```python
tracker.init(img0, gt_bbox0)
bbox, up_flag = tracker.update(img1)
print(tracker.model._last_score)
```
Expected: Hata yok, `tracker.model._last_score` bir `float` değer içeriyor (tipik olarak 0-2 aralığında).

- [ ] **Step 4: Commit**

```bash
git add trackers/siamdt_rcnn.py
git commit -m "feat: expose confidence score via model._last_score"
```

---

### Task 5: `utils/validate_free_detect.py` — Yaklaşım A'nın doğrulama betiği

**Files:**
- Create: `utils/validate_free_detect.py`

**Interfaces:**
- Consumes: `free_detect`, `default_preprocess` (Task 2), `SiamDTTracker` (mevcut, `.model`/`.device` attribute'ları üzerinden), `libs.data.UAVtir` (mevcut dataset sınıfı).
- Produces: Konsola IoU>0.5 recall yüzdesi basan, tekrar çalıştırılabilir bir betik. Yeni bir public arayüz/tip tanımlamaz — Bölüm 5'teki (tasarım belgesi) "önce ölç, sonra entegre et" adımını karşılar, Task 6'nın gidip gitmeyeceğine karar verdirir.

- [ ] **Step 1: Betiği oluştur**

```python
"""free_detect'in (trackers/auto_detect.py) Anti-UAV410 test setindeki ilk-kare
tespit kalitesini olcen bagimsiz bir dogrulama betigi. Egitim/agirlik
degistirmez; sadece mevcut checkpoint'in template-siz tespit kalitesini
raporlar. Kullanim: `python utils/validate_free_detect.py`.
"""
import init_paths
import libs.data as data
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
        img = __import__('libs.ops', fromlist=['read_image']).read_image(img_files[0])
        img_tensor, img_metas = default_preprocess(transforms, img, tracker.device)
        candidates = free_detect(tracker.model, img_tensor, img_metas)

        total += 1
        if candidates and compute_iou(candidates[0].bbox, gt) >= iou_thr:
            hits += 1
        print(f'[{i}] candidates={len(candidates)} '
              f'top1_iou={compute_iou(candidates[0].bbox, gt) if candidates else 0.0:.3f}')

    recall = hits / total if total else 0.0
    print(f'\nRecall@IoU{iou_thr} (ilk kare, {total} sekans): {recall:.3f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Manuel doğrulama (gerçek ortamda)**

Run: `python utils/validate_free_detect.py`
(Gerekirse dosyanın en altındaki `main(...)` çağrısına `root_dir`/`cfg_file`/`ckp_file` parametrelerini kendi ortamına göre geçir.)
Expected: Hata fırlatmadan tamamlanır, her sekans için bir satır ve sonunda bir `Recall@IoU0.5` yüzdesi basar. Bu sayı **düşükse** (ör. <%50), tasarım belgesindeki Yaklaşım C'ye (ayrı `rpn_head_plain`) geçmeyi değerlendirin — bu betiğin amacı tam olarak bu kararı verdirmek.

- [ ] **Step 3: Commit**

```bash
git add utils/validate_free_detect.py
git commit -m "test: add free_detect recall validation script"
```

---

### Task 6: `auto_tracking_test_demo.py` — uçtan uca giriş noktası

**Files:**
- Create: `auto_tracking_test_demo.py` (mevcut `tracking_test_demo.py` ile aynı seviyede, repo kökünde)

**Interfaces:**
- Consumes: `AutoInitTracker` (Task 3), `SiamDTTracker` (mevcut), `data.EvaluatorUAVtir` (mevcut, değişmedi — `AutoInitTracker.forward_test`'in `Tracker.forward_test` ile aynı imzayı taşıması sayesinde değişiklik gerekmiyor).

- [ ] **Step 1: Betiği oluştur**

```python
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
```

- [ ] **Step 2: Manuel doğrulama (gerçek ortamda)**

Run: `python auto_tracking_test_demo.py`
Expected: `tracking_test_demo.py` ile aynı şekilde `results/`/`reports/` altına dosya yazar; farkı, `init_bbox`'ın ground-truth yerine `free_detect` çıktısından gelmesidir. Konsol çıktısında hata yoksa, birkaç sekansın sonucunu `tracking_test_demo.py`'nin (ground-truth init ile) ürettiği sonuçla IoU açısından karşılaştırarak genel tutarlılığı gözle kontrol edin.

- [ ] **Step 3: Regresyon kontrolü**

Run: `python tracking_test_demo.py` (değiştirilmemiş, orijinal giriş noktası)
Expected: Bu plan öncesi ile **birebir aynı** sonuçları üretir — çünkü `trackers/siamdt_tracking.py` hiç değişmedi ve `trackers/siamdt_rcnn.py`'deki tek satır sadece yeni bir attribute set ediyor, mevcut dönüş değerlerini/davranışı etkilemiyor.

- [ ] **Step 4: Commit**

```bash
git add auto_tracking_test_demo.py
git commit -m "feat: add end-to-end auto-detect tracking entry point"
```

---

## Plan Sonrası Notlar

- Task 5'in ölçtüğü recall yeterince yüksekse (kullanıcının kendi eşik kararı), plan burada tamamlanmış sayılır.
- Yetersizse, tasarım belgesindeki **Yaklaşım C** (ayrı `rpn_head_plain`, Anti-UAV410 üzerinde eğitim) ayrı bir spec/plan döngüsü olarak ele alınmalı — bu plan kapsamında değil.
