# Optical Flow Tabanlı Motion Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SiamDT'nin otomatik-başlatma/yeniden-tespit akışına (`AutoInitTracker`), ego-motion telafili, optical-flow tabanlı bir hareket (motion) önceliği eklemek — hedef kaybolduğunda `free_detect` adaylarını "nereye gitmiş olabilir" tahminine göre yeniden sıralamak için.

**Architecture:** Yeni bir `optical_flow/` paketinde (cv2/numpy'a fonksiyon-içi bağımlı, saf-mantık kısmı bağımlılıksız) flow hesaplama + ego-motion çıkarma + EMA ile yumuşatılmış hız tahmini (`VelocityTracker`). `trackers/auto_detect.py`'deki `AutoInitTracker`, her `TRACKING` karesinde bu tahmini günceller, her `NEED_DETECT` karesinde ekstrapolasyonla bir konum tahmini üretip adayları buna göre yeniden sıralar (`_rerank_with_motion`). Bkz. onaylanmış tasarım: `docs/superpowers/specs/2026-08-15-optical-flow-motion-prior-design.md`.

**Tech Stack:** Python, OpenCV (cv2, klasik Farneback optical flow — eğitim gerektirmez), NumPy, stdlib `unittest` (saf-mantık testleri için).

## Global Constraints

- `VelocityTracker` (`optical_flow/motion_prior.py`), hem `velocity_fn` hem `to_gray_fn`'i **enjekte edilebilir** parametre olarak alır (varsayılan: gerçek cv2-tabanlı implementasyonlar). Bu, tasarım belgesindeki "`auto_detect.py`'nin `to_gray`'i kendisi çağırır" fikrinden **daha iyi bir çözüm** olarak plan yazımı sırasında bulundu: `VelocityTracker.update(img, bbox)` artık ham görüntüyü alıp griye çevirmeyi **kendi içinde** yapıyor, `trackers/auto_detect.py` bu sayede `optical_flow` paketinden **hiç import etmiyor** — motion_tracker tamamen duck-typed (`update(img,bbox)`/`predict(bbox,n)`/`reset()`). Tasarım belgesindeki asıl amaç (her TRACKING karesinde ego-motion telafili hız takibi) korunuyor, sadece "griye kim çeviriyor" detayı değişti. Bu değişiklik hem `optical_flow`'un test edilebilirliğini (Task 2, `to_gray_fn` enjekte edilerek) hem de `auto_detect.py`'nin taşınabilirliğini (artık gerçekten sıfır sibling/paket importu) iyileştiriyor.
- Ağır bağımlılıklar (`cv2`, `numpy`) **fonksiyon içi (local) import** — `optical_flow/motion_prior.py`'nin saf-mantık kısımları (`extrapolate_bbox`, `_smooth_velocity`, `VelocityTracker` enjekte edilmiş fake'lerle) bu değerlendirme ortamında da (cv2/numpy kurulu değil) test edilebilir kalır.
- Varsayılan sabitler: `smoothing=0.7`, rerank `weight=0.15`, rerank `scale=50.0`, min flow örneği eşiği `25` piksel — hepsi parametre olarak dışa açık, gerçek veride kalibrasyon gerektirir (tasarım belgesi Bölüm 4).
- `trackers/siamdt_rcnn.py`'ye ve önceki plandan (`2026-08-14-otomatik-tespit-entegrasyonu.md`) gelen dosyalara **bu planda dokunulmuyor** — sadece `trackers/auto_detect.py` genişletiliyor (geriye dönük uyumlu: `motion_tracker=None` varsayılan, verilmezse eski davranış birebir korunur).
- Bu değerlendirme ortamında `cv2`/`numpy`/`torch` kurulu değil (önceki planda da doğrulandı) — Task 1, 2, 4, 5'teki testler stdlib-only olduğu için bu ortamda çalıştırılabilir; Task 3, 6, 7 gerçek ortamda **manuel doğrulama** gerektirir.

---

### Task 1: `extrapolate_bbox` ve `_smooth_velocity` — saf mantık

**Files:**
- Create: `optical_flow/__init__.py`
- Create: `optical_flow/motion_prior.py`
- Create: `tests/test_motion_prior.py`

**Interfaces:**
- Produces: `extrapolate_bbox(bbox: list[float], velocity: tuple[float,float], frames_elapsed: float) -> list[float]`, `_smooth_velocity(old: tuple[float,float], new: tuple[float,float], smoothing: float) -> tuple[float,float]`.

- [ ] **Step 1: `optical_flow/__init__.py`'yi boş oluştur**

İçerik: boş dosya (sadece paketleşme için).

- [ ] **Step 2: Testi yaz (henüz başarısız olacak)**

`tests/test_motion_prior.py`:

```python
import unittest

from optical_flow.motion_prior import _smooth_velocity, extrapolate_bbox


class TestExtrapolateBbox(unittest.TestCase):
    def test_zero_frames_elapsed_returns_same_bbox(self):
        result = extrapolate_bbox([0, 0, 10, 10], (5.0, -3.0), 0)
        self.assertEqual(result, [0.0, 0.0, 10.0, 10.0])

    def test_shifts_bbox_by_velocity_times_frames(self):
        result = extrapolate_bbox([0, 0, 10, 10], (2.0, -1.0), 3)
        self.assertEqual(result, [6.0, -3.0, 16.0, 7.0])

    def test_preserves_bbox_size(self):
        result = extrapolate_bbox([10, 20, 30, 50], (1.0, 1.0), 5)
        width = result[2] - result[0]
        height = result[3] - result[1]
        self.assertAlmostEqual(width, 20.0)
        self.assertAlmostEqual(height, 30.0)


class TestSmoothVelocity(unittest.TestCase):
    def test_smoothing_one_keeps_old(self):
        result = _smooth_velocity((3.0, 4.0), (10.0, -10.0), smoothing=1.0)
        self.assertEqual(result, (3.0, 4.0))

    def test_smoothing_zero_takes_new(self):
        result = _smooth_velocity((3.0, 4.0), (10.0, -10.0), smoothing=0.0)
        self.assertEqual(result, (10.0, -10.0))

    def test_smoothing_half_averages(self):
        result = _smooth_velocity((0.0, 0.0), (10.0, 0.0), smoothing=0.5)
        self.assertEqual(result, (5.0, 0.0))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 3: Testi çalıştırıp başarısız olduğunu doğrula**

Run (repo kökünden): `python -m unittest tests.test_motion_prior -v`
Expected: `ModuleNotFoundError: No module named 'optical_flow.motion_prior'` — FAIL.

- [ ] **Step 4: `optical_flow/motion_prior.py`'yi oluştur**

```python
"""Optical flow tabanli, ego-motion telafili hedef hizi tahmini.

Agir bagimliliklar (cv2, numpy) fonksiyon ici (local) import edilir; boylece
saf mantik kismi (extrapolate_bbox, _smooth_velocity, VelocityTracker -
enjekte edilmis fake'lerle) bu paketler kurulu olmayan bir ortamda da test
edilebilir. Bu dosya kasitli olarak trackers/* gibi sibling dosyalara
bagli degil - bagimsiz kopyalanabilir olsun diye.
"""


def extrapolate_bbox(bbox, velocity, frames_elapsed):
    """bbox'i, velocity (piksel/kare) ile frames_elapsed kare kadar ileri
    kaydirir. Kutu boyutu degismez, sadece konum kayar."""
    dx, dy = velocity
    shift_x = dx * frames_elapsed
    shift_y = dy * frames_elapsed
    x1, y1, x2, y2 = bbox
    return [x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y]


def _smooth_velocity(old, new, smoothing):
    """Uzel hareketli ortalama (EMA): smoothing=1 -> tamamen eski deger
    korunur, smoothing=0 -> tamamen yeni degere atlanir."""
    old_x, old_y = old
    new_x, new_y = new
    return (
        smoothing * old_x + (1 - smoothing) * new_x,
        smoothing * old_y + (1 - smoothing) * new_y,
    )
```

- [ ] **Step 5: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_motion_prior -v`
Expected: 6 test de PASS.

- [ ] **Step 6: Commit**

```bash
git add optical_flow/__init__.py optical_flow/motion_prior.py tests/test_motion_prior.py
git commit -m "feat: add extrapolate_bbox/_smooth_velocity pure-logic motion math"
```

---

### Task 2: `VelocityTracker` — enjekte edilebilir durum makinesi

**Files:**
- Modify: `optical_flow/motion_prior.py` (Task 1'in üzerine ekleme)
- Modify: `tests/test_motion_prior.py` (Task 1'in üzerine ekleme)

**Interfaces:**
- Consumes: `extrapolate_bbox`, `_smooth_velocity` (Task 1).
- Produces: `VelocityTracker(smoothing=0.7, velocity_fn=None, to_gray_fn=None)` — metotlar: `.update(img, bbox)`, `.predict(last_bbox, frames_elapsed) -> list[float]`, `.reset()`. `velocity_fn(prev_gray, curr_gray, bbox) -> tuple[float,float] | None` ve `to_gray_fn(img) -> Any` enjekte edilebilir (varsayılanları Task 3'te tanımlanacak `_default_velocity_fn`/`to_gray` — bu Task'ta sadece isim olarak referans veriliyor, testler her zaman kendi fake'lerini enjekte ettiği için henüz tanımlanmamış olmaları sorun değil).

- [ ] **Step 1: Testi `tests/test_motion_prior.py`'ye ekle**

Dosyanın sonuna:

```python
from optical_flow.motion_prior import VelocityTracker


class TestVelocityTracker(unittest.TestCase):
    def test_first_update_does_not_call_velocity_fn(self):
        calls = []

        def fake_velocity_fn(prev_gray, curr_gray, bbox):
            calls.append((prev_gray, curr_gray, bbox))
            return (1.0, 2.0)

        vt = VelocityTracker(
            smoothing=0.7, velocity_fn=fake_velocity_fn,
            to_gray_fn=lambda img: img)
        vt.update('frame0', [0, 0, 10, 10])

        self.assertEqual(calls, [])

    def test_second_update_calls_velocity_fn_and_applies_ema(self):
        vt = VelocityTracker(
            smoothing=0.5, velocity_fn=lambda p, c, b: (10.0, 0.0),
            to_gray_fn=lambda img: img)
        vt.update('frame0', [0, 0, 10, 10])
        vt.update('frame1', [0, 0, 10, 10])

        self.assertAlmostEqual(vt._velocity[0], 5.0)
        self.assertAlmostEqual(vt._velocity[1], 0.0)

    def test_none_velocity_keeps_previous(self):
        results = iter([(5.0, 5.0), None])
        vt = VelocityTracker(
            smoothing=0.5, velocity_fn=lambda p, c, b: next(results),
            to_gray_fn=lambda img: img)
        vt.update('frame0', [0, 0, 1, 1])
        vt.update('frame1', [0, 0, 1, 1])  # velocity -> (2.5, 2.5)
        vt.update('frame2', [0, 0, 1, 1])  # fn None doner, degismemeli

        self.assertAlmostEqual(vt._velocity[0], 2.5)
        self.assertAlmostEqual(vt._velocity[1], 2.5)

    def test_predict_extrapolates_using_current_velocity(self):
        vt = VelocityTracker(
            smoothing=0.0, velocity_fn=lambda p, c, b: (2.0, -1.0),
            to_gray_fn=lambda img: img)
        vt.update('frame0', [0, 0, 10, 10])
        vt.update('frame1', [0, 0, 10, 10])

        predicted = vt.predict([0, 0, 10, 10], frames_elapsed=3)

        self.assertEqual(predicted, [6.0, -3.0, 16.0, 7.0])

    def test_reset_clears_state(self):
        vt = VelocityTracker(
            smoothing=0.0, velocity_fn=lambda p, c, b: (2.0, -1.0),
            to_gray_fn=lambda img: img)
        vt.update('frame0', [0, 0, 10, 10])
        vt.update('frame1', [0, 0, 10, 10])
        vt.reset()

        self.assertEqual(vt._velocity, (0.0, 0.0))
        predicted = vt.predict([0, 0, 10, 10], frames_elapsed=5)
        self.assertEqual(predicted, [0.0, 0.0, 10.0, 10.0])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Testi çalıştırıp başarısız olduğunu doğrula**

Run: `python -m unittest tests.test_motion_prior -v`
Expected: `TestVelocityTracker` testleri `ImportError: cannot import name 'VelocityTracker'` ile FAIL. Task 1'deki 6 test PASS olmaya devam eder.

- [ ] **Step 3: `VelocityTracker`'ı `optical_flow/motion_prior.py`'nin sonuna ekle**

```python
class VelocityTracker:
    """Her TRACKING karesinde .update(img, bbox) ile beslenir; ic tarafta
    ham goruntuyu griye cevirip (to_gray_fn) bir onceki griyle karsilastirir,
    ego-motion telafili hedef hizini (velocity_fn) EMA ile yumusatir. Hem
    velocity_fn hem to_gray_fn enjekte edilebilir - boylece cv2/numpy
    kurulu olmadan da test edilebilir, varsayilanlari (gercek cv2 tabanli
    implementasyonlar) Task 3'te eklenecek _default_velocity_fn/to_gray'e
    baglanir."""

    def __init__(self, smoothing=0.7, velocity_fn=None, to_gray_fn=None):
        self.smoothing = smoothing
        self._velocity_fn = velocity_fn if velocity_fn is not None else _default_velocity_fn
        self._to_gray_fn = to_gray_fn if to_gray_fn is not None else to_gray
        self._prev_gray = None
        self._velocity = (0.0, 0.0)

    def update(self, img, bbox):
        gray = self._to_gray_fn(img)
        if self._prev_gray is not None:
            new_velocity = self._velocity_fn(self._prev_gray, gray, bbox)
            if new_velocity is not None:
                self._velocity = _smooth_velocity(self._velocity, new_velocity, self.smoothing)
        self._prev_gray = gray

    def predict(self, last_bbox, frames_elapsed):
        return extrapolate_bbox(last_bbox, self._velocity, frames_elapsed)

    def reset(self):
        self._prev_gray = None
        self._velocity = (0.0, 0.0)
```

- [ ] **Step 4: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_motion_prior -v`
Expected: 11 test de PASS (Task 1'deki 6 + Task 2'deki 5).

- [ ] **Step 5: Commit**

```bash
git add optical_flow/motion_prior.py tests/test_motion_prior.py
git commit -m "feat: add VelocityTracker with injectable velocity_fn/to_gray_fn"
```

---

### Task 3: cv2/numpy-tabanlı gerçek implementasyonlar

**Files:**
- Modify: `optical_flow/motion_prior.py` (Task 2'nin üzerine ekleme)

**Interfaces:**
- Produces: `to_gray(img)`, `estimate_flow(prev_gray, curr_gray)`, `estimate_ego_motion(flow) -> tuple[float,float]`, `estimate_target_velocity(flow, bbox, ego_motion) -> tuple[float,float] | None`, `_default_velocity_fn(prev_gray, curr_gray, bbox) -> tuple[float,float] | None`. Bunlar `VelocityTracker`'ın (Task 2) varsayılanlarıdır.

**Neden bu fonksiyonlar için otomatik test yok:** `cv2`/`numpy` bu değerlendirme ortamında kurulu değil (Global Constraints). Doğrulaması gerçek ortamda, Task 6'daki betikle yapılacak.

- [ ] **Step 1: `optical_flow/motion_prior.py`'nin sonuna ekle**

```python
_MIN_FLOW_SAMPLES = 25
_BBOX_PADDING_RATIO = 0.5
_MIN_PADDING_PX = 4


def to_gray(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_flow(prev_gray, curr_gray):
    """Klasik Farneback yogun optical flow - egitim gerektirmez."""
    import cv2
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def estimate_ego_motion(flow):
    """Tum karenin medyan akisi = kamera hareketinin kaba tahmini (hedef
    goruntude cok kucuk bir alan kapladigi icin medyan arka plana/kameraya
    ait harekete hakim olur)."""
    import numpy as np
    fx = flow[..., 0].reshape(-1)
    fy = flow[..., 1].reshape(-1)
    return (float(np.median(fx)), float(np.median(fy)))


def estimate_target_velocity(flow, bbox, ego_motion):
    """Kutu icindeki (hafifce genisletilmis) medyan akis - ego_motion =
    hedefin gorece hizi. Kutu/orneklem cok kucukse None doner (VelocityTracker
    bu karede guncelleme yapmaz, onceki hizi korur)."""
    import numpy as np
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(bw * _BBOX_PADDING_RATIO, _MIN_PADDING_PX)
    pad_y = max(bh * _BBOX_PADDING_RATIO, _MIN_PADDING_PX)
    px1 = max(0, int(x1 - pad_x))
    py1 = max(0, int(y1 - pad_y))
    px2 = min(w, int(x2 + pad_x))
    py2 = min(h, int(y2 + pad_y))
    region = flow[py1:py2, px1:px2]
    if region.size == 0 or region.shape[0] * region.shape[1] < _MIN_FLOW_SAMPLES:
        return None
    fx = region[..., 0].reshape(-1)
    fy = region[..., 1].reshape(-1)
    ego_x, ego_y = ego_motion
    return (float(np.median(fx)) - ego_x, float(np.median(fy)) - ego_y)


def _default_velocity_fn(prev_gray, curr_gray, bbox):
    try:
        flow = estimate_flow(prev_gray, curr_gray)
        ego = estimate_ego_motion(flow)
        return estimate_target_velocity(flow, bbox, ego)
    except Exception:
        return None
```

- [ ] **Step 2: Bu ortamdaki testlerin hâlâ geçtiğini doğrula**

Run: `python -m unittest tests.test_motion_prior -v`
Expected: 11 test de hâlâ PASS (yeni eklenenler local-import olduğu için, çağrılmadıkça modül import'unu bozmaz).

- [ ] **Step 3: Manuel doğrulama (gerçek ortamda, cv2/numpy kurulu makinede)**

```python
import cv2
from optical_flow.motion_prior import to_gray, estimate_flow, estimate_ego_motion, estimate_target_velocity

img0 = cv2.imread('<ardisik iki kareden birincisi>.jpg')
img1 = cv2.imread('<ikincisi>.jpg')
g0, g1 = to_gray(img0), to_gray(img1)
flow = estimate_flow(g0, g1)
ego = estimate_ego_motion(flow)
vel = estimate_target_velocity(flow, [100, 100, 130, 130], ego)
print('ego-motion:', ego, 'target velocity:', vel)
```
Expected: Hata fırlatmadan çalışır, `ego` ve `vel` iki elemanlı sayı çiftleri olarak döner.

- [ ] **Step 4: Commit**

```bash
git add optical_flow/motion_prior.py
git commit -m "feat: add cv2/numpy-based flow, ego-motion, and velocity estimation"
```

---

### Task 4: `_rerank_with_motion` — saf mantık (auto_detect.py)

**Files:**
- Modify: `trackers/auto_detect.py` (önceki plandan mevcut dosyanın üzerine ekleme)
- Modify: `tests/test_auto_detect.py` (önceki plandan mevcut dosyanın üzerine ekleme)

**Interfaces:**
- Consumes: `Candidate` (önceki plan, Task 1).
- Produces: `_rerank_with_motion(candidates: list[Candidate], predicted_bbox: list[float], weight=0.15, scale=50.0) -> list[Candidate]`.

- [ ] **Step 1: Testi `tests/test_auto_detect.py`'ye ekle**

Dosyanın sonuna:

```python
from auto_detect import _rerank_with_motion


class TestRerankWithMotion(unittest.TestCase):
    def test_boosts_candidate_near_prediction(self):
        candidates = [
            Candidate(bbox=[100, 100, 110, 110], score=0.5),
            Candidate(bbox=[0, 0, 10, 10], score=0.48),
        ]
        result = _rerank_with_motion(candidates, [0, 0, 10, 10], weight=0.15, scale=50.0)

        self.assertEqual(result[0].bbox, [0, 0, 10, 10])

    def test_does_not_override_much_higher_score(self):
        candidates = [
            Candidate(bbox=[100, 100, 110, 110], score=0.9),
            Candidate(bbox=[0, 0, 10, 10], score=0.31),
        ]
        result = _rerank_with_motion(candidates, [0, 0, 10, 10], weight=0.15, scale=50.0)

        self.assertEqual(result[0].bbox, [100, 100, 110, 110])

    def test_bonus_decays_with_distance(self):
        candidates = [
            Candidate(bbox=[0, 0, 10, 10], score=0.5),
            Candidate(bbox=[1000, 1000, 1010, 1010], score=0.5),
        ]
        result = _rerank_with_motion(candidates, [0, 0, 10, 10], weight=0.15, scale=50.0)

        self.assertEqual(result[0].bbox, [0, 0, 10, 10])
        self.assertAlmostEqual(result[1].score, 0.5)

    def test_empty_candidates_returns_empty(self):
        result = _rerank_with_motion([], [0, 0, 10, 10], weight=0.15, scale=50.0)
        self.assertEqual(result, [])
```

(Not: `if __name__ == '__main__': unittest.main()` bloğu dosyanın sonunda zaten var — yeni test sınıfını o bloktan **önce** ekle.)

- [ ] **Step 2: Testi çalıştırıp başarısız olduğunu doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: `TestRerankWithMotion` testleri `ImportError: cannot import name '_rerank_with_motion'` ile FAIL. Diğer testler PASS olmaya devam eder.

- [ ] **Step 3: `_rerank_with_motion`'ı `trackers/auto_detect.py`'ye ekle**

`free_detect` fonksiyonunun hemen altına ekle:

```python
def _rerank_with_motion(candidates, predicted_bbox, weight=0.15, scale=50.0):
    """candidate.score'a, predicted_bbox'a merkez-mesafesi temelli bir bonus
    ekler, sonra skora gore yeniden siralar. weight kasitli olarak kucuk
    tutulur (tipik 0-1 skor araligina gore) - motion prior yanlis
    yonlendirse bile acik ara daha yuksek skorlu bir adayi geride
    birakamaz, sadece yakin skorlu adaylar arasinda ayirt edici olur."""
    def _center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    px, py = _center(predicted_bbox)
    reranked = []
    for c in candidates:
        cx, cy = _center(c.bbox)
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        bonus = weight * max(0.0, 1.0 - dist / scale)
        reranked.append(Candidate(bbox=c.bbox, score=c.score + bonus))
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked
```

- [ ] **Step 4: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: Tüm testler PASS (önceki plandan gelen 11 + bu Task'taki 4 = 15).

- [ ] **Step 5: Commit**

```bash
git add trackers/auto_detect.py tests/test_auto_detect.py
git commit -m "feat: add _rerank_with_motion candidate reranking"
```

---

### Task 5: `AutoInitTracker` entegrasyonu

**Files:**
- Modify: `trackers/auto_detect.py` (`AutoInitTracker.__init__`, `._try_detect`, `._run_loop`)
- Modify: `tests/test_auto_detect.py`

**Interfaces:**
- Consumes: `_rerank_with_motion` (Task 4), `free_detect` (önceki plan), bir `motion_tracker` nesnesi (`VelocityTracker` ile aynı arayüz — `.update(img,bbox)`, `.predict(bbox,n)`, `.reset()` — duck-typed, `trackers/auto_detect.py` **`optical_flow`'dan hiç import etmiyor**, bkz. Global Constraints).
- Produces: `AutoInitTracker.__init__`'e yeni parametreler: `motion_tracker=None, motion_weight=0.15, motion_scale=50.0`. Davranış değişikliği: `motion_tracker` verilmezse (varsayılan) **eski davranış birebir korunur**.

- [ ] **Step 1: Testi `tests/test_auto_detect.py`'ye ekle**

`TestRerankWithMotion` sınıfının altına, `if __name__ == '__main__':` bloğundan önce:

```python
class FakeMotionTracker:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.update_calls = []
        self.reset_calls = 0

    def update(self, img, bbox):
        self.update_calls.append((img, bbox))

    def predict(self, last_bbox, frames_elapsed):
        return self.predictions.pop(0)

    def reset(self):
        self.reset_calls += 1


class TestAutoInitTrackerMotionIntegration(unittest.TestCase):
    @patch('auto_detect.free_detect')
    def test_motion_tracker_updates_every_tracking_frame(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[0, 0, 1, 1], score=0.9)]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.9)  # lost_thr=0.8 uzerinde -> TRACKING'te kal
        motion = FakeMotionTracker(predictions=[])
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            motion_tracker=motion)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        # frame0: NEED_DETECT (motion.update cagrilmaz), frame1-2: TRACKING
        self.assertEqual(len(motion.update_calls), 2)

    @patch('auto_detect.free_detect')
    def test_predicted_bbox_used_to_rerank_on_redetect(self, mock_detect):
        mock_detect.side_effect = [
            [Candidate(bbox=[0, 0, 1, 1], score=0.9)],
            [Candidate(bbox=[5, 5, 6, 6], score=0.5),
             Candidate(bbox=[50, 50, 51, 51], score=0.48)],
        ]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)  # dusuk skor -> NEED_DETECT'e gec
        motion = FakeMotionTracker(predictions=[[5, 5, 6, 6]])
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            motion_tracker=motion, patience=1)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        self.assertEqual(tracker.init_calls[-1], [5, 5, 6, 6])

    @patch('auto_detect.free_detect')
    def test_reset_called_only_on_genuine_redetect(self, mock_detect):
        mock_detect.side_effect = [
            [Candidate(bbox=[0, 0, 1, 1], score=0.9)],
            [Candidate(bbox=[5, 5, 6, 6], score=0.9)],
        ]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)
        motion = FakeMotionTracker(predictions=[[0, 0, 1, 1]])
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            motion_tracker=motion, patience=1)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        # frame0'daki soguk-baslangic init() reset tetiklemez (henuz son
        # bilinen kutu yoktu); sadece frame2'deki gercek yeniden-tespit tetikler.
        self.assertEqual(motion.reset_calls, 1)

    @patch('auto_detect.free_detect')
    def test_works_without_motion_tracker(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[1, 2, 3, 4], score=0.9)]
        tracker = FakeTracker(update_script=[])
        model = FakeModel()
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None))

        bboxes = wrapper._run_loop(['frame0'])

        self.assertEqual(bboxes, [[1, 2, 3, 4]])
```

- [ ] **Step 2: Testi çalıştırıp başarısız olduğunu doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: Yeni 4 test, `AutoInitTracker.__init__()`'in `motion_tracker` parametresini kabul etmemesi nedeniyle `TypeError` ile FAIL. Diğer testler PASS olmaya devam eder.

- [ ] **Step 3: `AutoInitTracker`'ı güncelle**

`__init__`'i şu şekilde değiştir:

```python
    def __init__(self, tracker, model, transforms, preprocess_fn=None,
                 lost_thr=0.8, patience=1, score_thr=0.3, max_candidates=5,
                 motion_tracker=None, motion_weight=0.15, motion_scale=50.0):
        self.tracker = tracker
        self.model = model
        self.transforms = transforms
        self._preprocess_fn = preprocess_fn
        self.lost_thr = lost_thr
        self.patience = patience
        self.score_thr = score_thr
        self.max_candidates = max_candidates
        self.motion_tracker = motion_tracker
        self.motion_weight = motion_weight
        self.motion_scale = motion_scale
```

`_try_detect`'i şu şekilde değiştir (yeni bir `predicted_bbox` parametresi eklendi):

```python
    def _try_detect(self, img, predicted_bbox=None):
        img_tensor, img_metas = self._preprocess(img)
        candidates = free_detect(
            self.model, img_tensor, img_metas,
            score_thr=self.score_thr, max_candidates=self.max_candidates)
        if predicted_bbox is not None and candidates:
            candidates = _rerank_with_motion(
                candidates, predicted_bbox,
                weight=self.motion_weight, scale=self.motion_scale)
        return candidates
```

`_run_loop`'u tamamen şununla değiştir:

```python
    def _run_loop(self, images):
        """Saf orkestrasyon mantigi. images: onceden yuklenmis kare listesi
        (turu _preprocess_fn'in kabul ettigi turle ayni olmali). Donus:
        her karenin kutusunu iceren list[list[float]]."""
        state = 'NEED_DETECT'
        low_score_streak = 0
        last_known_bbox = None
        frames_since_loss = 0
        bboxes = []
        for img in images:
            if state == 'NEED_DETECT':
                predicted = None
                if self.motion_tracker is not None and last_known_bbox is not None:
                    predicted = self.motion_tracker.predict(last_known_bbox, frames_since_loss)
                    frames_since_loss += 1
                candidates = self._try_detect(img, predicted_bbox=predicted)
                if candidates:
                    self.tracker.init(img, candidates[0].bbox)
                    state = 'TRACKING'
                    low_score_streak = 0
                    if self.motion_tracker is not None and last_known_bbox is not None:
                        self.motion_tracker.reset()
                    last_known_bbox = None
                    bboxes.append(candidates[0].bbox)
                else:
                    bboxes.append(bboxes[-1] if bboxes else [0.0, 0.0, 0.0, 0.0])
            else:
                bbox, _up_flag = self.tracker.update(img)
                # bbox GPU torch.Tensor olabilir (onceki plandaki gerekce
                # aynen gecerli) - once duz listeye cevir.
                bbox = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                bboxes.append(bbox)
                if self.motion_tracker is not None:
                    self.motion_tracker.update(img, bbox)
                score = getattr(self.model, '_last_score', 0.0)
                if score < self.lost_thr:
                    low_score_streak += 1
                    if low_score_streak >= self.patience:
                        state = 'NEED_DETECT'
                        low_score_streak = 0
                        last_known_bbox = bbox
                        frames_since_loss = 0
                else:
                    low_score_streak = 0
        return bboxes
```

- [ ] **Step 4: Testi çalıştırıp geçtiğini doğrula**

Run: `python -m unittest tests.test_auto_detect -v`
Expected: Tüm testler PASS (15 + bu Task'taki 4 = 19).

- [ ] **Step 5: Commit**

```bash
git add trackers/auto_detect.py tests/test_auto_detect.py
git commit -m "feat: wire VelocityTracker-compatible motion prior into AutoInitTracker"
```

---

### Task 6: `utils/validate_motion_prior.py` — doğrulama betiği

**Files:**
- Create: `utils/validate_motion_prior.py`

**Interfaces:**
- Consumes: `optical_flow.motion_prior.to_gray/estimate_flow/estimate_ego_motion/estimate_target_velocity` (Task 3), `libs.data.UAVtir` (mevcut).
- Produces: Konsola (a) kare-başı flow hesaplama süresi, (b) normal takip sırasında tahmin edilen konumun bir sonraki gerçek kutuya ortalama merkez hatası (piksel) basan bir betik. Tasarım belgesi Bölüm 4'teki sabitlerin kalibrasyonuna karar verdirir.

- [ ] **Step 1: Betiği oluştur**

```python
"""optical_flow/motion_prior.py'nin gercek Anti-UAV410 verisindeki maliyetini
(kare-basi sure) ve dogrulugunu (tahmin edilen sonraki-kare konumunun
gercek kutuya ortalama merkez hatasi, piksel) olcen bagimsiz bir dogrulama
betigi. Kullanim: `python utils/validate_motion_prior.py`.
"""
import time

import init_paths
import libs.data as data
import libs.ops as ops
from optical_flow.motion_prior import (
    estimate_ego_motion, estimate_flow, estimate_target_velocity, to_gray,
)


def center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def main(root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/',
         num_sequences=5, max_frames_per_seq=30):
    dataset = data.UAVtir(root_dir=root_dir, subset='test')

    total_time = 0.0
    total_frames = 0
    center_errors = []

    for i in range(min(num_sequences, len(dataset))):
        img_files, target = dataset[i]
        anno = target['anno']
        n = min(max_frames_per_seq, len(img_files) - 1)

        prev_gray = None
        for f in range(n):
            img = ops.read_image(img_files[f])
            gray = to_gray(img)

            if prev_gray is not None:
                begin = time.time()
                flow = estimate_flow(prev_gray, gray)
                ego = estimate_ego_motion(flow)
                velocity = estimate_target_velocity(flow, anno[f], ego)
                elapsed = time.time() - begin
                total_time += elapsed
                total_frames += 1

                if velocity is not None:
                    vx, vy = velocity
                    px, py = center(anno[f])
                    predicted_center = (px + vx, py + vy)
                    actual_center = center(anno[f + 1])
                    err = ((predicted_center[0] - actual_center[0]) ** 2 +
                           (predicted_center[1] - actual_center[1]) ** 2) ** 0.5
                    center_errors.append(err)

            prev_gray = gray

        print(f'[{i}] sekans tamamlandi ({n} kare)')

    avg_time_ms = (total_time / total_frames * 1000) if total_frames else 0.0
    avg_err = (sum(center_errors) / len(center_errors)) if center_errors else float('nan')
    print(f'\nKare-basi flow suresi: {avg_time_ms:.2f} ms ({total_frames} kare)')
    print(f'Ortalama merkez hatasi (1 kare ileri tahmin): {avg_err:.2f} piksel '
          f'({len(center_errors)} olcum)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Manuel doğrulama (gerçek ortamda)**

Run: `python utils/validate_motion_prior.py`
(Gerekirse `main(...)` çağrısına kendi `root_dir` yolunu geç.)
Expected: Hata fırlatmadan tamamlanır, kare-başı süre (ms) ve ortalama merkez hatası (piksel) basar. Süre çok yüksekse (ör. gerçek-zamanlılık hedefini aşıyorsa) veya hata çok büyükse, tasarım belgesi Bölüm 4'teki sabitlerin (özellikle `smoothing`) yeniden kalibre edilmesi gerekir.

- [ ] **Step 3: Commit**

```bash
git add utils/validate_motion_prior.py
git commit -m "test: add motion prior timing/accuracy validation script"
```

---

### Task 7: `auto_tracking_test_demo.py`'ye bağla + regresyon kontrolü

**Files:**
- Modify: `auto_tracking_test_demo.py` (önceki plandan mevcut dosya)

**Interfaces:**
- Consumes: `VelocityTracker` (Task 2/3), `AutoInitTracker` (Task 5, `motion_tracker` parametresi).

- [ ] **Step 1: `auto_tracking_test_demo.py`'yi güncelle**

Dosyanın importlarına ekle:
```python
from optical_flow.motion_prior import VelocityTracker
```

`auto_tracker = AutoInitTracker(...)` satırını şununla değiştir:
```python
    auto_tracker = AutoInitTracker(
        tracker, tracker.model, transforms,
        lost_thr=0.8, patience=1,
        motion_tracker=VelocityTracker(smoothing=0.7))
```

- [ ] **Step 2: Manuel doğrulama (gerçek ortamda)**

Run: `python auto_tracking_test_demo.py`
Expected: Önceki plandaki (motion prior'sız) `auto_tracking_test_demo.py` çalıştırmasıyla aynı şekilde hata vermeden tamamlanır; `results/`/`reports/` altına yazar.

- [ ] **Step 3: Regresyon kontrolü**

Run: `python tracking_test_demo.py` (hâlâ değişmemiş, orijinal giriş noktası)
Expected: Bu plan öncesiyle **birebir aynı** sonuç — çünkü `trackers/siamdt_tracking.py` ve `trackers/siamdt_rcnn.py`'ye bu planda hiç dokunulmadı.

- [ ] **Step 4: Commit**

```bash
git add auto_tracking_test_demo.py
git commit -m "feat: wire VelocityTracker motion prior into end-to-end demo"
```

---

## Plan Sonrası Notlar

- Task 6'nın ölçtüğü kare-başı süre ve ortalama hata, tasarım belgesi Bölüm 4'teki sabitlerin
  (`smoothing`, `weight`, `scale`, min piksel eşiği) kalibrasyonu için kullanılmalı.
- Süre çok yüksek çıkarsa (gerçek-zamanlılık hedefini aşıyorsa), flow'u her karede değil belirli
  bir örnekleme sıklığında (ör. 2 karede bir) çalıştırmak bir sonraki iyileştirme adımı olabilir —
  bu planın kapsamında değil.
