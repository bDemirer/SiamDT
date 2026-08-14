# SiamDT — Optical Flow Tabanlı Motion Prior Tasarımı

> Durum: Onaylandı (brainstorming oturumu)
> Önceki ilgili çalışma: `docs/superpowers/specs/2026-08-14-otomatik-tespit-entegrasyonu-design.md`
> (auto-detect/re-detection), `SIAMDT_GELISTIRME_ONERILERI.md` madde A5 (motion prior — Kalman
> alternatifi) ve o maddenin altındaki "Neden optical flow değil?" notu.

---

## 1. Problem ve Kapsam

`AutoInitTracker` (önceki tasarım) hedef kaybolduğunda `free_detect` ile tüm görüntüde yeniden
tarama yapıyor, ama **hiçbir hareket bilgisi kullanmıyor** — adaylar arasından sadece skora göre
seçim yapılıyor. `SIAMDT_GELISTIRME_ONERILERI.md` A5 maddesi bunu bir Kalman filtresiyle çözmeyi
öneriyordu; bu tasarım aynı ihtiyacı **optical flow tabanlı bir hız tahmini** ile çözüyor.

**Kapsam:**
1. Yeni, bağımsız bir modül (`optical_flow/motion_prior.py`) — flow hesaplama, ego-motion
   (kamera hareketi) telafisi, hedef hız tahmini, ekstrapolasyon.
2. `AutoInitTracker`'a entegrasyon — her `TRACKING` karesinde hız tahminini güncellemek, her
   `NEED_DETECT` karesinde bu tahmini `free_detect` adaylarını yeniden sıralamak için kullanmak.

**Kapsam dışı:** Öğrenilmiş (deep) optical flow (RAFT/FlowNet — "sıfır yeni ağırlık" ilkesine
aykırı), normal `TRACKING` akışındaki mevcut arka-plan-bastırma/IoU-konsensüs mantığının
değiştirilmesi (`trackers/siamdt_rcnn.py`'ye bu tasarımda dokunulmuyor).

## 2. Neden "her karede", neden dense/Farneback, neden ego-motion telafisi

Bu üç karar, brainstorming sırasında sırayla netleşti:

- **Her `TRACKING` karesinde çalıştırma:** Optical flow, **küçük yer değiştirme** varsayımına
  dayanır — ardışık kareler arasında güvenilir, uzun boşluklarda (ör. sadece kurtarma anında
  hesaplansaydı) güvenilmezdi. Ayrıca Farneback klasik/CPU-tabanlı bir yöntem olduğu için Swin
  Transformer forward geçişine kıyasla kare-başı maliyeti görece küçük — "her karede çalıştırma"
  fikri ilk bakışta pahalı görünse de, asıl kazanç (doğruluk) maliyetten (gerçek zamanlılık) daha
  ağır basıyor.
- **Dense Farneback (cv2), öğrenilmiş flow değil:** Eğitim gerektirmiyor, cv2 zaten muhtemel bir
  bağımlılık (görüntü IO için), taşınabilir tek dosya.
- **Ego-motion (kamera hareketi) telafisi baştan dahil:** Anti-UAV410 görüntüleri hareketli kamera
  platformundan çekiliyor — telafisiz flow, hedef hareketini kamera hareketinden ayıramaz
  (bkz. `SIAMDT_GELISTIRME_ONERILERI.md`, "Neden optical flow değil?" notu — bu tasarım o notta
  işaret edilen eksikliği gideriyor).

## 3. Mimari

### 3.1 Yeni klasör: `optical_flow/motion_prior.py`

**Ağır bağımlılıklı katman** (cv2/numpy, fonksiyon içi local import — `auto_detect.py`'deki
desenle aynı, taşınabilirlik için):

```python
def to_gray(img): ...                                  # cv2.cvtColor
def estimate_flow(prev_gray, curr_gray): ...            # cv2.calcOpticalFlowFarneback
def estimate_ego_motion(flow): ...                      # tum karenin medyan akisi
def estimate_target_velocity(flow, bbox, ego_motion): ...  # kutu ici medyan akis - ego_motion
```

`estimate_target_velocity`, kutuyu örnekleme öncesi hafifçe **genişletir** (padding) — çok küçük
hedeflerde (Anti-UAV410'da tipik) yeterli piksel örneği olsun diye. Genişletilmiş bölge yine de
çok küçükse (piksel sayısı eşik altındaysa) `None` döner — çağıran taraf (`VelocityTracker`) bunu
"bu karede güncelleme yok" olarak yorumlar.

**Saf mantık katmanı** (stdlib-only, cv2/numpy gerektirmez):

```python
def extrapolate_bbox(bbox, velocity, frames_elapsed): ...   # bbox + velocity * frames_elapsed
def _smooth_velocity(old, new, smoothing=0.7): ...           # EMA
```

**`VelocityTracker`** (aynı dosyada, durum tutan sınıf):

```python
class VelocityTracker:
    def __init__(self, smoothing=0.7):
        self.smoothing = smoothing
        self._prev_gray = None
        self._velocity = (0.0, 0.0)

    def update(self, gray, bbox):
        """Her TRACKING karesinde cagrilir. Onceki gri kare varsa flow hesaplar,
        ego-motion'i cikarir, EMA ile hiz tahminini gunceller. Flow/velocity
        hesaplanamazsa (cok kucuk kutu, ilk kare, hata) onceki hizi korur."""
        ...

    def predict(self, last_bbox, frames_elapsed):
        """NEED_DETECT'te cagrilir. last_bbox'i biriken hizla frames_elapsed
        kare kadar ileri ekstrapole eder."""
        return extrapolate_bbox(last_bbox, self._velocity, frames_elapsed)

    def reset(self):
        self._prev_gray = None
        self._velocity = (0.0, 0.0)
```

### 3.2 `AutoInitTracker` entegrasyonu (`trackers/auto_detect.py` genişletilir)

- `__init__`'e `motion_tracker=None` parametresi eklenir — `None` ise motion prior tamamen devre
  dışı kalır (geriye dönük uyumlu, `optical_flow` modülü hiç import edilmeden de çalışır).
- `_run_loop`'un `TRACKING` dalında: `bbox, _up_flag = self.tracker.update(img)`'den hemen sonra,
  `motion_tracker` varsa `motion_tracker.update(to_gray(img), bbox)` çağrılır (local import).
- `TRACKING → NEED_DETECT` geçişinde: `self._last_known_bbox = bbox`, `self._frames_since_loss = 0`
  saklanır.
- `NEED_DETECT` dalında, `_try_detect`'ten önce: `motion_tracker` ve `_last_known_bbox` varsa
  `predicted = motion_tracker.predict(self._last_known_bbox, self._frames_since_loss)` hesaplanır,
  `self._frames_since_loss += 1`. Yoksa `predicted = None`.
- `free_detect`'in döndürdüğü adaylar, `predicted is not None` ise `_rerank_with_motion(candidates,
  predicted, weight)` ile yeniden sıralanır (saf fonksiyon, `trackers/auto_detect.py` içine
  eklenir — cv2/numpy gerektirmez, sadece `candidate.bbox` üzerinde merkez-mesafesi hesaplar).
- Yeniden tespit başarılı olunca (`tracker.init()` çağrılınca): `motion_tracker.reset()`,
  `self._last_known_bbox = None`.

```python
def _rerank_with_motion(candidates, predicted_bbox, weight=0.15, scale=50.0):
    """candidate.score'a, predicted_bbox'a merkez-mesafesi temelli bir bonus ekler,
    sonra skora gore yeniden sirala. scale: bonusun ne kadar hizli sifira dustugunu
    belirleyen piksel olcegi (tunable)."""
    def center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    px, py = center(predicted_bbox)
    reranked = []
    for c in candidates:
        cx, cy = center(c.bbox)
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        bonus = weight * max(0.0, 1.0 - dist / scale)
        reranked.append(Candidate(bbox=c.bbox, score=c.score + bonus))
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked
```

**Not — bonus, birincil sinyali domine etmiyor:** `weight=0.15`, tipik `det_bboxes` skor aralığına
(0-1, softmax) göre **kasıtlı olarak küçük** tutuldu — maksimum bonus (mesafe=0 iken) bile açık ara
daha yüksek skorlu bir adayı geride bırakamaz, sadece skorları birbirine yakın adaylar arasında
ayırt edici olur. (İlk taslakta `weight=0.5` idi; öz-incelemede bunun uzak-skorlu bir adayı bile
öne geçirebileceği, yani "bonus baskın olmaz" iddiasıyla çeliştiği fark edilip düzeltildi.)

## 4. Varsayılan Sabitler (hepsi deneysel ayar gerektirir)

| Sabit | Varsayılan | Nerede |
|---|---|---|
| `smoothing` (EMA katsayısı) | 0.7 | `VelocityTracker.__init__` |
| `weight` (rerank bonus ağırlığı) | 0.15 | `_rerank_with_motion` |
| `scale` (rerank mesafe ölçeği, piksel) | 50.0 | `_rerank_with_motion` |
| Minimum piksel örneği eşiği (küçük kutu koruması) | 25 (5×5 padding sonrası) | `estimate_target_velocity` |

Bölüm 6'daki doğrulama betiği bu değerlerin gerçek veride kalibre edilmesi için kullanılacak —
tıpkı önceki tasarımdaki `lost_thr=0.8` gibi, buradaki değerler de makul başlangıç noktaları,
kesin doğru değil.

## 5. Kenar Durumları

| Durum | Davranış |
|---|---|
| Çok küçük hedef kutusu (birkaç piksel) | `estimate_target_velocity`, kutuyu padding ile genişletir; yine de yetersizse `None` döner, `VelocityTracker` önceki hızı korur. |
| `cv2` hedef ortamda yok/farklı | `motion_tracker=None` ile `AutoInitTracker` motion prior olmadan eskisi gibi çalışır — sert bağımlılık yok. |
| Uzun "kayıp" süresi | Ekstrapolasyon güvenilirliği düşer ama bonus küçük tutulduğu için (Bölüm 3.2 notu) en kötü ihtimalle nötr kalır, `free_detect` skorunu domine etmez. |
| Flow hesaplama hatası (bozuk kare, boyut uyuşmazlığı) | `VelocityTracker.update` hatayı yutar, önceki hızı korur, sekansın geri kalanını durdurmaz. |
| İlk kare (hiç hareket verisi yok) | `predicted=None`, mevcut `free_detect` davranışı değişmeden kullanılır. |

## 6. Test / Doğrulama Stratejisi

- **Stdlib-only, bu ortamda çalıştırılabilir:** `extrapolate_bbox`, `_smooth_velocity`,
  `_rerank_with_motion` — `unittest` ile test edilir (cv2/numpy gerekmez).
- **Gerçek ortamda manuel doğrulama:** `estimate_flow`/`estimate_ego_motion`/
  `estimate_target_velocity`/`VelocityTracker.update` gerçek görüntülerle.
- **Yeni doğrulama betiği — `utils/validate_motion_prior.py`:** `validate_free_detect.py` ile aynı
  ruhta:
  1. Birkaç test sekansında kare-başı flow hesaplama süresini ölçer (gerçek-zamanlılık maliyeti).
  2. Normal takip sırasında `predict()`'in bir sonraki gerçek kutuya ortalama merkez hatasını
     (piksel cinsinden) ölçer.
  Bu iki sayı, Bölüm 4'teki sabitlerin kalibrasyonuna ve motion prior'ın genel olarak faydalı olup
  olmadığına karar verdirecek.

## 7. Açık Riskler / Bilinmeyenler

- Farneback'in gerçek kare-başı maliyeti bu donanımda ölçülmedi — Bölüm 6, Adım 1 bunu netleştirecek.
- `weight`/`scale`/`smoothing` sabitleri veri setine özgü kalibrasyon gerektirir.
- `auto_detect.py`'nin "tek dosya, sıfır sibling import" ilkesi bu entegrasyonla kısmen bozuluyor —
  taşırken artık `optical_flow/` klasörü + güncellenmiş `auto_detect.py` birlikte kopyalanmalı
  (kullanıcı bu ödünleşimi bilerek onayladı).

---

*Bu belge, kullanıcıyla yapılan brainstorming oturumunun onaylanmış çıktısıdır. Sıradaki adım:
`writing-plans` becerisiyle uygulama planı çıkarmak.*
