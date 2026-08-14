# SiamDT — Otomatik Tespit Entegrasyonu: Uygulama Talimatları

> Bu belge, **asıl çalışılan (production) SiamDT kod tabanına** — burada geliştirilen kopyadan
> farklı/değişikliğe uğramış olabilecek bir kopyaya — "template-siz otomatik başlatma + kayıp-hedef
> sonrası yeniden-tespit" özelliğini elle entegre etmek için yazılmıştır. Claude (veya başka bir
> asistan) ile web üzerinden bu konuyu konuşurken referans olarak kullanılmak üzere **kendi başına
> yeterli** olacak şekilde hazırlanmıştır — önceki bir konuşmanın hatırlanmasına ihtiyaç duymaz.

Kod ve tam gerekçe: `https://github.com/bDemirer/SiamDT` (branch `main`).

---

## 1. Ne yapıyoruz, neden?

SiamDT bir **tracker**'dır (Siamese, template-tabanlı, global-search bir nesne dedektörü gibi
çalışır): ilk karede zorunlu bir başlangıç kutusuna (`init_bbox`) ihtiyaç duyar ve hedef tamamen
kaybolursa kendini toparlayamaz. Bu değişiklik iki şeyi ekliyor:

1. **İlk kare otomasyonu** — video başladığında hedefi (drone'u) sıfırdan bulup tracker'ı otomatik
   başlatmak, elle/ground-truth kutu vermeye gerek kalmadan.
2. **Kayıp-hedef sonrası yeniden-tespit** — takip sırasında güven skoru düşerse aynı mekanizma
   tekrar tetiklenip tracker'ı sıfırdan başlatır.

**Nasıl:** Mevcut, zaten eğitilmiş checkpoint'in `roi_head.bbox_head`'inde zaten var olan
**template'siz ("ham") dalı** kullanılıyor — bu dal, eğitim sırasında (Siamese eşleştirmeden
bağımsız olarak) `gt_bboxes_x`'e göre zaten denetimli eğitilmiş durumda. **Yeni ağırlık/eğitim
gerekmiyor.** Tek belirsizlik: `rpn_head`'in proposal kalitesinin template olmadan (hiç görmediği
bir girdi dağılımında) ne kadar iyi olduğu — bu yüzden Bölüm 4'te bir **doğrulama adımı** var.

Tam tasarım gerekçesi ve reddedilen alternatifler için:
`docs/superpowers/specs/2026-08-14-otomatik-tespit-entegrasyonu-design.md`
Görev görev uygulama planı (test kodu dahil) için:
`docs/superpowers/plans/2026-08-14-otomatik-tespit-entegrasyonu.md`

---

## 2. Ön koşullar

- Çalışan bir SiamDT ortamı: Python + PyTorch + mmdetection (vendored, `libs/swintransformer`)
  kurulu, `configs/*.py` + eğitilmiş bir `.pth` checkpoint mevcut.
- Anti-UAV410 (veya benzeri) test verisi, `libs/data.UAVtir`'in okuyabileceği formatta
  (`IR_label.json` + kareler).
- Hedef kod tabanı bu repodan **farklı olabilir** — bu yüzden aşağıdaki adımlar dosya içeriğini
  **satır numarasına göre değil, metin araması (anchor) ile** bulmaya dayanıyor.

---

## 3. Yapılacak Değişiklikler

### 3.1 Yeni dosyaları ekle (aynen kopyala)

Repodan şu 4 dosyayı **olduğu gibi** hedef koda kopyala (hiçbiri SiamDT'ye özgü iç detaylara değil,
mmdet'in standart `TwoStageDetector` API'sine dayanıyor — `extract_feat`, `rpn_head`, `roi_head`):

| Kaynak (bu repo) | Hedefte konulacak yer |
|---|---|
| `trackers/auto_detect.py` | `trackers/auto_detect.py` |
| `tests/test_auto_detect.py` | `tests/test_auto_detect.py` |
| `utils/validate_free_detect.py` | `utils/validate_free_detect.py` |
| `auto_tracking_test_demo.py` | repo kökü, `auto_tracking_test_demo.py` |

`trackers/auto_detect.py` **hiçbir sibling dosyaya import ile bağlı değil** (kasıtlı olarak) — tek
başına kopyalanıp yapıştırılabilir.

### 3.2 Mevcut dosyada tek satırlık değişiklik

`trackers/siamdt_rcnn.py` içinde, `_process_gallary` metodunun içinde şu metni ara:

```python
if tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9 and self.computeiou(det_bboxes[0, :-1], tra_bboxes[0, :-1])>0.8:
```

Bu satırın **hemen üstüne** ekle:

```python
self._last_score = float(tra_bboxes[0, -1] + det_bboxes[0, -1])
if tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9 and self.computeiou(det_bboxes[0, :-1], tra_bboxes[0, :-1])>0.8:
```

**Bu değişiklik başka hiçbir şeyi etkilemez** — `update()`'in dönüş imzası, mevcut takip davranışı
aynen kalır; sadece bir attribute (`model._last_score`) dışarı açılır.

### 3.3 Değiştirilmeyenler (önemli)

`trackers/siamdt_tracking.py`'ye **hiç dokunulmuyor** — `SiamDTTracker.update()`'in dönüş değeri
(`bbox, up_flag`) değişmiyor. Skor, dönüş değeriyle değil `model._last_score` attribute'u üzerinden
okunuyor (bkz. tasarım belgesi, Global Constraints). Eğer hedef kodda `update()` farklı bir imzaya
sahipse, `trackers/auto_detect.py` içindeki `AutoInitTracker._run_loop`'taki
`bbox, _up_flag = self.tracker.update(img)` satırını ona göre uyarlaman gerekir.

---

## 4. Doğrulama Adımları (sırayla)

### Adım 1 — Testler (bağımlılıksız, her ortamda çalışır)

```bash
python -m unittest tests.test_auto_detect -v
```
Beklenen: **11/11 test PASS**. Bu testler `torch`/`mmdet` gerektirmez (bu repoyu geliştirdiğim
ortamda da, torch kurulu olmadan, çalıştığını doğruladım).

### Adım 2 — `free_detect` çöküyor mu? (gerçek model/checkpoint gerekir)

```python
import init_paths
from trackers.siamdt_tracking import SiamDTTracker
from trackers.auto_detect import free_detect, default_preprocess
import libs.data as data
import libs.ops as ops

transforms = data.BasicPairTransforms(train=False)
tracker = SiamDTTracker('configs/<senin-config>.py', '<senin-checkpoint>.pth', transforms)
img = ops.read_image('<bir test karesi>.jpg')
img_tensor, img_metas = default_preprocess(transforms, img, tracker.device)
candidates = free_detect(tracker.model, img_tensor, img_metas)
print(candidates[:3])
```
Beklenen: Hata yok, `Candidate(bbox=[...], score=...)` içeren bir liste (boş liste de olabilir).

### Adım 3 — Skor kancası çalışıyor mu?

```python
tracker.init(img0, gt_bbox0)
bbox, up_flag = tracker.update(img1)
print(tracker.model._last_score)   # float, tipik olarak 0-2 arası
```

### Adım 4 — **En kritik adım: Recall ölçümü**

```bash
python utils/validate_free_detect.py
```
(Betiğin sonundaki `main(...)` çağrısına kendi `cfg_file`/`ckp_file`/`root_dir` yollarını geç.)

Çıktı, `Recall@IoU0.5 (ilk kare, N sekans): X.XXX` şeklinde bir sayı verir. **Bu sayı, bir sonraki
adıma karar verdiriyor** — bkz. Bölüm 5.

### Adım 5 — Uçtan uca + regresyon kontrolü

```bash
python auto_tracking_test_demo.py     # yeni, otomatik-başlatmalı akış
python tracking_test_demo.py          # mevcut, değişmemiş akış — sonuçlar ÖNCEKİYLE AYNI olmalı
```

---

## 5. Sonuç Değerlendirme / Karar Noktası

- **Recall (Adım 4) yeterince yüksekse** (kullanıcı kararı — ör. >%70-80 gibi bir eşik makul
  başlangıç noktası olabilir): entegrasyon tamamlanmış sayılır, `auto_tracking_test_demo.py`
  gerçek kullanım için hazırdır.
- **Recall düşükse:** `rpn_head`'in template olmadan (hiç görmediği bir girdi dağılımında) yeterince
  iyi proposal üretemediği anlamına gelir. Bu durumda tasarım belgesindeki **Yaklaşım C**'ye
  geçilmesi gerekir — orijinal `rpn_head`'e dokunmadan, ayrı ve sadece bu amaç için eğitilecek bir
  `rpn_head_plain` eklemek. Bu, **ayrı bir tasarım/plan döngüsü** gerektirir, bu belgenin kapsamında
  değildir.

---

## 6. Sorun Giderme (Troubleshooting)

| Belirti | Olası neden | Ne yap |
|---|---|---|
| `ImportError: cannot import name 'bbox2roi' from 'mmdet.core'` | Hedef koddaki mmdet sürümü farklı/vendored yapı farklı | `trackers/auto_detect.py` içindeki `free_detect`'te `from mmdet.core import bbox2roi` satırını, hedef kodun kendi importlarıyla (ör. `trackers/siamdt_rcnn.py`'nin tepesindeki import satırı) eşleştir. |
| `AttributeError: 'SiamDTRCNN' object has no attribute '_last_score'` | Bölüm 3.2'deki satır eklenmedi, ya da `update()` hiç çağrılmadan okunmaya çalışıldı | Anchor satırı tekrar ara (metin `_process_gallary` içinde farklı bir yerde olabilir — `up_flag` hesaplandığı yeri ara), satırı ekle. `_last_score`'u sadece en az bir `tracker.update()` çağrısından SONRA oku. |
| `free_detect` sürekli boş liste dönüyor | `score_thr=0.3` çok yüksek kalıyor olabilir, ya da Bölüm 5'teki "recall düşük" senaryosu | Önce `score_thr`'ı düşürüp (ör. 0.1) tekrar dene — eğer o zaman adaylar çıkıyorsa sadece eşik kalibrasyonu sorunu; hiç çıkmıyorsa Yaklaşım C gerekebilir. |
| `AutoInitTracker` sürekli `NEED_DETECT`↔`TRACKING` arasında gidip geliyor (flapping) | `lost_thr=0.8` senin veri dağılımına göre yanlış kalibre | `patience` parametresini artır (ör. 3-5) ve/veya `lost_thr`'ı deneysel olarak ayarla — skorun (`model._last_score`) normal takipte hangi aralıkta gezdiğini birkaç karede loglayıp ona göre karar ver. |
| `EvaluatorUAVtir.run(auto_tracker, ...)` hata veriyor (`AttributeError: 'AutoInitTracker' object has no attribute 'name'`) | `auto_tracking_test_demo.py`'deki `auto_tracker.name = tracker.name` satırı eksik/silinmiş | O satırı geri ekle; evaluator `tracker.name`'i sonuç dosyası adlandırmasında kullanıyor (`libs/data/evaluators/uavtir_eval.py`). |
| CUDA/GPU ile ilgili tip hatası (`bbox` dönüşümünde) | `tracker.update()`'ten dönen kutu GPU tensor'ü olabilir | `trackers/auto_detect.py`'deki `AutoInitTracker._run_loop` zaten bunu `.tolist()` ile ele alıyor — eğer hedef kodda bu satırı elle yeniden yazdıysan, aynı `hasattr(bbox, 'tolist')` kontrolünü koru. |
| `tracking_test_demo.py` (değişmemiş akış) eskisinden farklı sonuç veriyor | Beklenmiyor — regresyon riski sıfır olmalıydı | Bölüm 3.2'deki satırın **sadece** belirtilen yere eklendiğinden, başka hiçbir satırın değişmediğinden emin ol. Farklıysa, `trackers/siamdt_rcnn.py`'deki değişikliği geri al ve hangi satırın kazara değiştiğini `git diff` ile bul. |

---

## 7. Referans Dosyalar (bu repoda)

- Tasarım: `docs/superpowers/specs/2026-08-14-otomatik-tespit-entegrasyonu-design.md`
- Uygulama planı (tam test kodları dahil): `docs/superpowers/plans/2026-08-14-otomatik-tespit-entegrasyonu.md`
- Mimari/kod analizi: `SIAMDT_KOD_ANALIZI.md`
- Arama mekaniği detayları: `SIAMDT_ARAMA_ADIMLARI.md`
- Teknoloji/kavram anlatımı (sıfırdan): `SIAMDT_TEKNOLOJI_RAPORU.md`
- Diğer geliştirme önerileri (bu özellik dışındakiler): `SIAMDT_GELISTIRME_ONERILERI.md`
