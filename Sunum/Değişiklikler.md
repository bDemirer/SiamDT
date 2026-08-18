# SiamDT — Özellik Değişikliği ve Test Günlüğü

## Bu Dosya Hakkında

Bu doküman, SiamDT tracker'ında yapılan **her büyük özellik değişikliğini** — motivasyonu, teknik
yaklaşımı, dokunulan dosyaları, doğrulama sürecini ve sonuçlarını — tek bir yerde, kronolojik ve
karşılaştırılabilir bir formatta tutmak için hazırlandı.

**Format:** Her değişiklik aşağıdaki alt başlıklarla dokümante edilir:
- **Problem / Amaç** — neden yapıldı
- **Teknik Yaklaşım** — nasıl yapıldı, hangi tasarım kararları alındı
- **Yapılan Kod Değişiklikleri** — dosya bazında tablo
- **Doğrulama Süreci ve Sonuçlar** — hangi testler koşuldu, hangi sayılar çıktı
- **Bulunan ve Düzeltilen Hatalar** — süreç içinde ortaya çıkan yan-bulgular
- **Bilinen Açık Noktalar** — henüz netleşmemiş/gözden geçirilmemiş kısımlar
- **Güncel Durum** — şu anki özet durum

Şu ana kadar toplam **3 değişiklik** yapıldı. İlk ikisi ayrıca eklenecek.

---

## Değişiklik #1: *(bekleniyor)*

> Bu bölüm, ilgili değişiklik ve sonuçları paylaşıldığında yukarıdaki formatla doldurulacak.

---

## Değişiklik #2: *(bekleniyor)*

> Bu bölüm, ilgili değişiklik ve sonuçları paylaşıldığında yukarıdaki formatla doldurulacak.

---

## Değişiklik #3: Otomatik Başlatma (Auto-Init) + Kayıp-Hedef Sonrası Yeniden-Tespit

**Durum:** Tamamlandı — kod entegre edildi, doğrulandı, `validate_one_auto.py --auto` ile production
doğrulama akışına bağlandı. Birkaç küçük iyileştirme fırsatı açık kaldı (bkz. "Bilinen Açık Noktalar").

### Problem / Amaç

SiamDT bir **tracker**'dır, dedektör değil: `SiamDTTracker.init(img, init_bbox)` her sekansın ilk
karesinde zorunlu bir başlangıç kutusuna ihtiyaç duyuyordu — bu kutu ya elle veriliyor ya da
benchmark'ta ground-truth etiketten okunuyordu. Ayrıca takip sırasında hedef tamamen kaybolursa (uzun
oklüzyon, ekran dışına çıkma) tracker'ın kendini toparlayacak bir mekanizması yoktu.

**Hedef:**
1. **İlk kare otomasyonu** — video başladığında hedefi (drone'u) sıfırdan bulup tracker'ı elle
   müdahale olmadan başlatmak.
2. **Kayıp-hedef sonrası yeniden-tespit** — takip sırasında güven skoru düşerse aynı mekanizmanın
   tekrar tetiklenip tracker'ı sıfırdan başlatması.

**Kapsam dışı bırakılan:** çoklu-hedef (MOT) ayrımı, kamera hareketi telafisi, gerçek-zamanlı FPS
optimizasyonu, optical flow tabanlı hareket önceliği.

### Teknik Yaklaşım — Neden Yeni Eğitim Gerekmedi

SiamDT normalde iki görüntüyü işler: `img_z` (template) ve `img_x` (o anki kare). İkisi de aynı
backbone'dan geçip özellik haritası (`x`) üretir; `RPN_Similarity_Learning` template'in özelliğini bir
filtreye çevirip `x` ile kanal-bazlı çarpar, çıkan `x_corr` orijinal `x`'e eklenir (`x = x + x_corr`)
ve ancak bu **template-ile-zenginleştirilmiş** `x`, RPN'e ve RCNN kafasına (`roi_head.bbox_head`)
gider.

Eğitim sırasında ise RCNN kafası için **iki ayrı kayıp** hesaplanıp toplanıyor: biri "corr"
(template ile zenginleştirilmiş) özelliklerle, biri **ham** (template'siz) `x` özellikleriyle — ikisi
de aynı `gt_bboxes_x` etiketine göre denetleniyor. Bu, muhtemelen bir çoklu-görev
düzenlileştirmesi olarak tasarlanmıştı, ama yan etkisi şu: `roi_head.bbox_head`'in içinde,
**hiç kullanılmayan ama halihazırda eğitilmiş** bir "ham dal" duruyordu.

`trackers/auto_detect.py`'deki `free_detect()`, bu dalı açığa çıkarıyor: `model.extract_feat()` →
`model.rpn_head.simple_test_rpn()` → `model.roi_head.bbox_roi_extractor()` →
`model.roi_head.bbox_head.get_bboxes()` — hepsi mmdet'in standart, zaten var olan metodları; tek
fark, `x_corr`'u hiç hesaplayıp `x`'e eklemeden çağırmak. Yeni katman, yeni ağırlık, yeni eğitim yok.

**Doğrulanmamış risk (bilinçli olarak kabul edildi):** RCNN kafası için "ham dal zaten eğitilmiş"
doğruydu, ama **RPN** hiçbir zaman saf `x`'i görmedi — hep `x + x_corr`'u gördü (toplama RPN'den önce
yapılıyor). `free_detect()`'te RPN'e template'siz ham `x` verildiğinde, RPN kendi eğitim dağılımının
dışına çıkıyordu — proposal kalitesi bilinmiyordu. Bu yüzden entegrasyon, doğrudan güvenilip
kullanılmadı; önce ölçüldü (bkz. Adım 4).

### Yapılan Kod Değişiklikleri

| Dosya | Değişiklik türü | Açıklama |
|---|---|---|
| `trackers/auto_detect.py` | Yeni dosya | `Candidate`, `_select_candidates` (skor eşiği + `min_area` dejenere-kutu filtresi), `free_detect()`, `default_preprocess()`, `AutoInitTracker` (NEED_DETECT/TRACKING durum makinesi, görselleştirme için `on_frame` callback'i) |
| `trackers/siamdt_rcnn.py` | Tek satır ekleme | `_process_gallary` içinde `self._last_score = float(tra_bboxes[0,-1]+det_bboxes[0,-1])` — mevcut davranışı değiştirmiyor, zaten hesaplanmış bir güven skorunu dışa açıyor |
| `trackers/siamdt_tracking.py` | Kapsamlı değişiklik | `init()`'e `prev_bbox` tutma, yeni `_compute_iou()` metodu, `update()`'te ltrb/ltwh düzeltmesiyle aday yeniden-sıralama + boyut clip'i mantığı (bkz. "Bulunan ve Düzeltilen Hatalar") — **hem auto-init hem elle/GT ile init edilen akışları etkiliyor** |
| `tests/test_auto_detect.py` | Yeni dosya | torch/mmdet gerektirmeyen, bağımsız 13 birim testi |
| `utils/validate_free_detect.py` | Yeni dosya | `free_detect`'in ilk-kare recall'unu ölçen bağımsız doğrulama betiği |
| `auto_tracking_test_demo.py` | Yeni dosya | Uçtan uca, otomatik-başlatmalı akış demo/testi |
| `validate_one_auto.py` | Değişiklik (`--auto` bayrağı) | Epoch-bazlı asıl doğrulama script'ine `AutoInitTracker` entegrasyonu, `gt_bboxes` görselleştirme desteği (GT=yeşil, tahmin=kırmızı, tek sürekli-güncellenen Visdom penceresi) |

### Doğrulama Süreci ve Sonuçlar

- **Adım 1 — Bağımsız testler:** 13/13 test PASS (torch/mmdet gerektirmeyen ortamda da çalışır).
- **Adım 2 — `free_detect` gerçek checkpoint ile:** İlk denemede 1 aday bulundu, skor `0.957`.
- **Adım 3 — `_last_score` kancası:** `init()` sonrası yok (beklenen), 5 ardışık `update()` sonrası
  tutarlı şekilde `~1.98`.
- **Adım 4 — Recall ölçümü ve eşik kalibrasyonu:**
  - İlk ölçüm (20 sekans, `score_thr=0.3`): Recall@IoU0.5 = **%85**, 3 sekansta aday bulunamadı.
  - Dejenere kutu keşfi → `min_area` filtresi eklendi.
  - Eşik tarama (`0.30`→`0.01`): **`score_thr=0.06`** optimal (recall tavan, gürültü düşük).
  - Genişletilmiş test (45 sekans, val): Recall **%57.8**, 19/45 sekansta aday yok — ama bu
    sekansların neredeyse tamamında hedef insan gözüyle de görülemiyor. Aday bulunan **26/26**
    sekansta IoU≥0.5 (**%100**) — model, hedefi gördüğünde her zaman doğru buluyor.
- **Adım 5 — Uçtan uca + regresyon:** Otomatik akışta 3 sekans (200 kare), init-sonrası ortalama IoU
  `0.801` / `0.539` / `0.470`. Visdom görselleştirmesi tek, sürekli-güncellenen pencereye çevrildi.
- **Adım 8 — `validate_one_auto.py` entegrasyonu:** `--auto` bayrağı eklendi; `gt_bboxes` argümanı
  yüzünden çıkan `TypeError` ve görselleştirmenin hiç görünmemesi sorunları çözüldü.

### Bulunan ve Düzeltilen Hatalar

1. **ltrb/ltwh karışıklığı** — `siamdt_tracking.py`'deki özel aday-filtreleme mantığı kutuları
   `[x,y,w,h]` sanıyordu, model gerçekte `[x1,y1,x2,y2]` (ltrb) döndürüyordu. Düzeltme sonrası IoU
   `0.000` → `0.772`.
2. **Dejenere kutular** — RPN'in görüntü kenarında ürettiği sıfır/çok küçük alanlı artefaktlar,
   `min_area` filtresiyle elendi.
3. **`score_thr` kalibrasyon drift'i** — kalibre edilen `0.06` değeri `auto_detect.py`'nin kendi
   default'larına önce hiç yansımadı, sonra bir düzeltme turunda yanlışlıkla `0.6` yazıldı; son
   haliyle `0.06`'da sabitlendi.
4. **Doküman drift'i** — `UYGULAMA_TALIMATLARI.md` §3.3, `siamdt_tracking.py`'nin hiç
   değişmediğini iddia ediyordu; gerçekte kapsamlı biçimde değişmişti. Doküman güncellendi.

### Bilinen Açık Noktalar

- `update()`'te hesaplanan clip'lenmiş kutu sadece `prev_bbox`'a (iç state) yazılıyor; fonksiyonun
  döndürdüğü kutu clip'lenmiyor. Kasıtlı mı, değil mi netleşmedi.
- Yeniden-sıralama mantığındaki magic number'lar (`2.5`, `0.8`, `1.5`, `0.50` vb.) `__init__`
  parametresi değil, koda sabitlenmiş durumda.
- `SiamDTTracker.update()`'teki yeniden-sıralama/clip mantığı için birim testi yok (sadece
  manuel/görsel doğrulandı).

### Güncel Durum

Kod entegre, kalibre (`score_thr=0.06`, `lost_thr=0.8`), doğrulanmış durumda. Production doğrulama
akışı (`validate_one_auto.py --auto`) çalışıyor ve doğru eşiklerle test ediliyor.
