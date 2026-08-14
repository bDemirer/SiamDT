# SiamDT — Otomatik Tespit (Auto-Detect) Entegrasyonu Tasarımı

> Durum: Onaylandı (brainstorming oturumu, 2026-08-14)
> İlgili belgeler: `SIAMDT_KOD_ANALIZI.md`, `SIAMDT_ARAMA_ADIMLARI.md`, `SIAMDT_GELISTIRME_ONERILERI.md` (madde C1 ile örtüşür)

---

## 1. Problem ve Kapsam

SiamDT bir **tracker**'dır, dedektör değil: `SiamDTTracker.init(img, init_bbox)` her sekansın ilk
karesinde **zorunlu bir başlangıç kutusuna** ihtiyaç duyar (bkz. `SIAMDT_KOD_ANALIZI.md` §2.2,
`libs/tracker.py:37-56`, `libs/data/evaluators/uavtir_eval.py:151`). Bu kutu şu an ya elle veriliyor
ya da benchmark'ta ground-truth etiketten okunuyor. Ayrıca takip sırasında hedef tamamen kaybolursa
(uzun süreli oklüzyon, ekran dışına çıkma) tracker'ın kendini toparlayacak bir mekanizması yok.

**Bu tasarımın kapsamı:**
1. **İlk kare otomasyonu** — video başladığında hedefi sıfırdan bulup `init()`'i tetiklemek.
2. **Kayıp-hedef sonrası yeniden-tespit** — takip sırasında güven düşerse aynı mekanizmayı tekrar
   tetikleyip tracker'ı sıfırdan başlatmak.

**Kapsam dışı (bilinçli olarak ele alınmayan):** çoklu-hedef (MOT) ayrımı, kamera hareketi telafisi,
gerçek-zamanlı FPS optimizasyonu, optical flow tabanlı hareket önceliği (ayrı bir konuşma konusu,
bu tasarımın parçası değil).

## 2. Kısıt: Taşınabilirlik

Bu tasarımın uygulanacağı asıl kod tabanı, burada incelenen SiamDT kopyasından **farklı/değişikliğe
uğramış bir kopya** üzerinde çalışıyor — değişiklikler oradan buraya değil, **buradan oraya**
kopyala-yapıştır ile taşınacak. Bu yüzden tasarım şu ilkeye göre kurgulanmıştır:

- Yeni mantığın **tamamı tek bir yeni dosyada** toplanır.
- Mevcut dosyalara yapılan değişiklik **tek satırlık, kolay bulunabilir bir kanca (hook)** ile
  sınırlıdır.
- Yeni dosya, SiamDT'ye özgü iç detaylara değil, mmdet'in `TwoStageDetector` temelinden gelen
  **stabil, standart API'ye** (`extract_feat`, `rpn_head.simple_test_rpn`, `roi_head.bbox_roi_extractor`,
  `roi_head.bbox_head`) dayanır — hedef kod SiamDT'yi ne kadar değiştirmiş olursa olsun bu isimler
  mmdet uyumluluğu bozulmadıkça sabit kalır.

## 3. Mimari

### 3.1 Yeni dosya: `trackers/auto_detect.py`

**a) `free_detect(model, img_tensor, img_meta, score_thr=0.3, max_candidates=5)`**

Template gerektirmeyen, bağımsız bir tespit fonksiyonu:

```python
x = model.extract_feat(img_tensor)                              # x_corr YOK (Yaklaşım A)
proposal_list = model.rpn_head.simple_test_rpn(x, img_meta)     # ham proposal, tüm görüntü
rois = bbox2roi(proposal_list)
bbox_feats = model.roi_head.bbox_roi_extractor(
    x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
cls_score, bbox_pred = model.roi_head.bbox_head(bbox_feats)     # roi_head.bbox_head'in "ham dalı"
det_bboxes, det_labels = model.roi_head.bbox_head.get_bboxes(
    rois, cls_score, bbox_pred, img_meta['img_shape'], img_meta['scale_factor'],
    rescale=True, cfg=model.test_cfg.rcnn)
# score_thr üstü kalanları skora göre sırala, ilk max_candidates'i döndür
```

`RPN_Similarity_Learning`/`RCNN_Similarity_Learning` hiç çağrılmaz — template olmadan çalışır, hem
ilk karede hem yeniden-tespitte kullanılabilir.

**b) `AutoInitTracker`** — orkestrasyon sınıfı

Var olan tracker'ın **public API'sini** (`init(img, bbox)`, `update(img)`) kullanır, iç sınıflarına
dokunmaz. Kurucuda `model` ve `transforms` nesnelerini (çağıran kodun zaten elinde tuttuğu, ör.
`tracking_test_demo.py`'deki gibi) doğrudan parametre olarak alır — tracker nesnesinin içinden
introspect etmeye çalışmaz, çünkü attribute isimleri hedef kodda farklı olabilir.

```python
class AutoInitTracker:
    def __init__(self, tracker, model, transforms, lost_thr=0.8, patience=1):
        ...

    def forward_test(self, img_files):
        state = "NEED_DETECT"
        low_score_streak = 0
        for f, img_file in enumerate(img_files):
            img = read_image(img_file)
            if state == "NEED_DETECT":
                candidates = free_detect(self.model, *self._preprocess(img))
                if candidates:
                    self.tracker.init(img, candidates[0].bbox)
                    state = "TRACKING"
                    low_score_streak = 0
                    bboxes[f] = candidates[0].bbox
                else:
                    bboxes[f] = bboxes[f - 1] if f > 0 else np.zeros(4)
            else:
                bbox, up_flag, score = self.tracker.update(img)
                bboxes[f] = bbox
                if score < self.lost_thr:
                    low_score_streak += 1
                    if low_score_streak >= self.patience:
                        state = "NEED_DETECT"
                else:
                    low_score_streak = 0
        return bboxes
```

`patience` (varsayılan 1) ve `lost_thr` (varsayılan 0.8) kurucu parametresi olarak dışa açık —
karşı ortamda ayar gerekirse kod değişikliği gerekmez.

### 3.2 Mevcut koda tek satırlık kanca

`_process_gallary` içinde (template güncelleme kararının hesaplandığı satır,
`tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9` koşulunun yanına) şu satır eklenir:

```python
self._last_score = float(tra_bboxes[0, -1] + det_bboxes[0, -1])
```

`SiamDTTracker.update()`'in dönüş değerine bu skor eklenir (`bbox, up_flag, score`). Bu, mevcut
davranışı **değiştirmez** — sadece zaten hesaplanmış bir değeri dışarı açar.

**Not — iki farklı eşik, tek skor kaynağı:** `self._last_score` (`tra_bboxes[0,-1]+det_bboxes[0,-1]`)
**0-2 aralığında** bir değerdir (iki ayrı 0-1 skorun toplamı) — mevcut template güncelleme eşiği
zaten bu ölçekte `>1.9` (neredeyse maksimum, bilinçli olarak sıkı). Bu eşiği doğrudan tersine çevirip
"kayıp" saymak, normal takipte bile sık sık yanlış tetiklenir. Bunun yerine aynı skor, **ayrı ve
düşük** bir eşikle (`lost_thr=0.8`, yani aynı 0-2 ölçeğinde "orta-altı güven") yorumlanır — iki
farklı karar (template güncelle / hedef kayıp say) için iki farklı sabit, ama tek bir hesaplama.
`lost_thr`'ın kesin değeri veri setine bağlıdır; Bölüm 5'teki doğrulama betiğiyle deneysel olarak
ayarlanmalıdır — buradaki 0.8 sadece makul bir başlangıç noktasıdır.

## 4. Algoritma Seçimi ve Kademeli Yol Haritası

Üç yaklaşım değerlendirildi:

- **Yaklaşım A (seçildi, ilk adım):** `rpn_head`'e ham `x` verilir (x_corr eklenmez),
  `roi_head.bbox_head`'in zaten var olan "ham dalı" (`cls_score_x`/`bbox_pred_x`) kullanılır. **Sıfır
  yeni ağırlık, sıfır yeni eğitim.** Kullanıcının kendi verisiyle daha önce eğittiği checkpoint'te bu
  ham dal zaten `gt_bboxes_x`'e göre denetimli (supervised) eğitilmiş durumda — bu yüzden RCNN
  tarafının ek fine-tune gerektirmesi beklenmiyor. **Doğrulanmamış risk:** `rpn_head`, eğitim
  sırasında hiçbir zaman ham `x`'i (x_corr'suz) görmedi — proposal kalitesi bilinmiyor, Bölüm 5'teki
  doğrulama adımıyla ölçülecek.
- **Yaklaşım B (reddedildi):** Aynı paylaşılan `rpn_head`'i ham `x` üzerinde ince ayar yapmak. Mevcut
  Siamese takip akışında da kullanılan ağırlığı riske attığı için reddedildi.
- **Yaklaşım C (koşullu sonraki adım):** Yaklaşım A'nın doğrulaması yetersiz çıkarsa, `rpn_head`'i
  klonlayan ama **tamamen ayrı ağırlıklı** bir `rpn_head_plain` eklenip sadece bu, Anti-UAV410
  üzerinde eğitilecek. Orijinal `rpn_head` hiç değişmediği için mevcut tracking performansına sıfır
  regresyon riski taşır. Bu adım gerçekleşirse ek bir eğitim çalışması ve yeni bir checkpoint
  dosyasının (kod değil, ikili artefakt olarak) diğer bilgisayara taşınması gerekir.

## 5. Doğrulama / Test Stratejisi

1. **Önce ölç, sonra entegre et:** `free_detect`'i, mevcut `UAVtir` dataset sınıfı ve kullanıcının
   kendi eğittiği checkpoint ile birkaç test sekansının karelerinde çalıştırıp `gt_rect` ile
   IoU>0.5 karşılaştırması yapan küçük, bağımsız bir betik (`utils/validate_free_detect.py`)
   yazılacak. Basit bir recall/precision sayısı üretir. Bu adım, Yaklaşım C'ye geçilip
   geçilmeyeceğine karar verdirecek.
2. **Regresyon kontrolü:** `tracking_test_demo.py` (mevcut, dokunulmamış tracker akışı) değişiklik
   öncesi/sonrası aynı sonucu verdiği teyit edilir. Yeni dosya import edilip kullanılmadığı sürece
   davranış zaten aynı kalır (Bölüm 3.2'deki tek satır hariç, o da sadece bir attribute set ediyor).

## 6. Kenar Durumları

| Durum | Davranış |
|---|---|
| `free_detect` hiç aday bulamıyor | `NEED_DETECT` durumunda kalınır, bir sonraki karede tekrar denenir; o kare için bir önceki bilinen kutu (veya sekans başıysa sıfır kutu) yazılır. |
| Birden fazla makul aday (belirsiz sahne) | En yüksek skorlu aday (top-1) seçilir. Template olmadığı için mevcut arka-plan bastırma/IoU-konsensüs mantığı (bkz. `SIAMDT_ARAMA_ADIMLARI.md` §3.5-3.6) burada uygulanamaz — **bilinen bir sınır**, bu tasarımın kapsamı dışında bırakıldı. |
| Durum salınımı (skor eşik civarında gidip geliyor) | `patience` parametresiyle (varsayılan 1, artırılabilir) N kare üst üste düşük skor şartı eklenebilir. |
| Yeniden-tespit sonrası template güncelleme | **Tam sıfırlama** — `tracker.init()` yeni bulunan kutuyla sıfırdan çağrılır, eski template/arka-plan havuzu tamamen atılır (EMA harmanlama yok). |

## 7. Açık Riskler / Bilinmeyenler

- `rpn_head`'in ham `x` girdisindeki gerçek proposal kalitesi **doğrulanmadı** — Bölüm 5, Adım 1
  sonucuna göre Yaklaşım A yeterli mi yoksa Yaklaşım C mi gerekli, netleşecek.
- Çoklu-drone / karmaşık sahne ayrımı bu tasarımla çözülmüyor (kapsam dışı, Bölüm 6).
- `free_detect`'in ön-işleme (preprocessing) fonksiyonuna bağımlılığı — hedef koddaki `transforms`
  nesnesinin `_process_gallary`'ye benzer bir arayüz sunduğu varsayılıyor; farklıysa `_preprocess`
  yardımcı fonksiyonu hedef kodda küçük bir uyarlama gerektirebilir.

---

*Bu belge, kullanıcıyla yapılan brainstorming oturumunun onaylanmış çıktısıdır. Sıradaki adım:
`writing-plans` becerisiyle uygulama planı çıkarmak.*
