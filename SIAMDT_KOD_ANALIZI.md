# SiamDT Kod Tabanı — Derinlemesine Analiz

> Bu doküman `C:\Users\burak\Desktop\SiamDT` reposunun kod incelemesinden çıkarılmıştır.
> Tüm dosya/satır referansları repo köküne göredir.

---

## 1. Genel Bakış

**Problem:** Anti-UAV (İHA/drone) tracking — genelde termal (IR) video akışında, ilk karede verilen
hedef kutusundan (bbox) yola çıkarak dronu takip etmek. Klasik "komşulukta ara" (local search)
mantığı yerine, SiamDT **her karede tüm görüntüde hedefi yeniden bulan bir dedektör** gibi çalışır
("tracking-by-detection", Siamese-tarzı global arama).

**Mimari köken:** mmdetection tabanlı iki aşamalı bir dedektör (Faster/Cascade R-CNN), **Siamese**
hale getirilmiş:

- **Backbone:** Swin Transformer (`configs/siamdt_swin_tiny_adamw.py:9-24`), COCO'da önceden
  eğitilmiş Cascade Mask R-CNN ağırlıklarından (`pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar`)
  başlatılıyor.
- **Neck:** FPN (`configs/siamdt_swin_tiny_adamw.py:25-29`).
- **RPN + RoI Head:** standart mmdet `RPNHead` / `StandardRoIHead` + `Shared2FCBBoxHead`, ama ikisi
  de **template (z) özelliğiyle modüle edilerek** çalışıyor (bkz. Bölüm 3).

Genel iskelet (`libs/model.py`'daki soyut `Model`, `libs/tracker.py`'daki `Tracker`/`OxUvA_Tracker`,
`libs/data` altındaki GOT-10k/LaSOT/OTB/VOT/UAV123 dataset sınıfları, `Pair`-tabanlı eğitim akışı,
evaluator yapısı) Lianghua Huang'ın **GlobalTrack** (AAAI 2020) tarzı kod tabanının izlerini taşıyor
— sınıf isimlendirmesi, `init/update` API'si, cache/registry mekanizması neredeyse birebir aynı
desenler. SiamDT bunun üzerine iki özgün korelasyon modülü ekliyor:

- `RPN_Similarity_Learning` (tanımı `trackers/similarity_encoders.py:8-39`) → çağrıldığı yerde,
  `trackers/siamdt_rcnn.py:98`'deki kod içi yorumda **"Dual-Semantic Learning"** olarak anılıyor
- `RCNN_Similarity_Learning` (tanımı `trackers/similarity_encoders.py:42-58`) → çağrıldığı yerde,
  `trackers/siamdt_rcnn.py:148`'deki kod içi yorumda **"Versatile learning"** olarak anılıyor

Repo içinde makaleye doğrudan bir referans/link yok (README sadece "official implementation of
SiamDT" diyor); bu nedenle tam makale başlığını/venue'sunu **kesin olarak doğrulayamıyorum** —
sadece kod yapısından çıkarım yapıyorum.

**Veri seti odağı:** `datasets/wrappers.py:23-26` ve `tracking_test_demo.py:22-23`'te açıkça görülüyor
— asıl hedef **Anti-UAV410** (kod içinde kısaca `uavtir`) termal drone veri seti. COCO/GOT-10k/LaSOT/
ImageNet-VID/VisDrone gibi genel amaçlı setler ise ek/yardımcı eğitim verisi olarak karışıma
katılıyor (çoklu veri seti örnekleme, `sampling_prob`).

### Klasör yapısı

```
SiamDT/
├── configs/                          mmdet-tarzı model+eğitim config'leri
│   ├── siamdt_swin_tiny_adamw.py     AdamW optimizer varyantı
│   └── siamdt_swin_tiny_sgd.py       SGD optimizer varyantı (reg_class_agnostic=True, SmoothL1Loss)
├── trackers/                         SiamDT'ye özgü ASIL mantık
│   ├── siamdt_rcnn.py                 SiamDTRCNN: Siamese Faster/Cascade-RCNN modeli
│   ├── siamdt_tracking.py             SiamDTTracker: benchmark-uyumlu init()/update() sarmalayıcı
│   └── similarity_encoders.py         RPN_Similarity_Learning, RCNN_Similarity_Learning
├── libs/
│   ├── model.py                      soyut Model taban sınıfı (nn.Module)
│   ├── tracker.py                    soyut Tracker / OxUvA_Tracker taban sınıfı, frame-loop
│   ├── config/                       kendi mini Registry + Config sarmalayıcısı (mmcv'den bağımsız,
│   │                                  datasets/transforms/evaluators için)
│   ├── data/
│   │   ├── datasets/                 GOT10k, LaSOT, OTB, VOT, UAV123, DTB70, TrackingNet, NfS,
│   │   │                              TColor128, TLP, POT, OxUvA, MOT, ImageNet-VID, VisDrone, COCO,
│   │   │                              UAVtir (Anti-UAV410) — hepsi SeqDataset/ImageDataset türevi
│   │   ├── evaluators/                OTB/GOT10k/VOT/OxUvA/UAVtir tarzı benchmark koşucuları
│   │   ├── samplers/                  RandomIdentitySampler (ReID-tarzı, SiamDT eğitiminde kullanılmıyor)
│   │   └── transforms/                pair_transforms (mmdet_transforms.py, siamfc_transforms.py),
│   │                                   img_transforms (reid_transforms.py)
│   ├── ops/                          image/io/loss/metric/transform yardımcı fonksiyonları
│   └── swintransformer/              **vendored**: tam bir Swin-Transformer-Object-Detection /
│                                       mmdetection kopyası — backbone, mmdet çekirdeği (registry,
│                                       apis, core, models, datasets), tüm config zoo'su buradan geliyor
├── datasets/
│   └── wrappers.py                   PairWrapper: farklı dataset'lerden (z,x) template/search
│                                       çiftleri üreten mmdet-uyumlu eğitim sarmalayıcısı
├── utils/                            yardımcı scriptler
│   ├── obtain_pretrained_weights.py   Swin/Cascade checkpoint'inden sadece backbone+neck çıkarma
│   ├── gen_json_video.py / _2.py      video/etiket json üretimi
│   └── run_antiuav.py                 eski/alternatif bir benchmark koşucusu (bkz. Bölüm 5 — kırılgan)
├── tracking_train_demo.py            EĞİTİM entry point
├── tracking_test_demo.py             TAKİP (tracking) test/eval entry point
├── detection_test_demo.py            saf mmdet dedektör demo'su (SiamDT'siz, sadece Swin backbone testi)
└── init_paths.py                     sys.path'e libs/swintransformer ekleyen + modülleri register eden bootstrap
```

`libs/swintransformer` neredeyse tamamen üçüncü parti kod (mmdetection + Swin-T repo kopyası) —
buraya dokunman gerekmez, sadece backbone/mmdet altyapısını sağlar. Asıl "SiamDT" değeri
**`trackers/`** klasöründe ve onu besleyen **`configs/`, `libs/data`, `datasets/wrappers.py`**
içinde toplanmış.

---

## 2. Giriş Noktası ve Akış

İki ayrı ana akış var: **eğitim** ve **takip testi**. Ayrıca `detection_test_demo.py` SiamDT'yi
bypass eden saf bir mmdet demo'sudur (backbone/mmdet kurulumunu doğrulamak için).

### 2.1 Eğitim akışı — `tracking_train_demo.py`

```
python tracking_train_demo.py
  └─ init_paths (sys.path'e libs/swintransformer eklenir, trackers/* ve datasets/* register edilir)
  └─ main()                                            tracking_train_demo.py:118
       ├─ Config.fromfile('configs/siamdt_swin_tiny_sgd.py')
       ├─ build_detector(cfg.model, ...)                → SiamDTRCNN(...)   [mmdet registry üzerinden]
       │      SiamDTRCNN.__init__                        trackers/siamdt_rcnn.py:23-44
       │        ├─ TwoStageDetector.__init__ (backbone=Swin-T, neck=FPN, rpn_head, roi_head kurulur)
       │        └─ RPN_Similarity_Learning / RCNN_Similarity_Learning örneklenir + init_weights()
       ├─ build_dataset(cfg.data.train)                 → PairWrapper(...)   datasets/wrappers.py:53
       │        └─ _setup_base_dataset → Seq2Pair(UAVtir/GOT10k/LaSOT) veya Image2Pair(COCO)
       │           + RandomConcat (çoklu dataset karışımı, sampling_prob ile)
       └─ train_detector(model, datasets, cfg, ...)      [mmdet.apis, standart mmdet eğitim döngüsü]
            için her batch'te:
              PairWrapper.__getitem__                     datasets/wrappers.py:74-119
                └─ Seq2Pair.__getitem__ (rastgele bir z/x çift kare seçer, anno filtrelenir)
                     trackers/similarity_encoders.py ile ilgisi yok — bu saf veri hazırlama
                └─ ExtraPairTransforms/BasicPairTransforms uygulanır
                     libs/data/transforms/pair_transforms/mmdet_transforms.py:271-349
              → DataContainer'lara sarılmış {img_z, img_x, gt_bboxes_z, gt_bboxes_x, gt_labels, img_meta_*}
              → SiamDTRCNN.forward(img_z, img_x, img_meta_z, img_meta_x, return_loss=True, ...)
                     trackers/siamdt_rcnn.py:47-59
                   └─ forward_train(...)                  trackers/siamdt_rcnn.py:65-185  (bkz. Bölüm 3)
```

### 2.2 Takip testi akışı — `tracking_test_demo.py`

```
python tracking_test_demo.py                            tracking_test_demo.py:9-26
  ├─ transforms = data.BasicPairTransforms(train=False)
  ├─ tracker = SiamDTTracker(cfg_file, ckp_file, transforms, ...)
  │      SiamDTTracker.__init__                          trackers/siamdt_tracking.py:18-47
  │        ├─ Config.fromfile(cfg_file) → build_detector → SiamDTRCNN
  │        └─ load_checkpoint(model, ckp_file)            # eğitilmiş .pth yüklenir
  └─ evaluator = data.EvaluatorUAVtir(root_dir=..., subset='test')
       EvaluatorUAVtir → UAVtir_Eval.__init__              libs/data/evaluators/uavtir_eval.py:478-489
       evaluator.run(tracker, selected_seq='ALL')          libs/data/evaluators/uavtir_eval.py:120-190
         for her sequence:
           bboxes, times = tracker.forward_test(img_files, init_bbox, visualize=...)
               Tracker.forward_test                        libs/tracker.py:37-67
                 ├─ f==0 ise:  self.init(img, init_bbox)
                 │     SiamDTTracker.init                   trackers/siamdt_tracking.py:49-63
                 │       ├─ transforms._process_query(img, meta, bbox)   # resize/normalize/pad/ToTensor
                 │       └─ model._process_query(img, [bboxes], [meta])
                 │             SiamDTRCNN._process_query     trackers/siamdt_rcnn.py:351-391
                 │               ├─ backbone+FPN'den z özellik haritası çıkarılır (extract_feat)
                 │               ├─ RoIAlign ile template ROI feature'ı (_template) elde edilir
                 │               └─ RPN ile arka plan (background) proposal'ları çıkarılıp
                 │                  _bbox_feats_bg saklanır (arka plan bastırma için)
                 └─ f>0 ise:   current_box, up_flag = self.update(img)
                       SiamDTTracker.update                 trackers/siamdt_tracking.py:65-86
                         ├─ transforms._process_gallary(img, meta, None)
                         └─ model._process_gallary(img, [meta], rescale=True, ...)
                               SiamDTRCNN._process_gallary   trackers/siamdt_rcnn.py:428-518 (bkz. Bölüm 3)
                                 → en yüksek skorlu bbox + up_flag (template güncellendi mi) döner
       evaluator kendi eval()/report() metotlarıyla IoU tabanlı "Mixed Measure" skorunu hesaplar ve
       results/ + reports/ klasörlerine yazar.
```

### 2.3 `detection_test_demo.py`

SiamDT'nin Siamese mantığını hiç kullanmıyor; doğrudan `mmdet.apis.init_detector` /
`inference_detector` ile Swin tabanlı bir Cascade Mask R-CNN'i COCO checkpoint'iyle çalıştırıyor
(`detection_test_demo.py:6-19`). Kurulumu (backbone derlemesi, mmcv-full uyumluluğu vb.) doğrulamak
için bir sağlık kontrolü niteliğinde.

---

## 3. Mimari ve Ana Bileşenler

### 3.1 `SiamDTRCNN` (`trackers/siamdt_rcnn.py:20-44`)

mmdet'in `TwoStageDetector`'ından türetilmiş. Standart bileşenlere ek olarak iki korelasyon modülü
taşır:

| Bileşen | Kaynak | Girdi | Çıktı |
|---|---|---|---|
| `backbone` (Swin-T) | mmdet registry, `configs/*.py:9-24` | görüntü tensörü | çok ölçekli özellik haritaları |
| `neck` (FPN) | mmdet registry | backbone çıktıları | 5 seviyeli piramit (P2-P6) |
| `rpn_head` | mmdet `RPNHead` | modüle edilmiş x özellikleri | objectness skorları + kutu teklifleri (proposal) |
| `roi_head.bbox_roi_extractor` | mmdet `SingleRoIExtractor` (RoIAlign) | özellik haritası + RoI kutuları | 7×7 RoI özellikleri |
| `roi_head.bbox_head` | mmdet `Shared2FCBBoxHead` | RoI özellikleri | sınıf skoru + kutu regresyonu (delta) |
| `rpn_similarity_learning` | **SiamDT'ye özgü** | template + x özellik haritaları | x'e eklenecek "korelasyon" haritası |
| `rcnn_similarity_learning` | **SiamDT'ye özgü** | template RoI feat + x RoI feat | eşleşme (matching) RoI feature'ı |

### 3.2 Dual-Semantic RPN Modülasyonu — `RPN_Similarity_Learning` (`trackers/similarity_encoders.py:8-39`)

```python
class RPN_Similarity_Learning(nn.Module):
    def __init__(self, roi_size=7, channels=256, featmap_num=5):
        self.proj_query = nn.ModuleList([nn.Conv2d(channels, channels, roi_size, padding=0) for _ in range(featmap_num)])
        self.proj_out   = nn.ModuleList([nn.Conv2d(channels, channels, 1, padding=0) for _ in range(featmap_num)])

    def forward(self, template, feats_x):
        ...
        out_ij = [self.proj_query[k](query) * gallary[k] for k in range(len(gallary))]
        out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
        yield out_ij, i, j
```

- `template`: `bbox_feats_z` üzerinden RoIAlign ile çıkarılmış **7×7×256** boyutunda hedef özelliği.
- `proj_query`: 7×7 kernel'li konvolüsyon → template'i **1×1** boyutuna "sıkıştırıp" bir filtre
  haline getiriyor (klasik SiamFC/SiamRPN'deki depthwise cross-correlation kernel'i gibi).
- Bu 1×1 filtre, x'in her FPN seviyesindeki özellik haritasıyla **kanal-bazlı çarpılıyor**
  (`proj_query[k](query) * gallary[k]`) — yani konvolüsyon değil, kanal-bazlı element-wise çarpım.
- `proj_out`: 1×1 konvolüsyon ile sonucu tekrar projekte ediyor.
- Sonuç (`x_corr`), `siamdt_rcnn.py:101-104`'te orijinal x özellikleriyle **toplanıyor**
  (`x[i] + x_corr[i]`) ve bu "template-farkında" x, RPN'e giriyor. Böylece RPN artık sadece
  "nesne var mı" değil, "**bu template'e benzeyen** nesne var mı" sorusuna cevap arıyor.

### 3.3 Versatile RCNN Modülasyonu — `RCNN_Similarity_Learning` (`trackers/similarity_encoders.py:42-58`)

```python
class RCNN_Similarity_Learning(nn.Module):
    def __init__(self, channels=256):
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):
        return self.proj_out(self.proj_x(x) * self.proj_z(z))
```

RPN aşamasından gelen teklifler (proposals) üzerinde tekrar RoIAlign yapılıp elde edilen `bbox_feats_x`
ile template `bbox_feats_z` burada da kanal-bazlı çarpımla harmanlanıyor
(`siamdt_rcnn.py:149-150`, `203-204`, `276-277`). Çıkan `bbox_feats_corr`, standart
`roi_head.bbox_head`'e (Shared2FCBBoxHead) verilerek sınıf/kutu tahmini üretiliyor.

Önemli tasarım detayı: **eğitimde** hem "corr" (template ile modüle edilmiş) hem de "ham" x
özellikleriyle iki ayrı kayıp hesaplanıp toplanıyor (`siamdt_rcnn.py:158-183`) — modelin hem saf
nesne tespiti hem de template-eşleştirmesi öğrenmesini sağlamak için bir tür çoklu-görev (multi-task)
düzenlileştirme.

### 3.4 Loss fonksiyonları

Gerçek kayıplar **mmdet'in registry sistemi** üzerinden config dosyasından geliyor
(`configs/siamdt_swin_tiny_adamw.py:43-45,64-66`):

- RPN: `CrossEntropyLoss(use_sigmoid=True)` + `L1Loss` (SGD varyantında `SmoothL1Loss`)
- RCNN: `CrossEntropyLoss(use_sigmoid=False)` + `L1Loss` (SGD varyantında `SmoothL1Loss`)

Bunlar `roi_head.bbox_head.loss(...)` (`siamdt_rcnn.py:160-161,169-170`) ve `rpn_head.loss(...)`
(`siamdt_rcnn.py:113`) içinde mmdet'in standart mekanizmasıyla hesaplanıyor.

⚠️ **Dikkat:** `libs/ops/losses.py` içinde `balanced_bce_loss`, `focal_loss`, `ghmc_loss`,
`ohem_bce_loss`, `iou_loss`, `ghmr_loss`, `label_smooth_loss` gibi bir sürü özel kayıp fonksiyonu
var, ama grep ile doğruladığım kadarıyla **hiçbiri SiamDT tarafından import edilip kullanılmıyor**
— muhtemelen bu koleksiyonun türediği daha genel/eski repodan (GlobalTrack ekosistemi, örn. SiamFC/
TTFNet varyantları için) kalma, SiamDT'de ölü kod. Yeni bir loss eklemek istersen mmdet'in kendi
loss registry'sini (`libs/swintransformer/mmdet/models/losses/`) kullanmak, config'te
`loss_cls`/`loss_bbox` dict'ini değiştirmek daha doğru yol.

### 3.5 Bileşenler arası bağlantı şeması

```
             ┌────────────┐        ┌────────────┐
   img_z --> │  backbone  │  z --> │RoIAlign(gt_z)│ --> bbox_feats_z (template, 7x7x256)
             │  (Swin-T)  │        └────────────┘            │
             │   + FPN    │                                   │  (eğitimde: her instance)
             └────────────┘                                   │
                                                                ▼
             ┌────────────┐        ┌───────────────────────┐   ┌──────────────────────┐
   img_x --> │  backbone  │  x --> │RPN_Similarity_Learning │-->│  x_corr (5 FPN lvl)  │
             │  (Swin-T)  │        │  (proj_query*proj_out) │   └──────────────────────┘
             │   + FPN    │        └───────────────────────┘             │
             └────────────┘                     │                        │ x = x + x_corr
                                                  └────────────────────────┘
                                                                │
                                                                ▼
                                                         ┌─────────────┐
                                                         │  rpn_head   │ --> proposals
                                                         └─────────────┘
                                                                │
                                                                ▼
                                          ┌───────────────────────────────┐
                                          │ RoIAlign(proposals) --> bbox_feats_x │
                                          └───────────────────────────────┘
                                                                │
                          bbox_feats_z ───────────────────────►│
                                                                ▼
                                                ┌────────────────────────┐
                                                │RCNN_Similarity_Learning │ --> bbox_feats_corr
                                                │  (proj_z*proj_x*proj_out)│
                                                └────────────────────────┘
                                                                │
                                                                ▼
                                                    roi_head.bbox_head
                                                                │
                                                                ▼
                                                cls_score, bbox_pred --> final bbox
```

Test aşamasında ayrıca `_process_gallary` (`siamdt_rcnn.py:428-518`) içinde bir
**arka plan bastırma (background suppression)** ve **IoU tabanlı skor güçlendirme** mantığı var:
en iyi 5 aday, template'in arka plan proposal'larıyla (`_bbox_feats_bg`) tekrar eşleştirilip skorları
düşürülüyor; birbirine çok yakın (IoU>0.8) adaylar birbirinin skorunu artırıyor. Ayrıca yüksek
güvenli tespitlerde `_update_query` (`siamdt_rcnn.py:393-426`) ile **template online olarak
güncelleniyor** (exponential moving average; `self._learning_rate = 0.01` sabiti `_process_query`
içinde `siamdt_rcnn.py:353`'te tanımlanıyor, `_update_query` içindeki `siamdt_rcnn.py:402` ise bu
sabiti EMA formülünde sadece kullanıyor).

---

## 4. Veri Akışı

### 4.1 Eğitim verisi — dataset formatı ve dataloader

1. **Ham dataset sınıfları** (`libs/data/datasets/*.py`) — her biri `SeqDataset`'ten türer
   (`libs/data/datasets/dataset.py:15-65`). `__getitem__` bir sekans için `(img_files, target)`
   döndürür; `target['anno']` kutuları, `target['meta']` genişlik/yükseklik/frame sayısı içerir.
   Sonuçlar `cache/<dataset_name>.pkl` altında **cache'lenir** (`dataset.py:24-32`) — dataset
   yolu değiştiğinde cache silinmezse eski indeks kalabilir (README'deki "Issue 1" tam da bunu
   işaret ediyor: `ValueError: need at least one array to concatenate` → `cache/` klasörünü sil).
   - `UAVtir` (`libs/data/datasets/uavtir.py:18-96`): `root_dir/<subset>/<seq>/IR_label.json`
     + `*.jpg` dosyalarını okur; `gt_rect` alanı `[x,y,w,h]` formatında, `[x1,y1,x2,y2]`'ye
     çevriliyor (`uavtir.py:70-75`).

2. **Pair oluşturma** — `Seq2Pair` (`libs/data/datasets/samplers... ` hayır, doğrusu:
   `libs/data/datasets/structure.py:15-177`, ama modül aslında `libs/data/datasets/structure.py`
   içinde `Seq2Pair`/`Image2Pair`/`RandomConcat` tanımlı):
   - Bir sekanstan rastgele iki kare (`rand_z`, `rand_x`, `max_distance=300` kare içinde) seçilir.
   - `_filter` (`structure.py:129-163`) gürültülü/aşırı küçük-büyük/aşırı ince kutuları eler
     (alan>20px, kenar 10-960px arası, görüntüye oranı %1-%75 arası, en-boy oranı 0.2-5 arası).
   - COCO gibi görüntü-tabanlı (video olmayan) setler için `Image2Pair` kullanılır — z ve x aynı
     görüntünün iki kopyasıdır (`structure.py:191-198`).
   - Birden fazla dataset varsa `RandomConcat` (`structure.py:366-421`) `sampling_prob`'a göre
     karışık örnekleme yapar (en-boy oranına göre gruplanmış "group_flags" ile — mmdet'in
     "aspect ratio grouping" mantığına uyum için).

3. **`PairWrapper`** (`datasets/wrappers.py:52-158`) — mmdet'in `DATASETS` registry'sine
   kaydedilmiş asıl dataset sınıfı, `cfg.data.train` içinde `type='PairWrapper'` olarak
   kullanılıyor. Görevi:
   - `base_dataset` string'ini (`'uavtir_train'`, `'got10k_train,lasot_train'` gibi virgülle
     ayrılmış) `_datasets()` (`wrappers.py:11-32`) ile gerçek dataset nesnelerine çeviriyor —
     **veri yolları burada hardcoded** (`/media/data2/TrackingDatasets/...`, bkz. Bölüm 5).
   - `base_transforms` ile hangi augmentasyon setinin (`basic_train`/`extra_partial`/`extra_full`)
     uygulanacağını seçiyor (`wrappers.py:35-49`).
   - `__getitem__` (`wrappers.py:74-119`): en fazla `max_instances=8` hedef örnekleyip, sonucu
     mmcv'nin `DataContainer` (`DC`) yapılarına sarıyor — bu, mmdet'in `collate_fn`'inin farklı
     boyutlardaki tensörleri (görüntü boyutu, kutu sayısı) tek batch'te toplayabilmesi için gerekli.

4. **Transform zinciri** (`libs/data/transforms/pair_transforms/mmdet_transforms.py`):
   `ExtraPairTransforms` (train, `wrappers.py:41-45`'te config: `with_photometric=True,
   with_expand=False, with_crop=False`) → `PhotometricDistort` → `BasicPairTransforms`
   (`Rescale(1333,800)` → `Normalize` → `RandomFlip` → `PadToDivisor(32)` → `BoundBoxes` →
   `ToTensor`). Template (z) ve arama görüntüsü (x) **ayrı ayrı** işleniyor ama aynı transform
   zincirinden geçiyor (`_process_query` / `_process_gallary`, `mmdet_transforms.py:13-36`).

5. **Model'e giriş:** `img_z`, `img_x`, `img_meta_z`, `img_meta_x`, `gt_bboxes_z`, `gt_bboxes_x`,
   `gt_labels` — hepsi `SiamDTRCNN.forward_train`'e (`siamdt_rcnn.py:65-75`) argüman olarak gidiyor.

### 4.2 Test/çıkarım verisi ve post-processing

- Tek bir görüntü + init bbox → `SiamDTTracker.init` (`siamdt_tracking.py:49-63`): aynı
  `BasicPairTransforms(train=False)` zinciriyle (flip kapalı, `Rescale`+`Normalize`+`Pad`+`ToTensor`)
  önişleniyor, `_process_query` ile template özellik/RoI'ları çıkarılıp saklanıyor.
- Sonraki her kare → `SiamDTTracker.update` (`siamdt_tracking.py:65-86`): `_process_gallary`
  önişlemesi + `SiamDTRCNN._process_gallary` (Bölüm 3.5'teki arka-plan bastırma/IoU güçlendirme
  post-processing'i) → en yüksek skorlu kutu `rescale=True` ile **orijinal görüntü koordinatlarına**
  geri ölçekleniyor (`bbox_head.get_bboxes(..., rescale=True)`, mmdet standart mekanizması).
- `Tracker.forward_test` (`libs/tracker.py:37-67`) tüm sekans için bu init/update döngüsünü
  çalıştırıp `bboxes` dizisini döndürüyor.
- `UAVtir_Eval.run`/`.eval` (`libs/data/evaluators/uavtir_eval.py:120-190, 93-118`) tahminleri
  `IR_label.json`'daki `gt_rect`/`exist` alanlarıyla karşılaştırıp **IoU tabanlı "Mixed Measure"**
  (hedef yoksa doğru "yok" tahmini de 1 puan sayılıyor — Anti-UAV benchmark'ının kendine özgü
  metriği) hesaplıyor; sonuçlar `results/<dataset>/<tracker_name>/<seq>.txt` ve
  `reports/.../performance.json` altına yazılıyor.

---

## 5. Değişiklik İçin Kritik Noktalar

### 5.1 En sık dokunulacak dosyalar

| Amaç | Dosya(lar) |
|---|---|
| Yeni bir korelasyon/similarity mekanizması denemek | `trackers/similarity_encoders.py` (`RPN_Similarity_Learning`, `RCNN_Similarity_Learning`) |
| Siamese forward mantığını değiştirmek (ör. farklı bir arka-plan bastırma stratejisi) | `trackers/siamdt_rcnn.py` — özellikle `_process_gallary` (satır 428-518) ve `forward_train` (65-185) |
| Backbone değiştirmek (ör. Swin-S/B, ResNet) | `configs/siamdt_swin_tiny_*.py` içindeki `model.backbone` dict'i + `libs/swintransformer/configs/_base_` altındaki hazır backbone config'lerinden esinlenmek; pretrained ağırlık için `utils/obtain_pretrained_weights.py`'ı güncellemek gerekir |
| Yeni loss eklemek/değiştirmek | `configs/*.py` içindeki `loss_cls`/`loss_bbox` dict'leri (mmdet registry, tip adını değiştirmek yeterli) — **`libs/ops/losses.py`'a değil**, çünkü o dosya kullanılmıyor |
| Yeni bir dataset entegre etmek | 1) `libs/data/datasets/` altına `SeqDataset` türeten yeni sınıf + `libs/data/datasets/__init__.py`'a ekleme 2) `datasets/wrappers.py`'daki `_datasets()` fonksiyonuna yeni `elif` dalı 3) gerekirse yeni bir Evaluator (`libs/data/evaluators/`) |
| Augmentasyon/ön-işleme değiştirmek | `libs/data/transforms/pair_transforms/mmdet_transforms.py` (`BasicPairTransforms`, `ExtraPairTransforms`) |
| Eğitim hiperparametreleri (optimizer, lr schedule, epoch sayısı) | `configs/siamdt_swin_tiny_adamw.py` / `_sgd.py` alt kısmı (`optimizer`, `lr_config`, `runner`) |
| Değerlendirme/metrik mantığı | `libs/data/evaluators/uavtir_eval.py` (`eval`, `_calc_metrics`, `_calc_curves`) |

### 5.2 Kırılgan / dikkat edilmesi gereken noktalar

1. **Hardcoded veri yolları.** `datasets/wrappers.py:17-30` içinde GOT-10k, LaSOT, UAVtir gibi
   veri setlerinin kök dizinleri doğrudan koda gömülü
   (`/media/data2/TrackingDatasets/GOT-10k/...`, `/media/data2/TrackingDatasets/Anti-UAV410/...`).
   README de bunu doğruluyor: *"Change the dataset path in `datasets/wrappers.py`"*. Yeni bir
   makinede/ortamda çalıştırmadan önce bu yolları güncellemek şart.
   Aynı şekilde `tracking_test_demo.py:23`'te evaluator kök dizini hardcoded, `utils/run_antiuav.py:88`'de
   `videofilepath='/data3/publicData/antiUAVtestimages/'` hardcoded, `utils/run_antiuav.py:2`'de
   `CUDA_VISIBLE_DEVICES = "2"` sabitlenmiş.

2. **`cache/` klasörü ile dataset senkronizasyonu.** `SeqDataset.__init__`
   (`libs/data/datasets/dataset.py:24-32`) sekans sözlüğünü `cache/<name>.pkl` içine yazıp bir
   daha diskten okumuyor. Dataset dizinini değiştirirsen ama eski `.pkl` cache dosyası duruyorsa,
   eski (yanlış) yol/liste kullanılmaya devam eder → README'deki "Issue 1" hatası. **Dataset yolu
   veya içeriği her değiştiğinde `cache/` klasörünü elle silmek gerekiyor.**

3. **Batch boyutu varsayımı = 1.** `SiamDTRCNN.forward_train` ve `simple_test_bboxes` /
   `_process_gallary` kodunun büyük bölümü `gt_bboxes_x[0]`, `img_meta_x[0]`, `sampling_result` gibi
   **tek elemanlı listeler** varsayıyor (`siamdt_rcnn.py:123-134, 215-216`). Config'te de
   `imgs_per_gpu=1` (`configs/siamdt_swin_tiny_adamw.py:166`, yorumda "origin: 2" — yani orijinal
   değer 2 iken bilinçli olarak 1'e düşürülmüş). **`imgs_per_gpu`'yu 1'den büyük yapmak kodu
   kırar**; multi-batch desteklemek istiyorsan bu index'leme mantığını (`[0]` ile sabitlenmiş
   yerler) genellemek gerekir.

4. **`RCNN_Similarity_Learning.forward` tek instance varsayımı.** `assert len(z) == 1`
   (`trackers/similarity_encoders.py:52`) — template olarak sadece **tek bir hedef** destekleniyor;
   çoklu nesne takibi (MOT) için bu modül olduğu gibi kullanılamaz.

5. **`up_flag` / online template güncelleme eşiği sabit.** `_process_gallary`'de
   (`siamdt_rcnn.py:500`) `tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9` ve `IoU>0.8` gibi **elle
   ayarlanmış sabit eşikler** template'in ne zaman güncelleneceğini belirliyor; farklı bir
   dataset/domain'de bu eşiklerin yeniden ayarlanması gerekebilir. Benzer şekilde `Top_NUM=5`
   (satır 456) ve `_process_query`'deki arka-plan proposal sayısı sınırı (`>10`, satır 382, 418)
   sabit kodlanmış.

6. **`libs/ops/losses.py` ölü kod.** Bölüm 3.4'te belirtildiği gibi, bu dosyadaki hiçbir fonksiyon
   SiamDT tarafından çağrılmıyor (grep ile doğrulandı) — kafa karıştırmaması için, yeni bir loss
   eklerken buraya değil mmdet'in loss registry'sine bakmak gerekiyor.

7. **`utils/run_antiuav.py` muhtemelen bozuk/eski.** `mmdet.apis.inference_detector`'ı doğrudan
   `SiamDTRCNN` (Siamese, `img_z`+`img_x` gerektiren bir `forward` imzasına sahip) üzerinde
   çağırıyor (`utils/run_antiuav.py:84,140`) — ama standart `inference_detector` tek görüntü
   bekler ve `SiamDTRCNN.forward_test`'i çağırmaz (`forward()` her zaman `img_z`/`img_x` ister,
   `siamdt_rcnn.py:47-59`). Bu script'in güncel API ile çalıştığından emin olmadan kullanma;
   asıl doğrulanmış test yolu `tracking_test_demo.py` + `SiamDTTracker`'dır.

8. **`show_result` ve `aug_test` kasıtlı olarak devre dışı.** `SiamDTRCNN.aug_test` /
   `show_result` (`siamdt_rcnn.py:310-316`) `NotImplementedError` fırlatıyor — mmdet'in genel
   test/görselleştirme araçlarını (`tools/test.py` gibi) SiamDT modeliyle doğrudan kullanmaya
   çalışırsan burada patlar; görselleştirme için bunun yerine `visdom` tabanlı
   `libs/ops` / `Tracker.visualize` mekanizması (`libs/tracker.py:22-23,64-65`) kullanılmış.

9. **İki config dosyası arasında sessiz farklar.** `siamdt_swin_tiny_sgd.py` ile
   `siamdt_swin_tiny_adamw.py` sadece optimizer'da değil, `reg_class_agnostic` (True/False) ve
   loss tipinde de (`SmoothL1Loss` vs `L1Loss`) farklılaşıyor — hangi checkpoint'i hangi config'le
   yüklediğine dikkat etmek gerekiyor, aksi halde `bbox_head` boyut uyuşmazlığı (class-agnostic
   regresyon 4 kanal, class-specific `4*num_classes` kanal bekler) checkpoint yükleme hatası verir.

---

*Bu doküman, konuşma sırasında yapılan kod incelemesinin bir özetidir; sonraki sorular için
başlangıç noktası olarak kullanılabilir.*
