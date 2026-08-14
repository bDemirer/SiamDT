# SiamDT — Sıfırdan Teknoloji ve Kod Akışı Raporu

> Amaç: Bu belgeyi okuyan biri, önceden mmdetection/Swin Transformer/RPN gibi kavramları hiç
> bilmese dahi, SiamDT'nin **ne yaptığını, hangi teknolojileri neden kullandığını ve bunları nasıl
> birleştirdiğini** uçtan uca anlayabilsin. `SIAMDT_KOD_ANALIZI.md` (genel mimari raporu) ve
> `SIAMDT_ARAMA_ADIMLARI.md` (arama adımlarının mekanik detayı) ile birlikte okunmak üzere
> hazırlanmıştır; burada onların üzerine **temel kavramların sıfırdan anlatımı** eklenmiştir.

---

## 0. Bir Cümlede SiamDT

SiamDT, ilk karede verilen bir kutudan (bbox) yola çıkarak, **termal (kızılötesi) video akışında bir
drone'u karesi karesi takip eden** bir yazılım. Klasik trackerların aksine, her karede hedefi
**önceki konumun etrafında değil, görüntünün tamamında yeniden arıyor** — bu yüzden hedef geçici
olarak kaybolup başka bir yerde tekrar ortaya çıksa bile onu bulabiliyor.

---

## 1. Problem Ne, Neden Zor?

**Girdi:** Bir video (kare dizisi) + ilk karede hedefin kutusu (x, y, genişlik, yükseklik).
**Çıktı:** Sonraki her karede hedefin kutusu.

**Neden klasik "takip et" yöntemleri yetersiz kalıyor?**
Geleneksel trackerlar (ör. KCF, SiamFC, SiamRPN) şu mantıkla çalışır: "hedef bir önceki karede
şurada idiyse, bu karede de yakın bir yerdedir" — bu yüzden sadece önceki konumun etrafında küçük
bir pencerede (**local search**) arama yaparlar. Bir drone hızlı hareket ederse, kamera sallanırsa,
ya da hedef birkaç kare kaybolup (bulut arkası, ekran dışı vb.) başka bir yerde tekrar görünürse, bu
pencere dışında kalır ve tracker hedefi tamamen kaybeder.

**SiamDT'nin çözümü — "tracking-by-detection":**
Her karede, hedefi aramak yerine, **o karenin tamamında "bu template'e (ilk karedeki hedefe) benzeyen
ne var?" sorusunu soran bir nesne dedektörü** çalıştırılıyor (**global search**). Yani SiamDT aslında
klasik anlamda bir "tracker" değil — her karede yeniden çalışan, **template-şartlı (conditional) bir
nesne dedektörü**. Bu rapordaki teknolojilerin çoğu, "nasıl hızlı ve doğru bir nesne dedektörü
kurarım" ve "bu dedektörü nasıl belirli bir template'e duyarlı hale getiririm" sorularına cevap
veriyor.

---

## 2. Kullanılan Teknolojiler — Sıfırdan Anlatım

### 2.1 PyTorch (temel çerçeve)

Tüm proje **PyTorch** üzerine yazılmış — Python'da sinir ağı (neural network) kurmak, eğitmek ve
çalıştırmak için kullanılan açık kaynak kütüphane. Katmanlar (`nn.Conv2d`, `nn.Linear` vb.) birer
Python nesnesi, ileri hesap (`forward`) Python koduyla tanımlanıyor, geri yayılım (backpropagation)
otomatik hesaplanıyor. Bu rapordaki tüm `nn.Module` türevleri (`RPN_Similarity_Learning`,
`RCNN_Similarity_Learning`, `SiamDTRCNN`) PyTorch modülleri.

### 2.2 Evrişimli Sinir Ağları (CNN) — çok kısa hatırlatma

Bir görüntüyü sayısal bir "özellik haritasına" (feature map) dönüştürmenin klasik yolu: küçük
filtreleri (kernel) görüntü üzerinde kaydırarak (convolution/evrişim) kenar, doku, şekil gibi
örüntüleri yakalamak. Ard arda konvolüsyon katmanları, önce basit örüntüleri (kenar), sonra daha
soyut olanları (tekerlek, kanat, gövde) öğrenir. Çıktı, orijinal görüntüden daha küçük ama her
"piksel"i daha zengin bilgi (çok kanallı) taşıyan bir harita olur. SiamDT'de bu rolü klasik bir CNN
değil, **Swin Transformer** üstleniyor (bkz. 2.4).

### 2.3 Nesne Tespiti (Object Detection) ve İki Aşamalı Dedektörler

**Nesne tespiti**, bir görüntüde "neresi, ne" sorusuna cevap veren görev — hem kutu (lokalizasyon)
hem sınıf (ne olduğu) tahmini gerektirir. SiamDT'nin temeli, bu alanın klasik ve güçlü
yöntemlerinden biri olan **iki aşamalı dedektör** (Faster R-CNN / Cascade R-CNN ailesi):

**1. Aşama — RPN (Region Proposal Network):**
Görüntünün özellik haritası üzerinde, önceden tanımlı boyut/oranlarda **çapa (anchor)** kutuları
gezdirilir. Her anchor için iki basit soru sorulur:
- **Objectness:** burada herhangi bir "nesne" olabilir mi (evet/hayır skoru)?
- **Kutu düzeltme (delta):** anchor'ı gerçek nesneye oturtmak için nasıl kaydırıp ölçeklemeli?

Yüksek skorlu anchor'lar, çakışanları eleyen bir algoritma olan **NMS (Non-Maximum Suppression)**
ile süzülüp **proposal** (aday kutu) olarak ikinci aşamaya gönderilir. RPN'in görevi "sınıf
tahmini" değil, sadece **"nerede bir şey olabilir"** sorusuna kabaca cevap vermek — hızlı ama kaba
bir ön filtre.

**2. Aşama — RoI / RoIAlign + RCNN başı (head):**
Her proposal için, özellik haritasından o bölgeye denk gelen kısım kesilip **sabit boyuta**
(SiamDT'de 7×7) örneklenir — buna **RoIAlign** denir (RoI = Region of Interest). Bu sabit boyutlu
parça, birkaç tam-bağlantılı (fully-connected) katmandan geçirilerek **kesin sınıf** ve **kesin kutu
düzeltmesi** tahmin edilir. Bu, RPN'in kaba tahminini inceltip doğrulayan asıl "karar verici"
aşama.

**NMS (Non-Maximum Suppression):** Aynı nesne için üretilen birbirine çok yakın (üst üste binen)
birden fazla kutudan sadece en yüksek skorluyu tutup gerisini eleyen basit bir algoritma —
hem RPN çıktısında hem RCNN çıktısında kullanılıyor.

### 2.4 FPN — Feature Pyramid Network

Nesneler görüntüde çok farklı boyutlarda olabilir (uzaktaki küçük bir drone / yakın çekimde büyük
bir drone). Tek bir çözünürlükteki özellik haritası hem küçük hem büyük nesneleri iyi temsil edemez.
**FPN**, backbone'un farklı derinliklerdeki (dolayısıyla farklı çözünürlükteki) çıktılarını alıp,
hepsini aynı kanal sayısına indirip üst-aşağı (top-down) bilgi akışıyla birleştirerek **5 seviyeli
bir piramit** oluşturur: düşük seviye = yüksek çözünürlük (küçük nesneler için), yüksek seviye =
düşük çözünürlük ama zengin anlamsal bilgi (büyük nesneler için).

### 2.5 Swin Transformer (backbone — özellik çıkarıcı)

**Transformer**, önce doğal dil işlemede popülerleşen, "her eleman diğer tüm elemanlara ne kadar
dikkat etmeli" (self-attention) mantığıyla çalışan bir mimari. Görüntülere uygulanan hali **Vision
Transformer (ViT)** — görüntü küçük parçalara (patch) bölünür, her patch bir "kelime" gibi işlenir.
Ancak ViT'te her patch tüm diğer patch'lerle etkileşime girdiği için maliyet görüntü boyutunun
karesiyle artar — büyük görüntülerde yavaşlar.

**Swin Transformer** iki fikirle bunu çözer:
- **Pencere (window) attention:** Attention hesaplaması sadece küçük pencereler içinde yapılır (bu
  projede `window_size=7`) → maliyet lineer.
- **Kaydırılmış pencere (shifted window):** Ardışık katmanlarda pencereler yarım pencere kaydırılır,
  böylece komşu pencereler arası bilgi de akar.
- **Hiyerarşik (piramit) yapı:** CNN'ler gibi kademeli olarak küçülür (patch merging) — bu sayede
  FPN gibi CNN-tabanlı bileşenlerle doğrudan uyumludur; ResNet'in "drop-in" yerine geçebilir.

SiamDT'de kullanılan varyant **Swin-Tiny** (`embed_dim=96`, `depths=[2,2,6,2]`,
`configs/siamdt_swin_tiny_adamw.py:9-24`) — 4 aşamalı, her aşamada [2,2,6,2] adet Transformer bloğu.
Ağırlıklar sıfırdan değil, **COCO veri setinde önceden eğitilmiş bir Cascade Mask R-CNN**
checkpoint'inden (`pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar`) başlatılıyor — yani
model, "genel olarak nesne nedir" bilgisini transfer öğrenme (transfer learning) ile hazır alıyor.

### 2.6 Siamese Ağlar ve Bu Projeye Uyarlanışı

**Siamese ağ**, iki farklı girdiyi (burada: hedefin ilk kare görüntüsü = **template/z**, ve o anki
kare = **search/x**) **aynı** ağırlıklı bir alt-ağdan (backbone+FPN) geçirip, çıkan iki özellik
haritasını **karşılaştırarak** ("bunlar ne kadar benziyor?") bir sonuç üreten mimari türü.
Klasik kullanım alanı yüz/imza doğrulama ("bu iki resim aynı kişi mi") ve SiamFC/SiamRPN gibi
trackerlar.

SiamDT'nin özgün katkısı tam burada: Siamese karşılaştırmayı, klasik iki-aşamalı dedektörün **hem
RPN hem RCNN aşamasına** gömüyor (bkz. Bölüm 3) — yani "iki resim benziyor mu" sorusunu tek bir
skalerle cevaplamak yerine, **dedektörün proposal üretme ve doğrulama mantığının içine** yediriyor.

### 2.7 mmdetection (mmdet) — kullanılan framework

**mmdetection**, OpenMMLab'ın PyTorch tabanlı, açık kaynak **nesne tespiti framework'ü**. Yukarıda
anlatılan tüm parçaların (backbone, FPN, RPN, RoIAlign, NMS, loss fonksiyonları, eğitim döngüsü)
hazır, test edilmiş, birbirine takılabilir (registry ile) implementasyonlarını sağlıyor:

- **Registry:** Her bileşen bir isimle (`type='RPNHead'`, `type='FPN'`, `type='SwinTransformer'`)
  kayıtlı; config dosyasında sadece ismi ve parametreleri yazılır, mmdet arkada ilgili sınıfı bulup
  örnekler.
- **Config sistemi:** Python `dict` tabanlı, `_base_` ile miras alınabilen ayar dosyaları — model
  mimarisi + eğitim/test hiperparametreleri + veri pipeline'ı hep buradan yönetilir
  (`configs/siamdt_swin_tiny_adamw.py`).
- **`apis`:** `train_detector`, `init_detector`, `inference_detector` gibi hazır üst-seviye
  fonksiyonlar.

SiamDT, mmdet'i **kendi başına bağımlılık olarak kullanmıyor**; `libs/swintransformer/` altında
mmdet + Swin-Transformer-Object-Detection'ın **tam bir kopyasını** (vendored) taşıyor. `SiamDTRCNN`
sınıfı, mmdet'in `TwoStageDetector` sınıfından türetilip üzerine SiamDT'ye özgü iki modül
ekleniyor (Bölüm 3).

### 2.8 Özet — Katman Katman Teknoloji Yığını

| Katman | Teknoloji | Görevi |
|---|---|---|
| Alt yapı | PyTorch | Sinir ağı tanımlama/eğitme/çalıştırma |
| Framework | mmdetection (vendored) | Registry, config, hazır dedektör bileşenleri, eğitim döngüsü |
| Backbone | Swin Transformer (Tiny) | Görüntüden çok ölçekli özellik haritası çıkarma |
| Neck | FPN | Farklı ölçekleri 5 seviyeli piramide birleştirme |
| Aşama 1 | RPN (+ SiamDT'nin `RPN_Similarity_Learning`'i) | Tüm görüntüde, template'e duyarlı kaba adaylar üretme |
| Aşama 2 | RoIAlign + RCNN başı (+ SiamDT'nin `RCNN_Similarity_Learning`'i) | Adayları template ile ince taneli karşılaştırıp kesin skor/kutu üretme |
| Üst mantık | SiamDT'ye özgü `Tracker`/`init()`/`update()` sarmalayıcısı | Video karesi karesi işleyip template'i online güncelleme |

---

## 3. SiamDT Mimarisi — Parçalar Nasıl Birleşiyor

### 3.1 `SiamDTRCNN` (`trackers/siamdt_rcnn.py:20-44`)

mmdet'in `TwoStageDetector`'ından türetilmiş; standart bileşenlere ek olarak **iki özgün korelasyon
modülü** taşıyor:

| Bileşen | Kaynak | Girdi | Çıktı |
|---|---|---|---|
| `backbone` (Swin-T) | mmdet registry | görüntü | çok ölçekli özellik haritaları |
| `neck` (FPN) | mmdet registry | backbone çıktıları | 5 seviyeli piramit |
| `rpn_head` | mmdet `RPNHead` | template-modüle x | proposal'lar |
| `roi_head` (RoIAlign + `Shared2FCBBoxHead`) | mmdet | özellik + proposal | sınıf skoru + kutu |
| `rpn_similarity_learning` | **SiamDT'ye özgü** | template + x | RPN'e eklenecek `x_corr` |
| `rcnn_similarity_learning` | **SiamDT'ye özgü** | template RoI + proposal RoI | eşleşme özelliği |

### 3.2 `RPN_Similarity_Learning` — "Dual-Semantic Learning" (`trackers/similarity_encoders.py:8-39`)

```python
class RPN_Similarity_Learning(nn.Module):
    def __init__(self, roi_size=7, channels=256, featmap_num=5):
        self.proj_query = nn.ModuleList([nn.Conv2d(channels, channels, roi_size, padding=0) for _ in range(featmap_num)])
        self.proj_out   = nn.ModuleList([nn.Conv2d(channels, channels, 1, padding=0) for _ in range(featmap_num)])

    def forward(self, template, feats_x):
        out_ij = [self.proj_query[k](query) * gallary[k] for k in range(len(gallary))]
        out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
```

- `template` (`bbox_feats_z`): hedefin 7×7×256'lık RoIAlign özelliği.
- `proj_query` (7×7 kernel): template'i **1×1×256'lık tek bir vektöre** sıkıştırır — bu, hedefin
  "imzası" gibi düşünülebilir.
- Bu vektör, x'in her FPN seviyesindeki her uzamsal konumundaki 256 kanalı **kanal bazında**
  çarpar (broadcast) — konum-konum bir eşleştirme değil, "template'te hangi kanallar baskınsa
  onları öne çıkar" tarzı kaba bir kapı (gating) mekanizması.
- `proj_out` (1×1 kernel) sonucu tekrar projekte eder.
- Çıkan `x_corr`, orijinal `x`'e **rezidüel olarak eklenir** (`x = x + x_corr`,
  `siamdt_rcnn.py:101-104`) — RPN artık hem genel nesnelik hem template-benzerliği bilgisini aynı
  anda görür. Kod içi yorumda buna **"Dual-Semantic Learning"** deniyor: iki semantik kaynağın
  (genel + template-özgü) RPN girdisinde kaynaştırılması.

### 3.3 `RCNN_Similarity_Learning` — "Versatile Learning" (`trackers/similarity_encoders.py:42-58`)

```python
class RCNN_Similarity_Learning(nn.Module):
    def __init__(self, channels=256):
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):
        return self.proj_out(self.proj_x(x) * self.proj_z(z))
```

- RPN'den gelen proposal'lar üzerinde RoIAlign ile çıkarılan `bbox_feats_x`, template
  `bbox_feats_z` ile **3×3 konvolüsyon sonrası, konum konum** (7×7'nin her hücresi ayrı ayrı)
  çarpılır — RPN aşamasındaki tek-vektörlü kaba karşılaştırmadan daha ince taneli.
- Çıkan `bbox_feats_corr`, standart `roi_head.bbox_head`'e verilip sınıf/kutu tahmini üretir.
- **Eğitimde** hem bu "corr" (template ile modüle) hem "ham" (template'siz) özellikle **iki ayrı
  kayıp** hesaplanıp toplanıyor (`siamdt_rcnn.py:148-183`) — model hem saf nesne tespiti hem
  template-eşleştirmesi öğrensin diye çoklu-görev (multi-task) düzenlileştirme. Kod içi yorumda
  buna **"Versatile Learning"** (çok yönlü öğrenme) deniyor.

### 3.4 Bağlantı Şeması

```
img_z (template kare) --> Swin+FPN --> z ---> RoIAlign(gt kutusu) --> bbox_feats_z (7x7x256, template)
                                                                              |
img_x (o anki kare)   --> Swin+FPN --> x                                    |
                              |                                             |
                              v                                             |
                    RPN_Similarity_Learning(template, x) --> x_corr         |
                              |                                             |
                    x = x + x_corr  (Dual-Semantic füzyon)                  |
                              |                                             |
                              v                                             |
                          rpn_head --> proposal'lar (GLOBAL arama burada)   |
                              |                                             |
                              v                                             |
                 RoIAlign(proposal'lar) --> bbox_feats_x                    |
                              |                                             |
                              +<------------ bbox_feats_z ------------------+
                              |
                              v
                RCNN_Similarity_Learning (Versatile) --> bbox_feats_corr
                              |
                              v
                      roi_head.bbox_head
                              |
                              v
                cls_score, bbox_pred --> final bbox (skor + kutu)
```

Test aşamasında ayrıca (`_process_gallary`, `siamdt_rcnn.py:428-518`, detayları
`SIAMDT_ARAMA_ADIMLARI.md` §2-3'te):
- En iyi 5 aday **arka plan bastırma** ile (template'in çevresindeki arka plan proposal'larına da
  benziyorsa skoru düşürülür) filtrelenir,
- Çakışan (IoU>0.8) adaylar birbirinin skorunu **artırır** (konsensüs),
- Güvenli bir tespit bulunursa template **EMA (üstel hareketli ortalama)** ile online güncellenir.

---

## 4. Klasör Yapısı (kısa)

```
SiamDT/
├── configs/            model + eğitim ayarları (mmdet-tarzı config)
├── trackers/           SiamDT'ye özgü ASIL mantık
│   ├── siamdt_rcnn.py         SiamDTRCNN modeli (forward_train, _process_query, _process_gallary)
│   ├── siamdt_tracking.py     SiamDTTracker: init()/update() sarmalayıcı
│   └── similarity_encoders.py RPN_Similarity_Learning, RCNN_Similarity_Learning
├── libs/
│   ├── model.py / tracker.py  soyut Model/Tracker taban sınıfları
│   ├── data/                  dataset sınıfları, evaluator'lar, transform'lar
│   └── swintransformer/       vendored mmdet + Swin backbone kopyası
├── datasets/wrappers.py       PairWrapper: eğitimde (template, search) çifti üreten sarmalayıcı
├── tracking_train_demo.py     EĞİTİM giriş noktası
├── tracking_test_demo.py      TAKİP TESTİ giriş noktası
└── detection_test_demo.py     SiamDT'siz, saf mmdet dedektör sağlık kontrolü
```

Detaylı açıklama için `SIAMDT_KOD_ANALIZI.md` §1.

---

## 5. Kod Akışı — Eğitim (`tracking_train_demo.py`)

1. `Config.fromfile(...)` ile config okunur, `build_detector` ile `SiamDTRCNN` örneklenir
   (backbone+neck+rpn_head+roi_head kurulur, `RPN_Similarity_Learning`/`RCNN_Similarity_Learning`
   eklenir).
2. `build_dataset(cfg.data.train)` ile `PairWrapper` kurulur — bu, farklı veri setlerinden
   (Anti-UAV410/GOT-10k/LaSOT/COCO) rastgele **(template, search) çiftleri** üreten mmdet-uyumlu bir
   dataset sarmalayıcısı.
3. `train_detector(...)` (mmdet standart eğitim döngüsü) her batch'te:
   - `PairWrapper.__getitem__` bir video sekansından rastgele iki kare (z, x) seçer, kutuları
     filtreler, augmentasyon uygular.
   - `SiamDTRCNN.forward_train(img_z, img_x, ...)` çağrılır → Bölüm 3'teki tüm akış çalışır → RPN
     kaybı + RCNN kaybı (corr + ham) hesaplanıp geri yayılım yapılır.

## 6. Kod Akışı — Test / Tracking (`tracking_test_demo.py`)

1. `SiamDTTracker` kurulur, eğitilmiş `.pth` checkpoint yüklenir.
2. `EvaluatorUAVtir` her video sekansı için `tracker.forward_test(img_files, init_bbox)` çağırır —
   `init_bbox`, sekansın ilk karesindeki **ground-truth kutudan** okunur (bu kod kendi başına "drone
   nerede" tespiti yapmıyor, bir başlangıç kutusuna ihtiyaç duyuyor).
3. **İlk kare:** `SiamDTTracker.init` → `_process_query` (`siamdt_rcnn.py:351-391`): template
   özelliği (`bbox_feats_z`) ve arka plan proposal özellikleri (`bbox_feats_bg`) çıkarılıp saklanır.
4. **Sonraki her kare:** `SiamDTTracker.update` → `_process_gallary` — Bölüm 3.4'teki tüm zincir
   (global proposal → çift skorlama → top-5 → arka plan bastırma → IoU konsensüsü → EMA güncelleme)
   çalışır, en iyi kutu döner. **Adım adım mekanik detay için:** `SIAMDT_ARAMA_ADIMLARI.md`.
5. `UAVtir_Eval` tahminleri gerçek etiketlerle karşılaştırıp IoU tabanlı bir skor
   ("Mixed Measure") hesaplar, sonuçları `results/`/`reports/` altına yazar.

---

## 7. Veri Formatı (kısa)

- **Ana veri seti:** Anti-UAV410 (kod içi kısa adı `uavtir`) — termal drone videoları,
  `IR_label.json` içinde her kare için `gt_rect` (`[x,y,w,h]`) ve `exist` (hedef görünür mü) alanı.
- COCO/GOT-10k/LaSOT/ImageNet-VID/VisDrone gibi genel amaçlı setler, eğitimde çeşitlilik için
  karışıma ekleniyor (`sampling_prob` ile ağırlıklandırılmış örnekleme).
- Her eğitim örneği bir **(template kare, search kare)** çifti — video setlerinde aynı sekanstan iki
  farklı kare, görüntü setlerinde (COCO gibi) aynı görüntünün iki farklı augment edilmiş kopyası.

Detay: `SIAMDT_KOD_ANALIZI.md` §4.

---

## 8. Kırılgan / Dikkat Edilmesi Gereken Noktalar (özet)

Tam liste ve satır referansları `SIAMDT_KOD_ANALIZI.md` §5.2'de. Öne çıkanlar:

1. **Hardcoded veri yolları** (`datasets/wrappers.py`) — yeni ortamda çalıştırmadan önce
   güncellenmeli.
2. **`cache/` klasörü** dataset değişince elle silinmeli, yoksa eski indeks kullanılır.
3. **Batch boyutu = 1 varsayımı** — kodun büyük kısmı tek-elemanlı liste indexleme yapıyor
   (`imgs_per_gpu=1` bilinçli olarak sabitlenmiş).
4. **Tek hedef varsayımı** (`RCNN_Similarity_Learning`'de `assert len(z)==1`) — çoklu nesne takibi
   (MOT) için doğrudan kullanılamaz.
5. **Sabit eşikler** (`>1.9` güven toplamı, `IoU>0.8`, `Top_NUM=5`) elle ayarlanmış; farklı
   domain/veri setinde yeniden ayarlanması gerekebilir.
6. **`libs/ops/losses.py` ölü kod** — hiçbiri kullanılmıyor, yanıltıcı olabilir.
7. **İlk kare kutusu zorunlu** — bu bir tracker, sıfırdan dedektör değil; otomatik başlatma için
   ayrı bir dedektör eklenmesi gerekir (bkz. bir önceki sohbette konuşulan detector-ekleme fikri).

---

## 9. Hızlı Terim Sözlüğü

| Terim | Kısaca |
|---|---|
| Backbone | Görüntüden özellik haritası çıkaran ana ağ (burada Swin Transformer) |
| Neck | Backbone çıktılarını birleştiren ara katman (burada FPN) |
| FPN | Farklı ölçekleri 5 seviyeli piramide birleştiren yapı |
| RPN | Kaba aday kutu (proposal) üreten birinci aşama |
| Anchor | RPN'in her konumda denediği önceden tanımlı kutu şablonları |
| Proposal | RPN'in çıkardığı, henüz sınıflandırılmamış aday kutu |
| RoI / RoIAlign | Bir kutunun içindeki özelliği sabit boyuta kesip örnekleme işlemi |
| NMS | Çakışan kutulardan en iyisini tutup gerisini eleme |
| Template (z) | İlk karedeki hedef özelliği — "aranan şey" |
| Search/Gallery (x) | O anki kare — "arama yapılan yer" |
| Siamese | İki girdiyi aynı ağırlıklı ağdan geçirip karşılaştıran mimari türü |
| Global search | Aramanın önceki konumla sınırlı olmayıp tüm karede yapılması |
| EMA | Üstel hareketli ortalama — template'in yavaşça güncellenme yöntemi |
| mmdetection | Bu projenin üzerine kurulduğu nesne tespiti framework'ü |

---

## 10. Bu Rapor Nasıl Kullanılmalı

- Sunum hazırlarken sıra: **Bölüm 1 (problem) → Bölüm 2 (teknolojiler, sıfırdan) → Bölüm 3
  (SiamDT'nin bunları nasıl birleştirdiği) → Bölüm 5-6 (uçtan uca kod akışı)**.
- Mekanik/matematiksel detaya inmek gerekirse `SIAMDT_ARAMA_ADIMLARI.md`'ye,
  değişiklik yapılacak dosyaları bulmak için `SIAMDT_KOD_ANALIZI.md` §5.1 tablosuna bakılabilir.
- Sıradaki adım (konuşmada belirtildiği gibi) bu mimariye **bir dedektör ekleyerek ilk karedeki
  kutuyu otomatikleştirmek** — bu değişikliğe başlamadan önce Bölüm 3.2-3.3'teki template
  bağımlılığının (`assert len(z)==1` dahil) iyi anlaşılmış olması önemli.

---

*Bu doküman, `SIAMDT_KOD_ANALIZI.md` ve `SIAMDT_ARAMA_ADIMLARI.md` ile birlikte kullanılmak üzere,
konuşma boyunca yapılan kod incelemesinin "sıfırdan anlatım" odaklı bir sentezidir.*
