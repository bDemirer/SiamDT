# SiamDT — Arama (Tracking Update) Adımları ve Terim Sözlüğü

> Bu doküman, `SIAMDT_KOD_ANALIZI.md` raporu üzerine yapılan soru-cevap sohbetinden derlenmiştir.
> Amaç: her yeni karede hedefin nasıl arandığını adım adım ve geçen terimleri (Swin Transformer,
> FPN, RPN, RoI, global search) açıklamak. Tüm satır referansları repo köküne göredir.

---

## 1. Terim Sözlüğü

### 1.1 Swin Transformer (backbone)

Microsoft'un vision transformer'ı — ViT'ten iki temel farkla ayrılır:

- **Lokal (window) attention:** Görüntü küçük pencerelere bölünür (bu repoda `window_size=7`) ve
  self-attention sadece pencere içinde hesaplanır → maliyet görüntü boyutuyla lineer artar
  (ViT'teki global attention'ın karesel `O(n²)` maliyetinin aksine).
- **Shifted window:** Ardışık bloklarda pencereler yarım pencere kaydırılarak komşu pencereler
  arasında da bilgi akışı sağlanır.
- **Hiyerarşik yapı:** CNN'ler gibi kademeli olarak downsample eder (patch merging) — her aşamada
  çözünürlük yarıya iner, kanal sayısı ikiye katlanır. Bu sayede FPN gibi CNN-tabanlı dedektör
  bileşenleriyle doğrudan uyumludur.

**Bu repodaki kullanımı** (`configs/siamdt_swin_tiny_adamw.py:9-24`):
- `embed_dim=96`, `depths=[2,2,6,2]`, `num_heads=[3,6,12,24]` → **Swin-Tiny** varyantı.
- `out_indices=(0,1,2,3)` → 4 aşamanın tamamı `neck: FPN`'e veriliyor.
- COCO'da önceden eğitilmiş Cascade Mask R-CNN ağırlıklarından başlatılıyor
  (`pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar`).

### 1.2 FPN (Feature Pyramid Network — neck)

Backbone'un farklı çözünürlükteki 4 çıktısını (Swin'in 4 aşaması) alıp, hepsini aynı kanal sayısına
(256) projekte ederek **5 seviyeli** bir özellik piramidi (P2-P6) oluşturur.

**Config** (`configs/siamdt_swin_tiny_adamw.py:25-29`):
```python
neck=dict(type='FPN', in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)
```

**Kullanıldığı yerler:**
- `RPN_Similarity_Learning`, FPN'in her 5 seviyesinde ayrı ayrı çalışır (`featmap_num=5`).
- `roi_head.bbox_roi_extractor` (RoIAlign), template (z) ve proposal (x) özelliklerini bu FPN
  seviyelerinden keser.

### 1.3 RPN (Region Proposal Network)

İki aşamalı dedektörlerin (Faster/Cascade R-CNN) **birinci aşaması**. Özellik haritası üzerinde
önceden tanımlı **anchor** kutuları için iki şey tahmin eder:
- **Objectness skoru:** burada nesne var mı yok mu (ikili sınıflandırma).
- **Kutu regresyonu (delta):** anchor'ı gerçek kutuya oturtmak için gereken kaydırma/ölçekleme.

Yüksek skorlu anchor'lar NMS ile elenip **proposal** (aday kutu) olarak ikinci aşamaya (RoI head)
gönderilir.

**Config** (`configs/siamdt_swin_tiny_adamw.py:30-40`): standart mmdet `RPNHead`, FPN'in 5
seviyesinde (`strides=[4,8,16,32,64]`), her seviyede 3 en-boy oranıyla (`ratios=[0.5,1.0,2.0]`)
anchor üretir.

**SiamDT farkı:** RPN, ham FPN çıktısı yerine `RPN_Similarity_Learning`'in ürettiği `x_corr` ile
modüle edilmiş özellikleri görür (`siamdt_rcnn.py:101-104` eğitimde, `:437-441` testte) — yani
"nesne var mı" değil, **"template'e benzeyen nesne var mı"** sorusuna cevap arar.

### 1.4 RoI (Region of Interest) / RoIAlign

RPN'in ürettiği her proposal (veya template'in gt kutusu) için, özellik haritasından o bölgeye denk
gelen kısmı sabit boyutlu (7×7) bir özellik haritasına kesip örnekleyen işlem
(`roi_head.bbox_roi_extractor`, mmdet `SingleRoIExtractor`). İki aşamalı dedektörün **ikinci
aşamasının** (RCNN / bbox head) girdisini hazırlar.

**Not:** RoI, aramanın kapsamıyla (global mi lokal mi) ilgili değil — sadece "aday kutunun içindeki
özelliği nasıl sabit boyuta getiririm" sorusuna cevap veren bir kesme/örnekleme yöntemidir.

### 1.5 Global Search vs. Local Search

- **Local search (klasik trackerlar — SiamFC, SiamRPN, KCF vb.):** Önceki karedeki hedef konumunun
  etrafında **küçük bir kırpılmış pencerede** arama yapılır. Hedef hızlı hareket eder veya oklüzyona
  girerse pencere dışına çıkabilir → tracker hedefi kaybeder.
- **Global search (SiamDT):** Her karenin **tamamı** (`img_x`, kırpılmamış) arama alanı olarak
  kullanılır — önceki kareye göre hiçbir crop/kayma uygulanmaz
  (`x_src = self.extract_feat(img_x)`, `siamdt_rcnn.py:432`). Bu sayede hedef oklüzyondan sonra
  görüntünün herhangi bir yerinde tekrar ortaya çıksa bile yeniden bulunabilir.

RPN + RoI mekanizması, bu global arama alanı **içinde** adayları belirleyen standart iki-aşamalı
dedektör altyapısıdır — "global" olan şey arama alanının kapsamı, RPN/RoI ise o kapsam içindeki aday
belirleme/doğrulama yöntemidir. İkisi çelişmez, farklı seviyelerdedir.

### 1.6 "Dual-Semantic Learning" ve "Versatile Learning" (kod içi isimlendirme)

Bu iki isim, repo içinde sadece **yorum satırı** olarak geçiyor (`trackers/siamdt_rcnn.py:98,148`
eğitimde; `:436` testte) — repo içinde bu terimleri tanımlayan bir makale referansı yok, bu yüzden
aşağıdaki açıklama **kod yapısından çıkarım**dır (bkz. `SIAMDT_KOD_ANALIZI.md` §1: orijinal makale
doğrulanamıyor).

**"Dual-Semantic Learning"** → RPN aşamasındaki yorum (`RPN_Similarity_Learning` çağrısının üstünde).
`forward_train` içinde (`siamdt_rcnn.py:90-104`) iki farklı **semantik kaynak** birleştiriliyor:

```python
rois_z = bbox2roi(gt_bboxes_z)
bbox_feats_z = self.roi_head.bbox_roi_extractor(z[...], rois_z)
template = [...]

# Dual-Semantic Learning
x_corr = next(self.rpn_similarity_learning(template, x))[0]
x = [x[i] + x_corr[i] for i in range(len(x))]     # x = x + x_corr
```

1. **Ham `x`** — Swin+FPN'den gelen, COCO ön-eğitiminden miras kalan genel-amaçlı "bu bir nesne mi"
   semantiği (template'ten bağımsız, sınıf-agnostik).
2. **`x_corr`** — `RPN_Similarity_Learning`'in template ile `x`'i kanal-bazlı çarparak ürettiği,
   **hedefe özgü** ("bu template'e mi benziyor") semantik.

Bu ikisi rezidüel olarak toplanıp (`x = x + x_corr`) tek bir birleşik özellik haline getiriliyor ve
RPN'e bu birleşik özellik veriliyor — "dual" (çift/ikili) ismi, bu **iki semantiğin RPN girdisinde
kaynaştırılmasına** işaret ediyor gibi görünüyor. Önemli detay: bu aşamada ham `x` için ayrı bir
RPN kaybı **yok** — tek bir RPN kaybı (`rpn_head.loss`, `:113`) doğrudan bu birleştirilmiş özellik
üzerinden hesaplanıyor.

**"Versatile Learning"** → RCNN aşamasındaki yorum (Bölüm 3.4'teki `RCNN_Similarity_Learning`
çağrısının üstünde). Buradaki fark: gerçekten **iki ayrı kayıp** hesaplanıp toplanıyor
(`siamdt_rcnn.py:148-183`):

```python
# Versatile learning
bbox_feats_corr = self.rcnn_similarity_learning(bbox_feats_z, bbox_feats_x)
cls_score_corr, bbox_pred_corr = self.roi_head.bbox_head(bbox_feats_corr)
loss_bbox_corr = self.roi_head.bbox_head.loss(cls_score_corr, bbox_pred_corr, rois_x, *bbox_targets)

cls_score_x, bbox_pred_x = self.roi_head.bbox_head(bbox_feats_x)   # ham, template'siz
loss_bbox = self.roi_head.bbox_head.loss(cls_score_x, bbox_pred_x, rois_x, *bbox_targets)

# iki kaybı topla → loss_bbox
```

Yani model, RCNN aşamasında **hem** template ile eşleştirilmiş kutu tahminini **hem de** ham/
template-siz kutu tahminini aynı anda öğreniyor (çift görev/çoklu-görev düzenlileştirme) — "çok
yönlü" (versatile) isim bunu vurguluyor.

**Özetle ayrım:** "Dual-Semantic" **girdi tarafında** iki semantik kaynağın (genel + template-özgü)
füzyonunu; "Versatile" ise **çıktı/kayıp tarafında** çift görevli (template'li + template'siz)
eğitimi ifade ediyor gibi görünüyor. Bu yorum tamamen kod okumasına dayanıyor; orijinal makalede
farklı tanımlanmış olabilir.

---

## 2. Arama (Update) Adımları — `_process_gallary`

Kaynak: `trackers/siamdt_rcnn.py:428-518`, çağrı zinciri `siamdt_tracking.py:65-86` (`update()`) →
`SiamDTRCNN._process_gallary`.

**Ön koşul:** `init()` sırasında (`_process_query`, `siamdt_rcnn.py:351-391`) zaten çıkarılmış olan:
- `self._template` / `self._bbox_feats_z` — hedefin 7×7×256 template özelliği,
- `self._bbox_feats_bg` — hedef kutusunun dışındaki (IoU=0) proposal'lardan çıkarılmış **arka plan**
  özellikleri (en fazla 10 tane).

Her yeni karede sırayla:

**0. Girdi:** Ham görüntü. Önceki karenin konumuna dair hiçbir bilgi kullanılmaz (crop/kayma yok).

**1. Özellik çıkarma** (`:432`)
```python
x_src = self.extract_feat(img_x)
```
Swin backbone + FPN, tüm karenin 5 seviyeli özellik piramidini çıkarır.

**2. Template ile modülasyon — Dual-Semantic Learning** (`:437-441`)
```python
x_corr = next(self.rpn_similarity_learning(self._template, x))[0]
x = [x[i] + x_corr[i] for i in range(len(x))]
```
`RPN_Similarity_Learning`, template'i her FPN seviyesinde `x` ile kanal-bazlı çarpar (`x_corr`);
bu, orijinal `x`'e eklenir. Özellik haritası artık "burada hedefe benzeyen bir şey var mı" bilgisini
taşır.

**3. Proposal üretimi (RPN)** (`:443-444`)
```python
proposal_list = self.rpn_head.simple_test_rpn(x, img_meta_x)
```
Modüle edilmiş `x` üzerinden, **tüm görüntüde** aday kutular üretilir — crop/pencere kısıtlaması
yok. "Global search"ün gerçekleştiği adım budur.

**4. RCNN doğrulama — iki paralel dal** (`simple_test_bboxes`, `:187-263`)
Her proposal için RoIAlign ile özellik kesilir, sonra iki ayrı yoldan skorlanır:
- **`tra_bboxes`** ("tracking"): `RCNN_Similarity_Learning` ile template'e karşı eşleştirilmiş skor
  (`bbox_feats_z * bbox_feats_x`).
- **`det_bboxes`** ("detection"): ham `bbox_feats_x` ile, template'siz standart sınıflandırma skoru.

Her aday hem "template'e ne kadar benziyor" hem "genel olarak ne kadar nesne-gibi" açısından ayrı
ayrı puanlanır.

**5. Birleştirme ve Top-5 seçimi** (`:452-460`)
`tra_bboxes` + `det_bboxes` birleştirilip skora göre sıralanır, en iyi **5 aday** (`Top_NUM=5`)
alınır.

**6. Arka plan bastırma (background suppression)** (`:462-478`)
Bu 5 adayın her biri, `init()`'te saklanan arka plan özellikleriyle (`self._bbox_feats_bg`) tekrar
eşleştirilir (`simple_matching`). Aday arka plana da çok benziyorsa (ayırt edici değilse), skoru
düşürülür — drone'a benzeyen bulut/bina gibi yanlış pozitifleri elemek için.

**7. IoU ile skor güçlendirme** (`:482-497`)
5 aday birbiriyle karşılaştırılır; iki aday çakışıyorsa (IoU > 0.8), biri diğerinin skorunu
artırır — aynı hedefi işaret eden farklı proposal'lar birbirini teyit eder.

**8. En iyi kutu seçimi ve template güncelleme kararı** (`:500-504`)
```python
if tra_bboxes[0,-1] + det_bboxes[0,-1] > 1.9 and iou(det_bboxes[0], tra_bboxes[0]) > 0.8:
    up_flag = True
    self._update_query(x_src, proposal_list, [new_bbox], img_meta_x)
```
En yüksek skorlu `tra_bboxes[0]`/`det_bboxes[0]` çiftinin toplam güveni > 1.9 **ve** birbirleriyle
IoU > 0.8 ise tespit güvenilir sayılır; `up_flag=True` olur ve `_update_query` ile template
**EMA** (exponential moving average) ile güncellenir: `0.99*eski + 0.01*yeni`
(`self._learning_rate=0.01`, `_process_query` içinde `siamdt_rcnn.py:353` tanımlanır) — drone'un
görünümü zamanla değiştiği için.

**9. Sonuç** (`:510-518`, `siamdt_tracking.py:65-86`)
En yüksek skorlu kutu, `rescale=True` ile orijinal görüntü koordinatlarına geri ölçeklenip
`update()`'ten döner; `up_flag` de tracker'a template'in güncellenip güncellenmediğini bildirir.

### Özet akış

```
özellik çıkar (Swin+FPN, tüm kare)
  → template ile modüle et (RPN_Similarity_Learning, x_corr)
    → tüm görüntüde proposal üret (RPN — global arama burada gerçekleşir)
      → her adayı hem template'e (tra) hem genel nesneliğe (det) göre skorla (RCNN_Similarity_Learning)
        → en iyi 5'i al, arka planla karşılaştırıp elemine et (background suppression)
          → çakışan adayları birbirine destekle (IoU boost)
            → en iyisini seç; güvenliyse template'i EMA ile güncelle (up_flag)
```

---

## 3. Adımların Detaylı Mekaniği ("neyi nasıl" seviyesinde)

Bu bölüm, Bölüm 2'deki her adımın **içeride tam olarak ne hesapladığını** (tensör boyutları,
formüller, config sabitleri) açıklıyor.

### 3.1 Özellik çıkarma (`extract_feat`)

Swin backbone, görüntüyü 4×4'lük patch'lere bölüp gömme (embedding) uyguluyor, sonra 4 aşamadan
geçiriyor (`depths=[2,2,6,2]`) — her aşamada pencere içi self-attention + shifted-window
self-attention blokları çalışıyor, aşama sonunda **patch merging** ile çözünürlük yarıya iniyor.
Çıkan 4 harita (`[96,192,384,768]` kanal) FPN'e giriyor; FPN lateral 1×1 konvolüsyonlarla hepsini
256 kanala indirip üst-aşağı (top-down) toplayarak 5 seviyeli piramit (`P2..P6`) üretiyor. Sonuç:
5 tensör, her biri `[1, 256, H_k, W_k]` (k arttıkça H,W küçülüyor, stride'lar `[4,8,16,32,64]`).

### 3.2 Template ile modülasyon — `RPN_Similarity_Learning.forward` (`similarity_encoders.py:22-33`)

```python
query = template[i][j:j+1]          # [1, 256, 7, 7]  — tek hedefin template özelliği
gallary = [f[i:i+1] for f in feats_x]  # 5 FPN seviyesi, her biri [1, 256, H_k, W_k]
out_ij = [self.proj_query[k](query) * gallary[k] for k in range(5)]
out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
```

- `proj_query[k]`: `Conv2d(256, 256, kernel_size=7, padding=0)`. Girdi tam 7×7 olduğu için çıktı
  uzamsal olarak **1×1**'e küçülüyor — yani 7×7'lik template özelliği, her biri 256 kanallı **tek
  bir vektöre** (1×1×256) sıkıştırılıyor. Bu, template'in "imzası" gibi düşünülebilir.
- Bu 1×1×256 vektör, `gallary[k]` (`H_k×W_k×256`) ile **broadcast çarpım** yapılıyor: her uzamsal
  konumdaki 256 kanal, template vektöründeki karşılık gelen kanal değeriyle ölçekleniyor. Bu,
  **uzamsal bir kayan-pencere korelasyonu (cross-correlation) değil** — kanal bazlı bir kapı/ağırlık
  (channel gating) mekanizması: "template'te hangi kanallar baskınsa, x'in her yerinde o kanalları
  öne çıkar" mantığı.
- `proj_out[k]`: `Conv2d(256, 256, kernel_size=1)` — kanallar arası karışım yaparak sonucu tekrar
  projekte ediyor.
- Çıkan 5 harita, orijinal `x` ile toplanıyor (`x = x + x_corr`) — **rezidüel (residual)** bir
  ekleme, yani orijinal genel-amaçlı nesne bilgisini bozmadan üstüne "template'e ilgi" sinyali
  ekleniyor.

### 3.3 Proposal üretimi — `rpn_head.simple_test_rpn` (standart mmdet `RPNHead`)

`test_cfg.rpn` (`configs/siamdt_swin_tiny_adamw.py:107-112`):
```python
rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)
```
1. Her FPN seviyesinde, her konumda `scales=[8]` × `ratios=[0.5,1.0,2.0]` = **3 anchor** üretiliyor
   (`strides=[4,8,16,32,64]` ile konumlandırılmış).
2. RPN'in konvolüsyon katmanları her konum/anchor için **objectness skoru** (sigmoid) ve **4 delta**
   (dx,dy,dw,dh) tahmin ediyor.
3. `DeltaXYWHBBoxCoder.decode` ile anchor + delta → gerçek kutu koordinatına çevriliyor
   (`target_means=0, target_stds=1`).
4. Her seviyede skora göre en iyi `nms_pre=1000` kutu tutuluyor (çok fazlaysa), tüm seviyeler
   birleştirilip **NMS** (`iou_threshold=0.7`) uygulanıyor, sonuçta en fazla `max_per_img=1000`
   proposal kalıyor. Bu proposal'lar **tüm görüntüye** yayılmış olabilir — global arama burada
   somutlaşıyor.

### 3.4 RCNN doğrulama — `simple_test_bboxes` (`siamdt_rcnn.py:187-263`)

1. `bbox2roi(proposals)` → proposal listesini `(batch_idx, x1,y1,x2,y2)` formatında tek bir `rois`
   tensörüne çeviriyor.
2. `roi_head.bbox_roi_extractor` (**RoIAlign**, `output_size=7`, `featmap_strides=[4,8,16,32]`):
   her proposal'ın boyutuna göre **hangi FPN seviyesinden** özellik keseceğine karar veriyor (küçük
   kutular ince seviyeden, büyük kutular kaba seviyeden — FPN makalesindeki standart seviye atama
   formülü), sonra o kutunun içindeki alanı **7×7×256**'ya interpolasyonla örnekliyor
   (`bbox_feats_x`, şekil `[N, 256, 7, 7]`, N = proposal sayısı).
3. **`RCNN_Similarity_Learning.forward`** (`similarity_encoders.py:50-53`):
   ```python
   proj_out(proj_x(x) * proj_z(z))
   ```
   - `proj_z`: `Conv2d(256,256, kernel=3, padding=1)` template'e (`z`, `[1,256,7,7]`) uygulanıyor —
     3×3 konvolüsyon olduğu için **uzamsal komşuluğu da karıştırıyor** (RPN aşamasındaki 1×1'e
     sıkıştırmadan farklı olarak, burada template'in 7×7'lik uzamsal yapısı korunuyor).
   - `proj_x`: aynı şekilde `bbox_feats_x`'e (`[N,256,7,7]`) uygulanıyor.
   - İkisi **elementwise çarpılıyor** (`z` batch=1 olduğu için `N` proposal'a broadcast ediliyor) —
     yani her proposal, template ile **konum konum (7×7'nin her hücresi ayrı ayrı)** karşılaştırılmış
     oluyor; RPN aşamasındaki tek-vektörlü kanal gating'e göre daha ince taneli bir eşleştirme.
   - `proj_out`: `Conv2d(256,256,kernel=1)` ile sonucu (`bbox_feats_corr`) tekrar projekte ediyor.
4. `bbox_feats_corr`, `roi_head.bbox_head` (`Shared2FCBBoxHead`: 2 paylaşılan FC katmanı + ayrı
   sınıflandırma/regresyon FC'leri) içinden geçiyor → `cls_score_corr` (softmax ile olasılık),
   `bbox_pred_corr` (delta, `target_stds=[0.1,0.1,0.2,0.2]`).
5. `get_bboxes`: `F.softmax(cls_score)` → skor; `bbox_coder.decode` ile proposal + delta → kesin
   kutu; görüntü sınırlarına clip; `rescale=True` ise `scale_factor`'e bölünüp **orijinal görüntü
   koordinatlarına** geri ölçekleniyor; `multiclass_nms` (`score_thr=0.0`, `nms iou=0.5`,
   `max_per_img=100`) ile son liste süzülüyor → **`tra_bboxes`**.
6. Aynı 4-5 adımlar, `bbox_feats_corr` yerine **ham `bbox_feats_x`** (template ile hiç
   karşılaştırılmamış) ile tekrarlanıyor → **`det_bboxes`**. Yani aynı proposal'lar, biri
   "template'e göre", diğeri "template'siz genel nesnelik" açısından iki kez skorlanmış oluyor.

### 3.5 Top-5 ve arka plan bastırma (`siamdt_rcnn.py:452-478`)

1. `tra_bboxes` + `det_bboxes` birleştirilip (`ens_bboxes`) skora göre azalan sıralanıyor, ilk
   **5 kutu** (`Top_NUM=5`) alınıyor.
2. Her 5 aday için: o adayın kutusundan **tekrar RoIAlign** ile tek bir `bbox_feats_mm` (`[1,256,7,7]`)
   çıkarılıyor. Bu sefer `simple_matching(bbox_feats_mm, self._bbox_feats_bg, ...)` çağrılıyor —
   yani **rol değişiyor**: aday artık "template" (z) rolünde, `init()` sırasında ilk karede
   çıkarılmış ~10 **arka plan** proposal özelliği (`self._bbox_feats_bg`) ise "galeri" (x) rolünde.
   Aynı `RCNN_Similarity_Learning` + `bbox_head` + `get_bboxes` zinciri çalışıp, adayın her arka
   plan parçasıyla ne kadar eşleştiğini gösteren skorlar (`mm_bboxes`) üretiliyor.
3. `mm_max_scores` = adayın **en çok benzediği** arka plan parçasının skoru. Bu, adayın kendi
   skorundan çıkarılıyor (`newens_bboxes[mm,-1] -= mm_max_scores`). Yani: "bu aday drone'a
   benziyor ama aynı zamanda arka plandaki bir kümeye/binaya da çok benziyorsa, güvenini düşür."

### 3.6 IoU ile karşılıklı destekleme (`siamdt_rcnn.py:480-497`)

5 aday ikili ikili karşılaştırılıyor (`computeiou`, klasik kesişim/birleşim formülü). İki aday
`IoU > 0.8` ile çakışıyorsa (yani aslında aynı nesneyi işaret eden farklı proposal'lar), biri
diğerinin **orijinal (bastırma öncesi) skorunun** `IoU` ile ağırlıklandırılmış bir kısmını kendi
skoruna ekliyor (`ref_scores[jj] * iouvalue`). Bu, birbirini teyit eden tespitlerin öne çıkmasını
sağlayan basit bir oy birliği (consensus) mekanizması.

### 3.7 Karar ve template güncelleme (`siamdt_rcnn.py:500-504`, `_update_query`)

- Koşul: en yüksek skorlu `tra_bboxes[0]` ile `det_bboxes[0]`'in **toplam skoru > 1.9** (yani
  ikisi de neredeyse 1.0'a yakın, çok güvenli) **ve** ikisinin kutuları birbiriyle **IoU > 0.8**
  (yani template-tabanlı ve template-siz iki bağımsız yöntem aynı yerde hemfikir).
- Bu durumda `_update_query` çağrılıyor (`siamdt_rcnn.py:393-426`):
  - Yeni tespit edilen kutudan `bbox_feats_z` tekrar RoIAlign ile çıkarılıyor.
  - **EMA (exponential moving average)** ile template güncelleniyor:
    `self._bbox_feats_z = 0.99 * eski + 0.01 * yeni` (`_learning_rate=0.01`) — ani/gürültülü
    değişimlerin template'i bozmaması için yavaş bir kayan ortalama.
  - Arka plan havuzu da **bu karenin** RPN proposal'larından (IoU=0 olanlar, en fazla 10 tane)
    yeniden hesaplanıyor — yani arka plan referansı sadece ilk kareye değil, güncel sahneye de
    uyum sağlıyor.
- Koşul sağlanmazsa (`up_flag=False`) template ve arka plan havuzu **değişmeden** kalıyor — bir
  sonraki karede hâlâ en son güvenilir template kullanılıyor.

### 3.8 Son kutunun geri ölçeklenmesi

`rescale=True` parametresiyle `get_bboxes` içinde kutu, önişlemede uygulanan `scale_factor`'e
bölünerek (görüntü `BasicPairTransforms` ile `Rescale(1333,800)` + `PadToDivisor(32)` ile
büyütülüp/küçültülüp modele verilmişti) **orijinal, işlenmemiş kare koordinatlarına** geri
dönüştürülüyor — `Tracker.forward_test`'in ve nihayetinde evaluator'ın kullandığı koordinat
sistemi budur.

---

*Bu doküman `SIAMDT_KOD_ANALIZI.md` ile birlikte kullanılmak üzere hazırlanmıştır; ilgili bölümler
için o rapora bakılabilir (özellikle §3.2, §3.3, §3.5, §4.2).*
