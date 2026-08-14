# SiamDT — Olası Geliştirmeler (Artı/Eksi Analizi)

> Bu doküman, `SIAMDT_KOD_ANALIZI.md`'deki inceleme sırasında tespit edilen zayıf noktalardan ve
> mimari fırsatlardan yola çıkarak hazırlanmıştır. Her öneri için: **ne**, **nerede**, **artı**,
> **eksi** verilmiştir. Öncelik notu (Yüksek/Orta/Düşük) öneriyi ilk ele almanın getirisi/maliyeti
> oranına göre benim değerlendirmemdir; kesin bir gerçek değildir.

---

## A. Mimari / Model Geliştirmeleri

### A1. Sabit karar eşiklerini öğrenilebilir hale getirmek
**Ne:** `SiamDTRCNN._process_gallary` içindeki `Top_NUM=5`, `IoU>0.8` skor güçlendirme eşiği,
`tra+det skor>1.9` template güncelleme eşiği, arka-plan proposal sayısı `>10` gibi sabitler
(`trackers/siamdt_rcnn.py:456,477,496,500,382,418`) elle ayarlanmış. Bunları küçük bir MLP/kalibrasyon
katmanıyla ya da en azından config parametresi olarak dışa açmak.
- **Artı:** Farklı veri setlerine (görünür ışık vs termal, farklı çözünürlük) taşınabilirlik artar;
  hiperparametre araması otomatikleştirilebilir; A/B test etmek kolaylaşır.
- **Eksi:** Öğrenilebilir hale getirmek ek eğitim karmaşıklığı ve overfitting riski getirir; sadece
  config'e taşımak (öğrenmeden) düşük maliyetli ama sınırlı fayda sağlar — asıl kazanım ancak
  veri-driven kalibrasyonla gelir.
- **Öncelik:** Orta (config'e taşımak Düşük efor/Orta fayda; öğrenilebilir hale getirmek Yüksek efor).

### A2. Çoklu-instance (multi-object) template desteği
**Ne:** `RCNN_Similarity_Learning.forward` içindeki `assert len(z) == 1`
(`trackers/similarity_encoders.py:52`) tek hedefle sınırlıyor. Batch/instance eksenini koruyacak
şekilde (`z: [N,C,7,7]`, `x: [N,C,7,7]` broadcast çarpım) genellemek.
- **Artı:** Aynı mimari çoklu drone / çoklu obje takibi (MOT-tarzı) senaryolarına genişletilebilir;
  eğitimde `imgs_per_gpu`'yu 1'in üzerine çıkarmanın önündeki en büyük engel kalkar → eğitim hızı
  artar (GPU daha verimli kullanılır).
  bu (Bölüm 5.3'teki batch=1 kısıtını da çözer).
- **Eksi:** `_process_gallary`'deki arka-plan bastırma / IoU güçlendirme mantığı da paralel olarak
  vektörleştirilmeli — kod karmaşıklığı belirgin şekilde artar; regresyon testleri olmadan mevcut
  tek-instance davranışını bozma riski var.
- **Öncelik:** Yüksek (hem eğitim verimliliği hem gelecekteki MOT genişlemesi için temel taş).

### A3. Depthwise cross-correlation yerine daha ifadeli füzyon (Transformer/attention tabanlı)
**Ne:** `RPN_Similarity_Learning`/`RCNN_Similarity_Learning`'deki kanal-bazlı çarpım
(`proj_query(query) * gallary`) yerine SiamRPN++ sonrası literatürde yaygınlaşan
cross-attention (ör. TransT, STARK tarzı) katmanı denemek.
- **Artı:** Global bağlam yakalar (küçük/uzak drone hedeflerinde arka plan karmaşasını ayırt etmede
  daha güçlü olabilir — termal görüntülerde düşük kontrast sorunu için özellikle değerli); literatürde
  cross-correlation'a göre genelde daha yüksek başarı raporlanıyor.
- **Eksi:** Parametre ve FLOP artışı → gerçek-zamanlı takip hızını (FPS) düşürebilir; küçük IR
  hedeflerde (Anti-UAV410'da hedefler genelde çok küçük) attention'ın avantajı sınırlı kalabilir;
  eğitim verisi (Anti-UAV410 boyutu) yeterince büyük değilse transformer katmanı overfit edebilir.
- **Öncelik:** Orta — önce A4/A5 gibi daha ucuz iyileştirmelerle taban çizgisini yükseltip, sonra
  bunun getirisini ölçmek daha güvenli.

### A4. Çok-ölçekli / çoklu-kare template (temporal ensemble)
**Ne:** Şu an template yalnızca ilk karede çıkarılıp EMA ile güncelleniyor
(`_update_query`, `trackers/siamdt_rcnn.py:393-426`, `learning_rate=0.01`). Bunun yerine son N
karenin template özelliklerini bir havuzda tutup (ör. en güvenilir K tanesini) ortalama/attention
ile birleştirmek.
- **Artı:** Ani görünüm değişimlerine (drone dönüşü, ışık değişimi, kısmi kapanma) karşı daha
  dayanıklı olabilir; tek bir EMA parametresine bağımlılığı azaltır.
- **Eksi:** Bellek/hesap maliyeti artar (birden fazla template feature saklanmalı); template havuzuna
  hatalı (drift olmuş) bir kare girerse hatayı biriktirip daha da kötüleştirebilir — güvenilirlik
  filtresi (ör. sadece yüksek skorlu karelerde güncelle, zaten kısmen `up_flag` ile yapılıyor) şart.
- **Öncelik:** Orta.

### A5. Hareket/momentum bilgisini (motion prior) modele dahil etmek
**Ne:** Şu an her karede **tüm görüntüde** global arama yapılıyor (Kalman filtresi, önceki
konum/hız bilgisi kullanan bir search-region daraltma yok). Basit bir Kalman filtresi veya önceki
kutu konumuna göre bir "arama bölgesi önceliği" (soft prior olarak proposal skorlarına eklenen bir
mesafe cezası) eklenebilir.
- **Artı:** Yanlış-pozitifleri (görüntüdeki benzer görünümlü nesneler/gürültü) azaltır; küçük termal
  hedeflerde arka plan karmaşasını bastırmada mevcut IoU-tabanlı sezgisel yönteme (`siamdt_rcnn.py:482-497`)
  göre daha ilkeli bir çözüm olur; hesap maliyeti neredeyse sıfır (RPN sonrası proposal skorlarına
  eklenen basit bir çarpan/ceza).
- **Eksi:** Ani manevra yapan (hızlı yön değiştiren) drone'larda motion prior yanlış yönlendirebilir;
  ek bir state (hız/ivme) tutmak ve sıfırlama (kayıp/yeniden-tespit) mantığı gerektirir — implementasyon
  hatası riskini artırır.
- **Öncelik:** Yüksek (düşük maliyet/yüksek fayda oranı; "global search" tracker'ların bilinen zayıf
  noktası tam olarak budur).

**Neden Kalman/basit motion prior, optical flow değil?** Aynı problemi (hareket bilgisini modele
katmak) optical flow ile çözmek de akla gelebilir, ama bu senaryoda değeri sınırlı:

- **Kamera ego-motion'ı ayrımı bozuyor.** Optical flow'un asıl gücü, hareketli nesneyi durağan arka
  plandan ayırmak. Anti-UAV410 görüntüleri genelde **hareketli bir kamera platformundan** çekiliyor
  — bu durumda hem drone hem de arka plan optical flow'da "hareketli" görünür. Bu ayrımı gerçekten
  kullanabilmek için önce kamera hareketi telafisi (motion compensation) gerekir, bu da işi ciddi
  şekilde karmaşıklaştırır ve kapsam dışına taşırır.
- **Ek hesap maliyeti.** Dense optical flow (klasik Farneback ya da öğrenilmiş bir FlowNet) her
  karede ek bir hesaplama demek — gerçek-zamanlılık hedefiyle çelişebilir, halbuki Kalman-tabanlı
  motion prior neredeyse sıfır maliyetli (sadece proposal skorlarına eklenen bir mesafe cezası).
- **Kalman/motion prior aynı ihtiyacı çok daha ucuza karşılıyor.** Sadece hedefin kendi geçmiş
  konum/hız bilgisini kullanır, kamera hareketinden etkilenmez, mevcut `_process_gallary` akışına
  (Bölüm A5'teki gibi) küçük bir ek olarak entegre edilebilir.

**Sonuç:** Önce A5'i (Kalman/basit motion prior) deneyip yeterli olup olmadığını ölçmek, optical
flow'u ancak A5 yetersiz kalırsa (ör. çok manevralı hedeflerde) — ve o zaman da önce kamera hareketi
telafisiyle birlikte — değerlendirmek daha isabetli bir sıralama.

---

## B. Eğitim & Veri Pipeline Geliştirmeleri

### B1. Hardcoded veri yollarını config/ENV değişkenine taşımak
**Ne:** `datasets/wrappers.py:17-30` içindeki `/media/data2/TrackingDatasets/...` gibi mutlak yollar
ile `tracking_test_demo.py:23` ve `utils/run_antiuav.py:88`'deki sabit yollar, ortam değişkeni
(ör. `SIAMDT_DATA_ROOT`) veya `configs/*.py` içinde bir `data_root` alanına taşınmalı.
- **Artı:** Farklı makinelerde (yerel geliştirme, sunucu, Windows/Linux) tek satır değiştirerek
  çalıştırma imkânı; CI/otomasyon kurmayı kolaylaştırır; README'deki "yolu elle değiştir" adımını
  ortadan kaldırır.
- **Eksi:** Değişiklik riski düşük ama tüm dataset sınıflarının constructor imzalarını (`root_dir`
  parametreleri zaten var, sadece varsayılan değer/çağrı noktası değişecek) gözden geçirmek gerekir;
  geriye dönük uyumluluk için varsayılanları korumak gerekebilir.
- **Öncelik:** Yüksek (çok düşük efor, gerçek bir sürtünme noktasını çözüyor).

### B2. `cache/` klasörü için otomatik geçersiz kılma (invalidation)
**Ne:** `SeqDataset.__init__` (`libs/data/datasets/dataset.py:24-32`) dataset'i `cache/<name>.pkl`
içine yazıp bir daha güncellemiyor. `root_dir`/dosya sayısı/son değişiklik zamanına dayalı bir hash
imzası cache dosyasına eklenip, uyuşmazlıkta otomatik yeniden inşa edilebilir.
- **Artı:** README'deki "Issue 1" hatasının (`ValueError: need at least one array to concatenate`)
  kök nedenini ortadan kaldırır; kullanıcının elle `cache/` silmesi gerekmez, sessiz hatalara karşı
  koruma sağlar.
- **Eksi:** Hash hesaplama (özellikle büyük veri setlerinde dosya sayımı) küçük bir başlangıç
  maliyeti ekler; yanlış pozitif "cache geçersiz" tespiti gereksiz yeniden-tarama tetikleyebilir.
- **Öncelik:** Yüksek (bilinen, dokümante edilmiş bir kullanıcı sorununu çözüyor).

### B3. `imgs_per_gpu` / distributed eğitimi doğrulamak ve belgelemek
**Ne:** Config'te `imgs_per_gpu=1  # origin: 2` yorumu (`configs/siamdt_swin_tiny_adamw.py:166`)
ve kodun batch=1 varsayan index'leme mantığı (`gt_bboxes_x[0]`, vs.) var. A2 ile birlikte veya ondan
bağımsız olarak, en azından çoklu-GPU (`--gpus N` / `--launcher pytorch`) ile gerçek bir eğitim
denemesi yapılıp sonuç raporlanmalı.
- **Artı:** Eğitim süresi kısalır (Anti-UAV410 + GOT-10k + LaSOT karışımı büyük bir veri seti);
  mevcut haliyle "distributed çalışır mı" belirsizliği ortadan kalkar.
- **Eksi:** Eğer batch=1 varsayımı (A2 yapılmadan) sadece `--gpus` ile çoklu-GPU'ya çıkarılırsa,
  GPU-başı hâlâ batch=1 kalır (veri paralelliği), bu yüzden gerçek kazanç sınırlı olur — asıl
  darboğaz `imgs_per_gpu`'nun kendisi.
- **Öncelik:** Orta.

### B4. Karma hassasiyet (AMP/fp16) yolunu netleştirmek
**Ne:** Config'te hem `fp16 = None` yorumu ("do not use mmdet version fp16",
`configs/siamdt_swin_tiny_adamw.py:186`) hem de `EpochBasedRunnerAmp` / `DistOptimizerHook(use_fp16=True)`
(satır 183-194) var — iki farklı AMP mekanizması (mmdet'in kendi `fp16` config'i vs. NVIDIA
apex-tabanlı runner) aynı anda referans ediliyor, kafa karıştırıcı. Tek bir tutarlı yaklaşımda
sadeleştirmek (apex/`EpochBasedRunnerAmp` kalıyorsa mmdet fp16 referanslarını tamamen kaldırmak).
- **Artı:** Bellek kullanımı ve eğitim süresi azalır (Swin-T + FPN + iki similarity modülü orta
  boy bir model, IR görüntüler genelde yüksek çözünürlüklü); kod okunabilirliği artar.
- **Eksi:** `apex` kurulumu README'de zaten "Common Issue 4" olarak sorunlu işaretlenmiş — apex
  bağımlılığını sürdürmek platform/CUDA sürüm kırılganlığı taşır; mmdet'in native fp16'sına
  geçmek daha taşınabilir olur ama sayısal kararlılığı yeniden test etmek gerekir.
- **Öncelik:** Düşük/Orta (işlevsel bir hata değil ama teknik borç).

### B5. Ölü kodun (`libs/ops/losses.py`) temizlenmesi veya belgelenmesi
**Ne:** Hiçbir yerden çağrılmayan `balanced_bce_loss`, `focal_loss`, `ghmc_loss`, `ohem_bce_loss`,
`iou_loss`, `ghmr_loss`, `label_smooth_loss` fonksiyonları (bkz. önceki analiz, Bölüm 3.4).
- **Artı:** Yeni katılan bir geliştiricinin "bu loss'lar kullanılıyor mu?" diye vakit kaybetmesini
  önler; kod tabanı küçülür, statik analiz/linter gürültüsü azalır.
- **Eksi:** Silmek yerine sadece not eklemek daha güvenli olabilir — bu fonksiyonlar gelecekte
  RPN/RCNN loss'unu değiştirmek isteyenler için hazır bir başlangıç noktası olarak da işlev
  görüyor olabilir; tamamen silmek bu "referans kütüphane" değerini kaybettirir.
- **Öncelik:** Düşük (kozmetik/bakım işi, riski yok ama acil de değil).

### B6. Augmentasyon setini termal (IR) görüntülere göre özelleştirmek
**Ne:** `PhotometricDistort` (`libs/data/transforms/pair_transforms/mmdet_transforms.py:193-213`)
RGB fotometrik bozulmalar (parlaklık/kontrast/doygunluk/hue, kanal takası) uyguluyor — bunlar RGB
odaklı; IR görüntülerde "hue/saturation" kavramı anlamsız (genelde tek kanal, gri tonlama).
IR'a özgü augmentasyonlar (gürültü enjeksiyonu, sıcaklık-benzeri kontrast bozulması, düşük
çözünürlük simülasyonu) eklemek.
- **Artı:** Anti-UAV410 asıl hedef veri seti olduğu için (Bölüm 1) augmentasyonun domain'e uygun
  olması genelleme performansını doğrudan iyileştirebilir.
- **Eksi:** IR'a özgü augmentasyon tasarlamak deneysel bir iştir (hangi bozulmanın gerçek dünya
  IR gürültüsünü taklit ettiği belirsiz); yanlış tasarlanırsa performansı düşürebilir; ek
  hiperparametre araması gerektirir.
- **Öncelik:** Orta.

---

## C. Test / Inference-zamanı Takip Mantığı Geliştirmeleri

### C1. Kayıp/yeniden-tespit (re-detection) stratejisi
**Ne:** Şu an `up_flag` sadece template'in güncellenip güncellenmediğini işaretliyor
(`trackers/siamdt_rcnn.py:500-508`); hedefin tamamen kaybolduğu (ekran dışı, tam kapanma) durumlar
için özel bir "kayıp modu" (ör. arama alanını genişletme, çoklu-aday saklama, N kare boyunca düşük
güvenli tahminleri reddetme) yok.
- **Artı:** Anti-UAV410'da drone'lar sık sık kadraj dışına çıkıp geri girebiliyor; "occlusion/kayıp"
  farkındalığı olan bir tracker, `not_exist` metriğinde (bkz. `UAVtir_Eval.not_exist`,
  `libs/data/evaluators/uavtir_eval.py:86-91`) puan kazanabilir.
- **Eksi:** Durum makinesi (state machine) karmaşıklığı artar; "ne zaman kayıp say" eşiği yine elle
  ayarlanacak yeni bir hiperparametre olur — A1'deki sorunu çoğaltma riski var.
- **Öncelik:** Yüksek (doğrudan benchmark metriğini hedefliyor).

### C2. `SiamDTTracker.update` içinde toplu (batched) çoklu-ölçek arama
**Ne:** Şu an her kare tek bir ölçekte (`Rescale(1333,800)`) işleniyor. Küçük/uzak hedefler için
görüntüyü birden fazla ölçekte (ör. orijinal + 1.5x crop) tarayıp sonuçları birleştirmek.
- **Artı:** Küçük termal hedeflerin (birkaç piksel boyutunda drone) kaçırılma oranını azaltabilir.
- **Eksi:** Çıkarım süresi ölçek sayısıyla orantılı artar — gerçek zamanlılık gereksinimini
  zorlayabilir; A5 (motion prior) ile birlikte kullanılmazsa hesap israfı olabilir (her karede
  tüm görüntüyü çoklu ölçekte taramak yerine, sadece belirsizlik yüksekken tetiklemek daha akıllıca).
- **Öncelik:** Orta.

### C3. `run_antiuav.py`'ı düzeltmek veya kaldırmak
**Ne:** `utils/run_antiuav.py`, `SiamDTRCNN`'i (Siamese forward imzası gerektiren) standart mmdet
`inference_detector` ile çağırıyor (`utils/run_antiuav.py:84,140`) — bu muhtemelen artık çalışmıyor
(bkz. önceki analiz, Bölüm 5.7). Ya `SiamDTTracker` API'sini kullanacak şekilde güncellenmeli ya da
"deprecated" olarak işaretlenip/silinmeli.
- **Artı:** Kod tabanında yanıltıcı, çalışmayan bir "nasıl test edilir" örneğinin kalmasını önler;
  yeni katılan geliştiricinin yanlış script'i kullanıp zaman kaybetmesini engeller.
- **Eksi:** Script'in orijinal amacı (belki farklı bir checkpoint/config ile plain-detector modu)
  netleştirilmeden silinirse, gizli bir kullanım senaryosu kaybedilebilir — önce script'in gerçekten
  ölü mü yoksa farklı bir workflow'un parçası mı olduğu teyit edilmeli.
- **Öncelik:** Düşük/Orta (risk düşük, fayda orta — temizlik işi).

### C4. Hız optimizasyonu: ONNX/TensorRT export veya TorchScript
**Ne:** Şu an sadece PyTorch eager-mode çıkarım var (`torch.no_grad()` dekoratörleriyle,
`trackers/siamdt_tracking.py:49,65`). Swin backbone + iki similarity modülünü ONNX/TensorRT'ye
taşımak.
- **Artı:** Gerçek zamanlı (edge/gömülü donanım gibi) anti-UAV senaryolarında kritik olan FPS'i
  ciddi oranda artırabilir (Swin Transformer dikkat mekanizması TensorRT ile hızlanmaya oldukça
  müsaittir).
- **Eksi:** Swin'in kayan pencere (shifted-window) attention'ı ve `RoIAlign`/dinamik proposal sayısı
  gibi mmdet operasyonları ONNX'e taşınırken genelde sorun çıkarır (dinamik shape desteği,
  custom op'lar); önemli bir mühendislik yatırımı gerektirir, muhtemelen bazı operasyonların elle
  yeniden yazılmasını gerektirir.
- **Öncelik:** Düşük (değerli ama yüksek efor; önce doğruluk odaklı A/B/C maddeleri önceliklenmeli).

---

## D. Mühendislik / Kod Kalitesi Geliştirmeleri

### D1. Birim/entegrasyon testleri eklemek
**Ne:** Repoda SiamDT'ye özgü hiç test dosyası yok (sadece vendored `libs/swintransformer/tests/`
mmdet'in kendi testleri). En azından `RPN_Similarity_Learning`/`RCNN_Similarity_Learning` için
şekil/gradyan testleri, `Seq2Pair._filter`/`_sample_pair` için birim testleri, `UAVtir._construct_seq_dict`
için küçük bir sahte (fixture) veri setiyle entegrasyon testi eklemek.
- **Artı:** A/B/C maddelerindeki değişiklikler yapılırken regresyonları erken yakalar; refactoring
  güvenini artırır (özellikle A2 gibi index'leme mantığını değiştiren riskli işler için kritik).
- **Eksi:** Başlangıç yatırımı gerektirir; mmdet'in ağır bağımlılıkları (CUDA, mmcv-full derlemesi)
  yüzünden CI kurmak, bu repo için özellikle zahmetli olabilir (README'deki kurulum sorunlarına bakılırsa).
- **Öncelik:** Yüksek (uzun vadede tüm diğer değişikliklerin güvenliğini artırıyor).

### D2. İki config dosyasını ortak bir `_base_`'de birleştirmek
**Ne:** `siamdt_swin_tiny_sgd.py` ve `siamdt_swin_tiny_adamw.py` neredeyse aynı (sadece optimizer,
`reg_class_agnostic`, loss tipi farklı) ama tamamen ayrı dosyalar olarak duruyor — mmdet'in
`_base_` inheritance mekanizması burada kullanılmamış.
- **Artı:** Ortak kısımlarda yapılan bir değişikliğin (ör. backbone parametresi) iki dosyada da
  senkron kalmasını garanti eder; şu an olduğu gibi elle iki dosyayı da güncellemeyi unutma riskini
  ortadan kaldırır (bkz. önceki analiz, Bölüm 5.9 — "iki config arasında sessiz farklar").
- **Eksi:** mmdet'in `_base_`/`_delete_` mekanizmasına aşina olmayan biri için başlangıçta okumayı
  zorlaştırabilir (dosyalar arası atlama gerekir); büyük bir kazanç değil, orta vadeli bakım kolaylığı.
- **Öncelik:** Orta.

### D3. Logging ve deney takibi (experiment tracking)
**Ne:** Şu an sadece mmcv'nin dosya tabanlı logger'ı var. TensorBoard/W&B entegrasyonu (mmcv zaten
`TensorboardLoggerHook` destekliyor, sadece config'e eklenmemiş) ve model checkpoint'lerinin/
sonuçlarının (`results/`, `reports/`) versiyon bilgisiyle etiketlenmesi.
- **Artı:** A1-A6 gibi mimari denemelerin karşılaştırılması (hangi değişiklik neyi iyileştirdi)
  çok daha kolay hale gelir; ekip halinde çalışırken tekrarlanabilirlik artar.
- **Eksi:** Ek bağımlılık (wandb/tensorboard) ve disk/ağ kullanımı; hassas/özel veri setleriyle
  çalışırken bulut tabanlı araçlar (W&B) için gizlilik değerlendirmesi gerekir.
- **Öncelik:** Orta.

### D4. `Registry`/`libs/config` ile mmdet `DATASETS`/`DETECTORS` registry'lerinin iç içeliğini belgelemek
**Ne:** Kod tabanında **iki ayrı registry sistemi** paralel çalışıyor: `libs/config/registry.py`'daki
özel `Registry` (dataset/evaluator/transform sınıfları için, `@registry.register_module`) ve mmdet'in
kendi `DATASETS`/`DETECTORS` registry'si (`@DATASETS.register_module()`, `datasets/wrappers.py:52`).
Bu ayrımın neden var olduğunu (ör. bir tanesi benchmark/offline değerlendirme için, diğeri mmdet
eğitim döngüsüne entegrasyon için) bir `ARCHITECTURE.md`'de açıklamak.
- **Artı:** Yeni bir dataset/tracker eklerken "hangi registry'ye kaydetmeliyim?" sorusuna anında
  cevap verir; şu an bu sadece kodu okuyarak çıkarılabiliyor.
- **Eksi:** Salt dokümantasyon işi — kod davranışını değiştirmiyor, doğrudan ölçülebilir bir kazanç
  sağlamıyor ama onboarding süresini kısaltıyor.
- **Öncelik:** Düşük.

---

## Öncelik Özeti

| Öncelik | Maddeler |
|---|---|
| **Yüksek** (düşük efor / yüksek fayda veya bilinen somut sorunu çözüyor) | A2, A5, B1, B2, C1, D1 |
| **Orta** | A1, A3, A4, B3, B6, C2, C3, D2, D3 |
| **Düşük** | B4, B5, C4, D4 |

**Önerilen başlangıç sırası:** Önce B1/B2 (ortam sürtünmesini gidermek, ~1 günlük iş), ardından D1
(en azından similarity modülleri için temel testler), sonra A5 (motion prior — ucuz ve doğrudan
benchmark skorunu etkileyebilir) ve C1 (kayıp-hedef yönetimi). A2/A3 gibi daha büyük mimari
değişiklikler, D1'deki test altyapısı oturduktan sonra çok daha güvenli şekilde denenebilir.

---

*Bu doküman `SIAMDT_KOD_ANALIZI.md` ile birlikte okunmalıdır; her öneri, o dosyadaki ilgili
dosya/satır referanslarına dayanır.*
