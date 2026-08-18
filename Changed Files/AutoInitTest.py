"""
Bir klasordeki (etiketsiz) kareler uzerinde AutoInitTracker'i calistirir:
tracker'i elle/GT kutu vermeden, sadece free_detect() ile kendi kendine
baslatir, asagidaki AYARLAR blogunda belirtilen kadar kareyi isler,
sonuclari Visdom'da -- "main" degil, kendi ayri environment'inda --
CANLI gosterir ve her seyi diske kaydeder.

Bu bir DEGERLENDIRME (evaluator) script'i degildir - elimizde GT/label
olmadigi icin IoU/recall gibi bir metrik HESAPLAMAZ; sadece tracker'in ne
tahmin ettigini gosterir ve kaydeder.

On kosul: Visdom sunucusu ayrı bir terminalde calisiyor olmali:
    python -m visdom.server -port=5123

Kaydedilenler:
    test_results/frames/<run_name>/000000_<orijinal_ad>.jpg   (kutu cizilmis kareler)
    test_results/predictions/<run_name>.json                  ({'res': [[x1,y1,x2,y2], ...]})
"""

import init_paths  # noqa: F401  (mmdet/registry kayitlari icin, importu tetikliyor)

import glob
import json
import os
import os.path as osp
import re
import time

import cv2
from visdom import Visdom

import libs.data as data
import libs.ops as ops
from trackers import SiamDTTracker
from trackers.auto_detect import free_detect, default_preprocess

# =====================================================================
#   AYARLAR -- parametre yerine burada elle degistir
# =====================================================================
FRAME_DIR = '/home/gorsis3/Desktop/staj/2_Etiketlenecek_Veri/gorsel'                      # islenecek kare klasoru
NUM_FRAMES = 2000                                     # islenecek kare sayisi (bastan itibaren)

CONFIG_FILE = 'configs/siamdt_swin_tiny_sgd.py'
CHECKPOINT_FILE = '/home/gorsis3/Desktop/staj/SiamDT/work_dirs/siamdt_swin_tiny_sgd/epoch_2.pth'

SCORE_THR = 0.06     # free_detect aday-kabul skor esigi (Adim 4 kalibrasyonu)
LOST_THR = 0.8       # "hedef kayboldu" karar esigi (_last_score, 0-2 olcek)
PATIENCE = 1         # NEED_DETECT'e gecmeden once kac dusuk-skorlu kare beklenecek

OUTPUT_DIR = 'test_results'
RUN_NAME = None       # None -> frame klasoru adi + zaman damgasi

VISDOM_PORT = 5123
VISDOM_ENV = 'siamdt_auto_init_test'   # 'main' DEGIL -- ayri bir Visdom environment'i
VISDOM_WINDOW = 'auto_init_test'       # pencere adi (sabit -> tek, surekli guncellenen pencere)
# =====================================================================


def _natural_sort_key(path):
    """Dosya adindaki sayiyi sayi olarak siralar (ör. '2.jpg' < '10.jpg'),
    saf string siralamasinin '10.jpg' < '2.jpg' gibi yanlis sonuc vermesini
    onler. Dosya adinda sayi yoksa duz string siralamasina duser."""
    name = osp.basename(path)
    digits = re.findall(r'\d+', name)
    return (int(digits[-1]), name) if digits else (float('inf'), name)


def list_frame_files(frame_dir, num_frames):
    files = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        files.extend(glob.glob(osp.join(frame_dir, pattern)))
    files = sorted(files, key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(f'{frame_dir} icinde goruntu dosyasi bulunamadi.')
    if len(files) < num_frames:
        print(f'[UYARI] {num_frames} kare istendi ama klasorde {len(files)} tane var, '
              f'{len(files)} ile devam ediliyor.')
        num_frames = len(files)
    return files[:num_frames]


def main():
    frame_paths = list_frame_files(FRAME_DIR, NUM_FRAMES)
    print(f'{len(frame_paths)} kare islenecek (kaynak: {FRAME_DIR})')

    run_name = RUN_NAME or (
        osp.basename(osp.normpath(FRAME_DIR)) + '_' + time.strftime('%Y%m%d_%H%M%S'))

    frames_out_dir = osp.join(OUTPUT_DIR, 'frames', run_name)
    preds_out_dir = osp.join(OUTPUT_DIR, 'predictions')
    os.makedirs(frames_out_dir, exist_ok=True)
    os.makedirs(preds_out_dir, exist_ok=True)

    # --- Visdom baglantisi: "main" degil, kendi ayri environment'imiz ---
    viz = Visdom(port=VISDOM_PORT, env=VISDOM_ENV)
    assert viz.check_connection(), (
        f'Visdom sunucusuna baglanilamadi. Once calistir: '
        f'python -m visdom.server -port={VISDOM_PORT}')
    print(f'Visdom baglandi -> env="{VISDOM_ENV}", pencere="{VISDOM_WINDOW}"')

    # --- Tracker kurulumu ---
    # visualize=False: Tracker.__init__'in kendi "main" environment'inda
    # ikinci bir Visdom baglantisi acmasini istemiyoruz -- gorsellestirmeyi
    # yukarida kurdugumuz kendi Visdom baglantimizla biz yapiyoruz.
    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(
        CONFIG_FILE, CHECKPOINT_FILE, transforms,
        name_suffix=run_name + '_auto', visualize=False)

    print(f'Tracker hazir (score_thr={SCORE_THR}, lost_thr={LOST_THR}, patience={PATIENCE})')

    images = (ops.read_image(p) for p in frame_paths)

    boxes = []  # sadece kutu konumlari -> test_results/predictions/<run_name>.json

    def on_frame(f, img, bbox, up_flag, mode):
        # Hangi mekanizmanin bu kareyi urettigini kucuk bir metinle karenin
        # uzerine yaz (DETECT / NO-DETECT / TRACK / TRACK-FALLBACK). Bunu
        # show_image()'DAN ONCE cizmek gerekiyor: show_image kendi
        # RGB->BGR->RGB donusumunu img uzerinde yapip donduruyor, once
        # cizersek hem canli Visdom akisina hem diske kaydedilen kareye
        # ayni sekilde yansir.
        cv2.putText(
            img, f'mode: {mode}', (20, img.shape[0] - 20),
            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

        # Kendi Visdom ortamimiza (env=VISDOM_ENV) canli yayinla; fig sabit
        # tutuldugu icin tek, surekli guncellenen pencere olarak gorunur.
        annotated = ops.show_image(
            img, bbox, f, up_flag, viz, fig=VISDOM_WINDOW, visualize=True)

        # Kutusu cizilmis kareyi diske kaydet.
        frame_stem = osp.splitext(osp.basename(frame_paths[f]))[0]
        ops.save_image(
            osp.join(frames_out_dir, f'{f:06d}_{frame_stem}.jpg'), annotated)

        # Sadece kutunun konumunu biriktir.
        boxes.append([float(v) for v in bbox])

        if (f + 1) % 25 == 0 or (f + 1) == len(frame_paths):
            print(f'  [{f + 1}/{len(frame_paths)}] mode={mode} '
                  f'bbox={[round(v, 1) for v in bbox]}')

    # --- AutoInitTracker._run_loop ile AYNI durum makinesi, elle yurutuluyor
    # -- boylece hangi mekanizmanin (DETECT / TRACK / TRACK-FALLBACK)
    # kullanildigini da uretebiliyoruz. AutoInitTracker'in kendi (protected)
    # ic metodlarina bagimli olmamak icin free_detect/default_preprocess
    # dogrudan kullaniliyor. NOT: 'TRACK-FALLBACK' etiketi, siamdt_tracking.py
    # icindeki update()'e eklenen self._last_selection_mode kancasina
    # bagimli -- o kanca yoksa hepsi 'TRACK' gorunur.
    state = 'NEED_DETECT'
    low_score_streak = 0
    begin = time.time()
    for f, img in enumerate(images):
        if state == 'NEED_DETECT':
            img_tensor, img_metas = default_preprocess(
                transforms, img, next(tracker.model.parameters()).device)
            candidates = free_detect(
                tracker.model, img_tensor, img_metas,
                score_thr=SCORE_THR, max_candidates=5)
            if candidates:
                tracker.init(img, candidates[0].bbox)
                state = 'TRACKING'
                low_score_streak = 0
                bbox = candidates[0].bbox
                up_flag = True
                mode = 'DETECT'
            else:
                bbox = boxes[-1] if boxes else [0.0, 0.0, 0.0, 0.0]
                up_flag = False
                mode = 'NO-DETECT'
        else:
            bbox, up_flag = tracker.update(img)
            bbox = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
            selection_mode = getattr(tracker.model, '_last_selection_mode', None)
            mode = 'TRACK-FALLBACK' if selection_mode == 'fallback' else 'TRACK'
            score = getattr(tracker.model, '_last_score', 0.0)
            if score < LOST_THR:
                low_score_streak += 1
                if low_score_streak >= PATIENCE:
                    state = 'NEED_DETECT'
                    low_score_streak = 0
            else:
                low_score_streak = 0
        on_frame(f, img, bbox, up_flag, mode)
    elapsed = time.time() - begin

    # --- Tahminleri (kutu konumlarini) kaydet ---
    # {'res': [...]} -- libs/data/evaluators/uavtir_eval.py'nin kendi sonuc
    # dosyalarinda kullandigi formatla ayni.
    pred_file = osp.join(preds_out_dir, f'{run_name}.json')
    with open(pred_file, 'w') as f_out:
        json.dump({'res': boxes}, f_out)

    print(f'\nTamamlandi. {len(frame_paths)} kare, {elapsed:.1f} sn.')
    print(f'  Kareler   -> {frames_out_dir}')
    print(f'  Tahminler -> {pred_file}')


if __name__ == '__main__':
    main()
