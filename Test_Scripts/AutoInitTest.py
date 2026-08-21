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
from trackers.auto_detect import AutoInitTracker

# =====================================================================
#   AYARLAR -- parametre yerine burada elle degistir
# =====================================================================
FRAME_DIR = '/home/gorsis3/Desktop/staj/SiamDT/gorsel'                      # islenecek kare klasoru
NUM_FRAMES = 23000                              # islenecek kare sayisi (START_FRAME_INDEX'ten itibaren)
START_FRAME_INDEX = 0                         # siralanmis dosya listesinde baslanacak sira (0 = bastan)

CONFIG_FILE = 'configs/siamdt_swin_tiny_sgd.py'
CHECKPOINT_FILE = '/home/gorsis3/Desktop/staj/SiamDT/work_dirs/siamdt_swin_tiny_sgd/epoch_2.pth'

SCORE_THR = 0.06     # free_detect aday-kabul skor esigi (Adim 4 kalibrasyonu)
LOST_THR = 0.8       # "hedef kayboldu" karar esigi (_last_score, 0-2 olcek)
PATIENCE = 1         # NEED_DETECT'e gecmeden once kac dusuk-skorlu kare beklenecek
DETECT_SIZE_MULT_START = 1.3   # kayiptan hemen sonra ilk denemedeki sinir
DETECT_SIZE_MULT_STEP = 0.3    # her basarisiz denemede sinir bu kadar gevser
DETECT_SIZE_MULT_MAX = 3.0     # sinir en fazla bu degere kadar gevseyebilir

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


def list_frame_files(frame_dir, num_frames, start_frame_index=0):
    files = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        files.extend(glob.glob(osp.join(frame_dir, pattern)))
    files = sorted(files, key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(f'{frame_dir} icinde goruntu dosyasi bulunamadi.')
    if start_frame_index >= len(files):
        raise ValueError(
            f'START_FRAME_INDEX={start_frame_index} ama klasorde sadece {len(files)} kare var '
            f'(gecerli araligim: 0-{len(files) - 1}).')
    available = len(files) - start_frame_index
    if available < num_frames:
        print(f'[UYARI] {start_frame_index}. kareden itibaren {num_frames} kare istendi ama '
              f'sadece {available} kare kaldi, {available} ile devam ediliyor.')
        num_frames = available
    return files[start_frame_index:start_frame_index + num_frames]


def main():
    frame_paths = list_frame_files(FRAME_DIR, NUM_FRAMES, START_FRAME_INDEX)
    print(f'{len(frame_paths)} kare islenecek (kaynak: {FRAME_DIR}, '
          f'baslangic index: {START_FRAME_INDEX})')

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

    auto_tracker = AutoInitTracker(
        tracker, tracker.model, transforms,
        lost_thr=LOST_THR, patience=PATIENCE, score_thr=SCORE_THR,
        detect_size_mult_start=DETECT_SIZE_MULT_START,
        detect_size_mult_step=DETECT_SIZE_MULT_STEP,
        detect_size_mult_max=DETECT_SIZE_MULT_MAX)
    print(f'AutoInitTracker hazir (score_thr={SCORE_THR}, lost_thr={LOST_THR}, patience={PATIENCE})')

    images = (ops.read_image(p) for p in frame_paths)

    boxes = []  # sadece kutu konumlari -> test_results/predictions/<run_name>.json
    last_known_bbox = [0.0, 0.0, 0.0, 0.0]

    def on_frame(f, img, bbox, up_flag, mode):
        nonlocal last_known_bbox
        img = img.copy()
        cv2.putText(
            img, f'mode: {mode}', (20, img.shape[0] - 20),
            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

        draw_bbox = None if (mode == 'NO-DETECT' or bbox is None) else bbox
        annotated = ops.show_image(
            img, draw_bbox, f, up_flag, viz, fig=VISDOM_WINDOW, visualize=True)

        frame_stem = osp.splitext(osp.basename(frame_paths[f]))[0]
        ops.save_image(
            osp.join(frames_out_dir, f'{f:06d}_{frame_stem}.jpg'), annotated)

        record_bbox = bbox if bbox is not None else last_known_bbox
        boxes.append([float(v) for v in record_bbox])
        last_known_bbox = record_bbox

        if (f + 1) % 25 == 0 or (f + 1) == len(frame_paths):
            print(f'  [{f + 1}/{len(frame_paths)}] mode={mode} '
                  f'bbox={[round(v, 1) for v in record_bbox]}')


    begin = time.time()
    auto_tracker._run_loop(images, on_frame=on_frame)
    elapsed = time.time() - begin

    pred_file = osp.join(preds_out_dir, f'{run_name}.json')
    with open(pred_file, 'w') as f_out:
        json.dump({'res': boxes}, f_out)

    print(f'\nTamamlandi. {len(frame_paths)} kare, {elapsed:.1f} sn.')
    print(f'  Kareler   -> {frames_out_dir}')
    print(f'  Tahminler -> {pred_file}')


if __name__ == '__main__':
    main()
