import os
import sys
import unittest

# trackers/__init__.py eager-importar torch (siamdt_tracking.py uzerinden),
# bu yuzden `from trackers.auto_detect import ...` torch kurulu olmayan
# ortamlarda calismaz. auto_detect.py hicbir sibling dosyaya bagimli
# olmadigindan (tasarim geregi), paketi bypass edip dosyayi dogrudan
# bagimsiz bir modul olarak import ediyoruz.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackers'))
from auto_detect import Candidate, _select_candidates  # noqa: E402


class TestSelectCandidates(unittest.TestCase):
    def test_filters_below_threshold(self):
        det_bboxes = [
            [0, 0, 10, 10, 0.9],
            [5, 5, 15, 15, 0.1],
        ]
        result = _select_candidates(det_bboxes, score_thr=0.3, max_candidates=5)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].score, 0.9)

    def test_sorts_descending_by_score(self):
        det_bboxes = [
            [0, 0, 10, 10, 0.4],
            [1, 1, 11, 11, 0.8],
            [2, 2, 12, 12, 0.6],
        ]
        result = _select_candidates(det_bboxes, score_thr=0.0, max_candidates=5)
        self.assertEqual([c.score for c in result], [0.8, 0.6, 0.4])

    def test_caps_at_max_candidates(self):
        det_bboxes = [[i, i, i + 1, i + 1, 1.0 - i * 0.01] for i in range(10)]
        result = _select_candidates(det_bboxes, score_thr=0.0, max_candidates=3)
        self.assertEqual(len(result), 3)

    def test_accepts_tensor_like_objects_via_tolist(self):
        class FakeTensor:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

        det_bboxes = FakeTensor([[0, 0, 10, 10, 0.7]])
        result = _select_candidates(det_bboxes, score_thr=0.3, max_candidates=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, [0, 0, 10, 10])

    def test_empty_input_returns_empty_list(self):
        result = _select_candidates([], score_thr=0.3, max_candidates=5)
        self.assertEqual(result, [])

    def test_candidate_is_namedtuple_with_bbox_and_score(self):
        c = Candidate(bbox=[1, 2, 3, 4], score=0.5)
        self.assertEqual(c.bbox, [1, 2, 3, 4])
        self.assertEqual(c.score, 0.5)


from unittest.mock import patch

from auto_detect import AutoInitTracker


class FakeTracker:
    def __init__(self, update_script):
        self.update_script = list(update_script)
        self.init_calls = []

    def init(self, img, bbox):
        self.init_calls.append(bbox)

    def update(self, img):
        return self.update_script.pop(0)


class FakeModel:
    def __init__(self, last_score=0.0):
        self._last_score = last_score


class TestAutoInitTrackerRunLoop(unittest.TestCase):
    @patch('auto_detect.free_detect')
    def test_calls_init_when_target_found(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[1, 2, 3, 4], score=0.9)]
        tracker = FakeTracker(update_script=[])
        model = FakeModel()
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None))

        bboxes = wrapper._run_loop(['frame0'])

        self.assertEqual(tracker.init_calls, [[1, 2, 3, 4]])
        self.assertEqual(bboxes, [[1, 2, 3, 4]])

    @patch('auto_detect.free_detect')
    def test_stays_in_need_detect_when_no_candidates(self, mock_detect):
        mock_detect.return_value = []
        tracker = FakeTracker(update_script=[])
        model = FakeModel()
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None))

        bboxes = wrapper._run_loop(['frame0', 'frame1'])

        self.assertEqual(tracker.init_calls, [])
        self.assertEqual(bboxes, [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    @patch('auto_detect.free_detect')
    def test_switches_to_need_detect_after_low_score_then_redetects(self, mock_detect):
        mock_detect.side_effect = [
            [Candidate(bbox=[0, 0, 1, 1], score=0.9)],
            [Candidate(bbox=[9, 9, 10, 10], score=0.9)],
        ]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)  # default lost_thr=0.8 altinda
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            patience=1)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        self.assertEqual(tracker.init_calls, [[0, 0, 1, 1], [9, 9, 10, 10]])

    @patch('auto_detect.free_detect')
    def test_patience_delays_switch_to_need_detect(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[0, 0, 1, 1], score=0.9)]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),
            ([0, 0, 1, 1], False),
        ])
        model = FakeModel(last_score=0.1)
        wrapper = AutoInitTracker(
            tracker, model, transforms=None, preprocess_fn=lambda img: (None, None),
            patience=2)

        wrapper._run_loop(['frame0', 'frame1', 'frame2'])

        # patience=2: iki dusuk-skor karesi gerekiyor, bu yuzden 3 karelik
        # sekansta sadece frame0'daki ilk detect init() cagirir.
        self.assertEqual(len(tracker.init_calls), 1)

    @patch('auto_detect.free_detect')
    def test_high_score_resets_streak(self, mock_detect):
        mock_detect.return_value = [Candidate(bbox=[0, 0, 1, 1], score=0.9)]
        tracker = FakeTracker(update_script=[
            ([0, 0, 1, 1], False),  # dusuk skor -> streak=1
            ([0, 0, 1, 1], False),  # yuksek skor -> streak=0
            ([0, 0, 1, 1], False),  # tekrar dusuk skor -> streak=1 (patience=2'yi tetiklemez)
        ])
        scores = iter([0.1, 0.9, 0.1])

        class ScriptedModel:
            @property
            def _last_score(self):
                return next(scores)

        wrapper = AutoInitTracker(
            tracker, ScriptedModel(), transforms=None,
            preprocess_fn=lambda img: (None, None), patience=2)

        wrapper._run_loop(['frame0', 'frame1', 'frame2', 'frame3'])

        self.assertEqual(len(tracker.init_calls), 1)


if __name__ == '__main__':
    unittest.main()
