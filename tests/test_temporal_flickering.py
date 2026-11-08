import unittest
import numpy as np
from metrics.temporal_flickering import score_frames, calculate_mae


class FlickeringTests(unittest.TestCase):
    def test_identical_frames(self):
        f = np.full((8, 8, 3), 127, np.uint8)
        self.assertEqual(score_frames([f, f.copy(), f.copy()]), 1.0)

    def test_brightness_jump_unsigned_safe(self):
        black = np.zeros((8, 8, 3), np.uint8)
        white = np.full_like(black, 255)
        self.assertEqual(score_frames([white, black, white]), 0.0)
        self.assertEqual(calculate_mae(white, black), 255.0)

    def test_regular_translation_is_penalized(self):
        # A white 2x2 square moves one pixel horizontally per frame.
        frames = []
        for x in [1, 2, 3]:
            f = np.zeros((8, 8, 3), np.uint8)
            f[2:4, x:x+2] = 255
            frames.append(f)
        # Each transition changes four of the 64 pixels across all channels.
        self.assertAlmostEqual(score_frames(frames), 1 - 4/64)

    def test_color_channels_not_grayscale(self):
        a = np.zeros((4, 4, 3), np.uint8)
        b = a.copy()
        b[:, :, 0] = 255
        self.assertAlmostEqual(score_frames([a, b]), 2/3)

    def test_video_mean_over_transitions(self):
        frames = [np.full((2, 2, 3), v, np.uint8) for v in [0, 0, 255]]
        self.assertEqual(score_frames(frames), 0.5)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            score_frames([np.zeros((2, 2, 3), np.uint8)])
        with self.assertRaises(ValueError):
            calculate_mae(np.zeros((2, 2, 3)), np.zeros((3, 3, 3)))


if __name__ == '__main__':
    unittest.main()
