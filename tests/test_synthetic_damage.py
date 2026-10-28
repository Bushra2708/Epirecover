"""
tests/test_synthetic_damage.py
------------------------------
Tests for synthetic damage benchmark generator (D0 to D6).
"""

import unittest
import numpy as np
from src.evaluation.synthetic_damage import SyntheticDamageGenerator


class TestSyntheticDamage(unittest.TestCase):

    def setUp(self):
        self.gen = SyntheticDamageGenerator(seed=123)
        self.img = np.full((200, 200, 3), 200, dtype=np.uint8)

    def test_d0_clean_baseline(self):
        """D0 damage must produce zero damaged pixels."""
        out = self.gen.apply_damage(self.img, "D0")
        self.assertEqual(out.damage_level, "D0")
        self.assertEqual(out.damage_ratio, 0.0)
        self.assertEqual(np.sum(out.damage_mask), 0)

    def test_d1_to_d6_increasing_damage_levels(self):
        """Higher damage levels should introduce verifiable damage masks."""
        for level in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            out = self.gen.apply_damage(self.img, level)
            self.assertEqual(out.damage_level, level)
            self.assertGreater(out.damage_ratio, 0.0)
            self.assertGreater(len(out.applied_operations), 0)
            self.assertEqual(out.damage_mask.shape, (200, 200))

    def test_invalid_level_raises(self):
        """Unsupported damage level must raise ValueError."""
        with self.assertRaises(ValueError):
            self.gen.apply_damage(self.img, "D7")


if __name__ == "__main__":
    unittest.main()
