import numpy as np
from unittest import TestCase
from cmaes import SepCMA


class TestSepCMA(TestCase):
    def test_restart_from_initial_mean(self):
        initial_mean = np.array([2.0, -3.0])
        optimizer = SepCMA(mean=initial_mean, sigma=1.3, seed=1)
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            solutions.append((x, float(np.sum(x**2))))
        optimizer.tell(solutions)

        self.assertFalse(np.array_equal(optimizer.mean, [2.0, -3.0]))
        np.testing.assert_array_equal(initial_mean, [2.0, -3.0])
        restarted = SepCMA(mean=initial_mean, sigma=1.3, seed=1)
        fresh = SepCMA(mean=np.array([2.0, -3.0]), sigma=1.3, seed=1)
        np.testing.assert_array_equal(restarted.ask(), fresh.ask())

    def test_optimizers_with_shared_initial_mean_are_independent(self):
        initial_mean = np.array([2.0, -3.0])
        optimizer = SepCMA(mean=initial_mean, sigma=1.3, seed=1)
        other = SepCMA(mean=initial_mean, sigma=1.3, seed=2)
        control = SepCMA(mean=initial_mean.copy(), sigma=1.3, seed=2)
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            solutions.append((x, float(np.sum(x**2))))
        optimizer.tell(solutions)

        np.testing.assert_array_equal(other.ask(), control.ask())
        self.assertEqual(other.generation, 0)

    def test_read_only_initial_mean(self):
        initial_mean = np.array([2.0, -3.0])
        initial_mean.setflags(write=False)
        optimizer = SepCMA(mean=initial_mean, sigma=1.3, seed=1)
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            solutions.append((x, float(np.sum(x**2))))
        optimizer.tell(solutions)

        self.assertEqual(optimizer.generation, 1)
        self.assertTrue(np.all(np.isfinite(optimizer.ask())))
        np.testing.assert_array_equal(initial_mean, [2.0, -3.0])
