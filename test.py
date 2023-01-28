import unittest
import helpers


class TestWeightedSum(unittest.TestCase):
    def test_weighted_sum_first_n_elements(self):
        lst = [1, 2, 3, 4, 5]
        b = 0.9
        n = 3
        expected_result = 5.23
        result = helpers.weighted_sum(lst, b, n)
        self.assertEqual(result, expected_result)

    def test_weighted_sum_first_n_elements_02(self):
        lst = [1, 2, 3, 4, 5]
        b = 1
        n = 5
        expected_result = 15
        result = helpers.weighted_sum(lst, b, n)
        self.assertEqual(result, expected_result)

    def test_weighted_sum_with_b_as_zero(self):
        lst = [1, 2, 3, 4, 5]
        b = 0
        n = 4
        expected_result = 1
        result = helpers.weighted_sum(lst, b, n)
        self.assertEqual(result, expected_result)

    def test_weighted_sum_with_n_as_zero(self):
        lst = [1, 2, 3, 4, 5]
        expected_result = 0
        result = helpers.weighted_sum(lst, b=0.9, n=0)
        self.assertEqual(result, expected_result)

    def test_weighted_sum_with_large_n(self):
        lst = [1, 2, 3, 4, 5]
        expected_result = 11.4265
        result = helpers.weighted_sum(lst, b=0.9, n=10)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
