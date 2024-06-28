import unittest
from GUI_production import collect_stock_data, analyze_stock_data, review_analysis

class TestStockAnalyzer(unittest.TestCase):
    def test_collect_stock_data(self):
        result = collect_stock_data("AAPL", "2023-01-01")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

    def test_analyze_stock_data(self):
        analysis = analyze_stock_data("AAPL", "Sample data summary")
        self.assertIsNotNone(analysis)
        self.assertIsInstance(analysis, str)

    def test_review_analysis(self):
        review = review_analysis("AAPL", "Sample analysis output")
        self.assertIsNotNone(review)
        self.assertIsInstance(review, str)

if __name__ == '__main__':
    unittest.main()
