import unittest

from src.answer_parser import parse_answer


ABCD = ["A", "B", "C", "D"]

# raw response -> (parsed prediction, parse_status)
CASES = {
    "A": ("A", "exact"),
    " B\n": ("B", "exact"),
    "A.": ("A", "normalized"),
    "(A)": ("A", "normalized"),
    "[D]": ("D", "normalized"),
    "**C**": ("C", "normalized"),
    "Answer: A": ("A", "normalized"),
    "Answer: D": ("D", "normalized"),
    "The answer is (B).": ("B", "normalized"),
    "The correct answer is C": ("C", "normalized"),
    "Option D": ("D", "normalized"),
    "Based on the video, the answer is B.": ("B", "normalized"),
    "A. One.": ("A", "normalized"),
    "B. They are father and son.": ("B", "normalized"),
    "C. A man in a black suit.": ("C", "normalized"),
    "A. A woman with short hair.": ("A", "normalized"),
    "The final answer is $\\boxed{B}$.": ("B", "normalized"),
    "Options are A. One, B. Three.\n\nThe final answer is $\\boxed{B}$.": ("B", "normalized"),
    "The answer is A.\n\n$\\boxed{\\text{C}}$": ("C", "normalized"),
    "$\\boxed{(D)}$": ("D", "normalized"),
    "$\\boxed{E}$": (None, "invalid"),
    "$\\boxed{A}$ or $\\boxed{B}$": (None, "ambiguous"),
    "A or B": (None, "ambiguous"),
    "A and C": (None, "ambiguous"),
    "A, B": (None, "ambiguous"),
    "A. B.": (None, "ambiguous"),
    "(A) or (C)": (None, "ambiguous"),
    "The answer is A, not B... actually the answer is B": (None, "ambiguous"),
    "A man in a black suit.": (None, "invalid"),
    "answer: b": (None, "invalid"),
    "a": (None, "invalid"),
    "E": (None, "invalid"),
    "I cannot determine.": (None, "invalid"),
    "": (None, "invalid"),
}


class AnswerParserTest(unittest.TestCase):
    def test_cases(self):
        for raw, expected in CASES.items():
            with self.subTest(raw=raw):
                parsed = parse_answer(raw, ABCD)
                self.assertEqual((parsed.prediction, parsed.status), expected)

    def test_out_of_range_letter_is_invalid(self):
        parsed = parse_answer("D", ["A", "B", "C"])
        self.assertEqual((parsed.prediction, parsed.status), (None, "invalid"))

    def test_none_is_invalid(self):
        parsed = parse_answer(None, ABCD)
        self.assertEqual((parsed.prediction, parsed.status), (None, "invalid"))


if __name__ == "__main__":
    unittest.main()
