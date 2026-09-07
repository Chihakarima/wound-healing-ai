import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from evaluate_retrieval import rank_of_expected


def test_rank_of_expected_finds_first_match():
    returned = ["Autre article", "Article attendu", "Encore un autre"]
    assert rank_of_expected(returned, ["Article attendu"]) == 2


def test_rank_of_expected_returns_none_when_absent():
    returned = ["Autre article", "Encore un autre"]
    assert rank_of_expected(returned, ["Article attendu"]) is None


def test_rank_of_expected_is_case_insensitive():
    returned = ["ARTICLE ATTENDU"]
    assert rank_of_expected(returned, ["article attendu"]) == 1


def test_rank_of_expected_accepts_multiple_acceptable_titles():
    returned = ["Autre article", "Un des titres acceptables"]
    assert rank_of_expected(returned, ["Titre A", "un des titres acceptables"]) == 2
