from evals.graders import cohens_kappa, grade_correctness, normalize


def test_normalize_ignores_dollar_signs_and_commas():
    assert normalize("$391,035 Million") == "391035 million"


def test_refusal_grading():
    assert grade_correctness({"must_refuse": True}, {"status": "refused"})["pass"]
    assert not grade_correctness({"must_refuse": True}, {"status": "answered"})["pass"]


def test_kappa_perfect_agreement_is_one():
    assert cohens_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
