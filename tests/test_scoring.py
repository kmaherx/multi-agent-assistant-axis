from src.scoring import extract_final_answer, is_correct


def test_extract_final_answer_basic():
    assert extract_final_answer("Final answer: 42") == 42.0
    assert extract_final_answer("Final Answer: $1,234.5") == 1234.5
    assert extract_final_answer("blah blah\nFinal answer: -7\n") == -7.0


def test_extract_final_answer_fallback_last_number():
    assert extract_final_answer("the answer is 99") == 99.0


def test_extract_final_answer_none():
    assert extract_final_answer("") is None
    assert extract_final_answer("no numbers here") is None


def test_is_correct():
    assert is_correct(42.0, 42)
    assert not is_correct(43.0, 42)
    assert is_correct(0.0, 0)
    assert is_correct(None, 42) is False
