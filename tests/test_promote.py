from promote import should_promote

def test_should_promote_when_new_is_better():
    assert should_promote(new_f1=0.60, champion_f1=0.55) is True

def test_should_not_promote_when_new_is_worse():
    assert should_promote(new_f1=0.50, champion_f1=0.55) is False

def test_should_not_promote_on_tie():
    assert should_promote(new_f1=0.55, champion_f1=0.55) is False
