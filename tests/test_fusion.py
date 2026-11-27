from app.fusion import reciprocal_rank_fusion, tokenize


def test_tokenize_splits_codes_and_lowercases():
    assert tokenize("Policy POL-305, MFA!") == ["policy", "pol", "305", "mfa"]


def test_item_ranked_well_by_both_lists_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    assert fused[0][0] == "b"  # 2nd + 1st beats 1st + 3rd


def test_item_only_in_one_list_is_kept():
    ids = [item for item, _ in reciprocal_rank_fusion([["a"], ["b"]])]
    assert set(ids) == {"a", "b"}


def test_weights_change_the_order():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 3.0])
    assert fused[0][0] == "b"
