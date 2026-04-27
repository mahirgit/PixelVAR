from pixelvar.data.splits import assert_no_split_leakage, make_id_splits, parse_pokemon_id


def test_pokemon_id_split_is_by_id():
    assert parse_pokemon_id("data/raw/pokemon/back/25_back.png") == "25"

    split_map = make_id_splits(["1", "1", "2", "3", "4", "5"], seed=42)
    records = [
        {"index": 0, "pokemon_id": "1", "split": split_map["1"]},
        {"index": 1, "pokemon_id": "1", "split": split_map["1"]},
        {"index": 2, "pokemon_id": "2", "split": split_map["2"]},
    ]
    assert_no_split_leakage(records)
