from experiment_data import build_grouped_collection_manifest, sha256_json


def test_grouped_manifest_keeps_counterfactual_pairs_together():
    lengths = [10] * 20
    pair_ids = [pair for pair in range(10) for _ in range(2)]
    manifest = build_grouped_collection_manifest(
        lengths, "dataset-hash", pair_ids, seed=9
    )
    assert manifest["version"] == 3
    assert manifest["sha256"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    episode_split = {
        episode: split
        for split, episodes in manifest["episodes"].items()
        for episode in episodes
    }
    for pair in range(10):
        assert episode_split[2 * pair] == episode_split[2 * pair + 1]
