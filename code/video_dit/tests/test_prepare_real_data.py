import json
import hashlib
from pathlib import Path

from prepare_real_data import main
from real_data import load_manifest


def test_cli_assigns_and_validates_a_manifest(tmp_path):
    rows = []
    index = 0
    for subject in ("cat", "dog"):
        for action in ("walking", "running"):
            for scene in ("grass", "snow"):
                index += 1
                rows.append(
                    {
                        "sample_id": f"v{index}",
                        "source_id": f"source-{index}",
                        "clip_path": f"clips/v{index}.mp4",
                        "prompt": f"a {subject} {action} in {scene}",
                        "subject": subject,
                        "action": action,
                        "scene": scene,
                        "source_url": f"https://example.test/v{index}",
                        "license": "CC-BY-4.0",
                        "attribution": "Example",
                        "sha256": f"{index:064x}",
                        "perceptual_hash": f"{index * 997:016x}",
                    }
                )
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "split.jsonl"
    report = tmp_path / "report.json"

    result = main(
        [
            "split",
            str(source),
            str(output),
            "--c1",
            "cat,running,snow",
            "--iid-fraction",
            "0",
            "--validation-fraction",
            "0",
            "--min-c1-source-groups",
            "1",
            "--report",
            str(report),
        ]
    )

    assert result == 0
    assert json.loads(report.read_text())["valid"] is True
    assert {record.split for record in load_manifest(output)} == {"train", "c1"}


def test_cli_reports_independent_coverage_without_assigning_splits(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "sample_id": "cat", "source_id": "source-cat", "clip_path": "clips/cat.mp4",
        "prompt": "a cat running in snow", "subject": "cat", "action": "running",
        "scene": "snow", "source_url": "https://example.test/cat", "license": "CC0",
        "attribution": "Author", "sha256": "a" * 64, "perceptual_hash": "0" * 16,
    }) + "\n")
    report = tmp_path / "coverage.json"

    assert main(["coverage", str(source), "--target-per-triple", "50",
                 "--report", str(report)]) == 0

    saved = json.loads(report.read_text())
    assert saved["triple_source_groups"] == {"cat|running|snow": 1}
    assert saved["triple_deficits"] == {"cat|running|snow": 49}
    assert saved["c1_train_support"]["cat|running|snow"][
        "train_pair_source_groups"
    ] == {"cat|running": 0, "cat|snow": 0, "running|snow": 0}


def test_coverage_cli_persists_preregistered_train_pair_threshold(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = []
    for sample_id, subject, action, scene in (
        ("held", "cat", "playing", "indoors"),
        ("cat-play", "cat", "playing", "outdoors"),
        ("play-in", "dog", "playing", "indoors"),
        ("cat-in", "cat", "walking", "indoors"),
    ):
        rows.append({
            "sample_id": sample_id, "source_id": sample_id,
            "clip_path": f"clips/{sample_id}.mp4",
            "prompt": f"a {subject} {action} in {scene}", "subject": subject,
            "action": action, "scene": scene,
            "source_url": f"https://example.test/{sample_id}", "license": "CC0",
            "attribution": "Author", "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
            "perceptual_hash": hashlib.sha256((sample_id + "phash").encode()).hexdigest()[:16],
        })
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "coverage.json"

    assert main([
        "coverage", str(source), "--target-per-triple", "1",
        "--min-train-pair-source-groups", "2", "--report", str(report),
    ]) == 0

    saved = json.loads(report.read_text())
    assert saved["min_train_pair_source_groups"] == 2
    assert saved["c1_train_support"]["cat|playing|indoors"][
        "train_pair_deficits"
    ] == {"cat|indoors": 1, "cat|playing": 1, "playing|indoors": 1}
