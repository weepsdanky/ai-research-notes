"""Local prompt-hidden UI for independent generated-video factor judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from generation_blind_review import DECISION_PROTOCOL, QUEUE_PROTOCOL, UNCLEAR_FACTOR


def _queue(data: bytes) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("blind generation queue must be valid UTF-8 JSON") from error
    if (
        not isinstance(value, dict) or value.get("protocol") != QUEUE_PROTOCOL
        or value.get("prompt_hidden") is not True
    ):
        raise ValueError("blind generation queue has an invalid prompt-hidden contract")
    records = value.get("records")
    vocabulary = value.get("factor_vocabulary")
    if not isinstance(records, list) or not records or not isinstance(vocabulary, dict):
        raise ValueError("blind generation queue requires records and factor vocabulary")
    for field in ("subjects", "actions", "scenes"):
        choices = vocabulary.get(field)
        if not isinstance(choices, list) or not choices or any(
            not isinstance(choice, str) or not choice for choice in choices
        ) or choices.count(UNCLEAR_FACTOR) != 1:
            raise ValueError("blind generation queue factor vocabulary is invalid")
    return value


def review_queue(queue_bytes: bytes, media_dir: Path) -> list[dict[str, str]]:
    """Return only blind IDs and locked local paths; never prompt assignments."""

    queue = _queue(queue_bytes)
    if media_dir.is_symlink() or not media_dir.is_dir():
        raise ValueError("blind review media directory is unavailable")
    rows: list[dict[str, str]] = []
    expected: set[str] = set()
    blind_ids: set[str] = set()
    for record in queue["records"]:  # type: ignore[index]
        if not isinstance(record, dict):
            raise ValueError("blind generation queue records must be objects")
        blind_id, filename = record.get("blind_id"), record.get("filename")
        if (
            not isinstance(blind_id, str) or not blind_id or blind_id in blind_ids
            or filename != f"{blind_id}.mp4"
        ):
            raise ValueError("blind generation queue IDs and filenames are invalid")
        blind_ids.add(blind_id); expected.add(filename)
        path = media_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"blind review media is missing: {blind_id}")
        data = path.read_bytes()
        if (
            record.get("bytes") != len(data)
            or record.get("video_sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(f"blind review media hash/size differs from queue: {blind_id}")
        rows.append({"blind_id": blind_id, "video_path": str(path.resolve())})
    entries = list(media_dir.iterdir())
    actual = {path.name for path in entries}
    if actual != expected or any(not path.is_file() for path in entries):
        raise ValueError("blind review media directory does not exactly match the queue")
    return rows


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remaining_review_ids(
    queue_bytes: bytes, decisions_path: Path, *, reviewer_id: str,
) -> list[str]:
    queue = _queue(queue_bytes)
    queue_ids = [str(row["blind_id"]) for row in queue["records"]]  # type: ignore[index]
    if not decisions_path.exists():
        return queue_ids
    try:
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("existing reviewer decisions are invalid JSON") from error
    if (
        not isinstance(decisions, dict) or decisions.get("protocol") != DECISION_PROTOCOL
        or decisions.get("queue_sha256") != hashlib.sha256(queue_bytes).hexdigest()
        or decisions.get("reviewer_id") != reviewer_id
        or not isinstance(decisions.get("records"), list)
    ):
        raise ValueError("existing reviewer decisions do not bind this queue/reviewer")
    completed = [
        row.get("blind_id") for row in decisions["records"] if isinstance(row, dict)
    ]
    if (
        len(completed) != len(decisions["records"]) or len(set(completed)) != len(completed)
        or any(blind_id not in queue_ids for blind_id in completed)
    ):
        raise ValueError("existing reviewer decisions contain duplicate or unknown blind IDs")
    return [blind_id for blind_id in queue_ids if blind_id not in completed]


def save_generation_decision(
    queue_bytes: bytes,
    media_dir: Path,
    decisions_path: Path,
    *,
    reviewer_id: str,
    blind_id: str,
    subject: str,
    action: str,
    scene: str,
    notes: str,
) -> list[str]:
    """Atomically save one queue-bound decision and return remaining blind IDs."""

    queue = _queue(queue_bytes)
    review_queue(queue_bytes, media_dir)
    reviewer_id = reviewer_id.strip()
    notes = notes.strip()
    if not reviewer_id or not notes:
        raise ValueError("reviewer_id and review notes are required")
    queue_ids = [str(row["blind_id"]) for row in queue["records"]]  # type: ignore[index]
    if blind_id not in queue_ids:
        raise ValueError("blind_id is absent from the review queue")
    vocabulary = queue["factor_vocabulary"]  # type: ignore[index]
    if (
        subject not in vocabulary["subjects"] or action not in vocabulary["actions"]
        or scene not in vocabulary["scenes"]
    ):
        raise ValueError("factor judgment is outside the frozen queue vocabulary")
    queue_sha = hashlib.sha256(queue_bytes).hexdigest()
    if decisions_path.exists():
        try:
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("existing reviewer decisions are invalid JSON") from error
        if (
            not isinstance(decisions, dict) or decisions.get("protocol") != DECISION_PROTOCOL
            or decisions.get("queue_sha256") != queue_sha
            or decisions.get("reviewer_id") != reviewer_id
            or not isinstance(decisions.get("records"), list)
        ):
            raise ValueError("existing reviewer decisions do not bind this queue/reviewer")
    else:
        decisions = {
            "protocol": DECISION_PROTOCOL, "queue_sha256": queue_sha,
            "reviewer_id": reviewer_id, "records": [],
        }
    records = {
        str(row.get("blind_id")): row for row in decisions["records"]
        if isinstance(row, dict)
    }
    if len(records) != len(decisions["records"]) or any(key not in queue_ids for key in records):
        raise ValueError("existing reviewer decisions contain duplicate or unknown blind IDs")
    records[blind_id] = {
        "blind_id": blind_id, "predicted_subject": subject,
        "predicted_action": action, "predicted_scene": scene, "notes": notes,
        "observed_video_sha256": next(
            str(row["video_sha256"]) for row in queue["records"]  # type: ignore[index]
            if row["blind_id"] == blind_id
        ),
    }
    decisions["records"] = [records[key] for key in queue_ids if key in records]
    _atomic_json(decisions_path, decisions)
    return [key for key in queue_ids if key not in records]


def build_demo(
    queue_path: Path, media_dir: Path, decisions_path: Path, reviewer_id: str,
):
    import gradio as gr

    queue_bytes = queue_path.read_bytes()
    rows = review_queue(queue_bytes, media_dir)
    paths = {row["blind_id"]: row["video_path"] for row in rows}
    queue = _queue(queue_bytes)
    vocabulary = queue["factor_vocabulary"]
    ids = [row["blind_id"] for row in rows]
    remaining = remaining_review_ids(
        queue_bytes, decisions_path, reviewer_id=reviewer_id,
    )
    initial = remaining[0] if remaining else None

    def load(blind_id):
        if blind_id not in paths:
            return None, "Queue complete."
        return paths[blind_id], f"Reviewing {blind_id}; prompt, split, and seed are sealed."

    def save(blind_id, subject, action, scene, notes):
        remaining = save_generation_decision(
            queue_bytes, media_dir, decisions_path,
            reviewer_id=reviewer_id, blind_id=blind_id,
            subject=subject, action=action, scene=scene, notes=notes,
        )
        if not remaining:
            return None, "Queue complete. Decisions are ready for reveal after another rater finishes."
        return remaining[0], f"Saved {blind_id}."

    with gr.Blocks(title="Blind generated-video factor review") as demo:
        gr.Markdown(
            "# Prompt-hidden generated-video review\n"
            "Watch the full animation. The prompt, split, seed, model ranking, and expected labels are sealed."
        )
        blind_id = gr.Dropdown(ids, value=initial, label="Blind ID")
        video = gr.Video(
            value=paths[initial] if initial else None, label="Generated animation",
        )
        subject = gr.Dropdown(vocabulary["subjects"], label="Visible subject")
        action = gr.Dropdown(vocabulary["actions"], label="Visible temporal action")
        scene = gr.Dropdown(vocabulary["scenes"], label="Visible scene")
        notes = gr.Textbox(label="Evidence / ambiguity notes")
        status = gr.Markdown(
            "Prompt, split, and seed are sealed." if initial else "Queue complete."
        )
        save_button = gr.Button(
            "Save and continue", variant="primary", interactive=bool(initial),
        )
        blind_id.change(load, inputs=blind_id, outputs=[video, status])
        save_button.click(
            save, inputs=[blind_id, subject, action, scene, notes],
            outputs=[blind_id, status], concurrency_limit=1,
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    if args.decisions.resolve() == args.queue.resolve() or (
        args.media_dir.resolve() in args.decisions.resolve().parents
    ):
        raise ValueError("review decisions must not overwrite queue or blind media")
    demo = build_demo(args.queue, args.media_dir, args.decisions, args.reviewer_id)
    demo.launch(server_name="127.0.0.1", server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
