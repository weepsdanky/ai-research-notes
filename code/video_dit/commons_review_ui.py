"""Local Gradio UI for human review of Commons training candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commons_review import (
    cache_review_preview, fetch_review_media, load_rows, save_review, summarize_reviews,
)
from web_ui import ensure_loopback_proxy_bypass


def build_ui(candidates_path: Path, preview_cache: Path | None = None):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is required for the review UI") from error

    rows = load_rows(candidates_path)
    if not rows:
        raise ValueError("candidate manifest is empty")
    titles = [str(row["title"]) for row in rows]
    cache_dir = preview_cache or candidates_path.parent / "review_previews"

    def load(title: str):
        current = next(row for row in load_rows(candidates_path) if row["title"] == title)
        if current.get("review_preview_url"):
            media = {
                "preview_url": current["review_preview_url"],
                "duration": current.get("duration_seconds", 0),
            }
        else:
            media = fetch_review_media(title)
        summary = summarize_reviews(load_rows(candidates_path))
        preview = cache_review_preview(title, str(media["preview_url"]), cache_dir)
        return (
            str(preview), current.get("description_url", ""),
            current.get("license", ""), current.get("attribution", ""),
            current.get("description", ""), current.get("review_status", "pending"),
            current.get("subject", ""), current.get("action", ""),
            current.get("scene", ""), current.get("prompt", ""),
            float(current.get("start_seconds", 0)), current.get("review_notes", ""),
            json.dumps({**summary, "duration": media["duration"]}, indent=2),
        )

    def save(title, status, subject, action, scene, prompt, start, notes):
        save_review(
            candidates_path, title, status=status, subject=subject, action=action,
            scene=scene, prompt=prompt, start_seconds=float(start), notes=notes,
        )
        return json.dumps(summarize_reviews(load_rows(candidates_path)), indent=2)

    with gr.Blocks(title="Commons real-video review") as demo:
        gr.Markdown(
            "# Commons real-video human review\n"
            "Preview is an official low-resolution derivative. Approval requires watching "
            "the video and entering the visible subject, action, scene and a truthful prompt."
        )
        title = gr.Dropdown(choices=titles, value=titles[0], label="Candidate")
        video = gr.Video(label="Official 240p review derivative")
        page = gr.Textbox(label="Commons description page", interactive=False)
        with gr.Row():
            license_name = gr.Textbox(label="License", interactive=False)
            attribution = gr.Textbox(label="Attribution", interactive=False)
        description = gr.Textbox(label="Commons description", interactive=False)
        with gr.Row():
            status = gr.Radio(
                choices=["pending", "approved", "rejected"], value="pending",
                label="Human decision",
            )
            subject = gr.Textbox(label="Subject")
            action = gr.Textbox(label="Action")
            scene = gr.Textbox(label="Scene")
        prompt = gr.Textbox(label="Truthful natural-language prompt")
        with gr.Row():
            start = gr.Number(value=0, minimum=0, label="Best 2-second segment start")
            notes = gr.Textbox(label="Review notes")
        save_button = gr.Button("Save review atomically", variant="primary")
        report = gr.Code(label="Review coverage", language="json")
        load_outputs = [
            video, page, license_name, attribution, description, status, subject,
            action, scene, prompt, start, notes, report,
        ]
        title.change(load, inputs=[title], outputs=load_outputs)
        demo.load(load, inputs=[title], outputs=load_outputs)
        save_button.click(
            save,
            inputs=[title, status, subject, action, scene, prompt, start, notes],
            outputs=[report],
            concurrency_limit=1,
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_loopback_proxy_bypass(args.server_name)
    build_ui(args.candidates).launch(
        server_name=args.server_name, server_port=args.server_port, share=False,
    )


if __name__ == "__main__":
    main()
