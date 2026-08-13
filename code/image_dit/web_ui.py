"""A dependency-light local Web UI for class-conditional DiT inference."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import io
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

from inference import DiTInference


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "demo_artifacts/image_dit_inference.pt"
)


HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image DiT Studio</title><style>
:root{color-scheme:dark;background:#090b10;color:#eef2ff;font-family:Inter,system-ui,sans-serif}
body{max-width:1100px;margin:auto;padding:36px 24px}.card{background:#131722;border:1px solid #293044;border-radius:18px;padding:22px;margin:18px 0}
h1{font-size:36px;margin:0}.sub{color:#9aa6c2}.meta{color:#c6d0ea;font-family:ui-monospace,monospace;font-size:13px}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:15px}
label{display:grid;gap:8px;color:#b8c2dc}select,input,button{padding:12px;border-radius:10px;border:1px solid #39435d;background:#0d111a;color:white}
button{background:#6657ff;border:0;font-weight:700;cursor:pointer}.results{display:grid;grid-template-columns:1fr 2fr;gap:22px;align-items:start}
img{width:100%;image-rendering:pixelated;border-radius:12px;background:#080a0e}.timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:10px}
.frame small{color:#8490ac}.status{min-height:22px;color:#9ee6c1}@media(max-width:700px){.results{grid-template-columns:1fr}}
</style></head><body><h1>Image DiT Studio</h1><p class="sub">生成一张图片；右侧轨迹仅用于观察扩散算法，不是视频内容。</p>
<p class="meta">__METADATA__</p><section class="card controls"><label>类别<select id="class">__OPTIONS__</select></label>
<label>CFG scale<input id="cfg" type="number" min="1" max="10" step="0.25" value="5"></label>
<label>Seed<input id="seed" type="number" value="42"></label>
<label>轨迹帧数<input id="frames" type="number" min="2" max="30" value="12"></label>
<label>Sampler<select id="sampler"><option value="ddpm">DDPM（完整质量）</option><option value="ddim">DDIM（快速质量对照）</option><option value="paper_ddpm">Paper respaced DDPM</option></select></label>
<label>采样步数（DDIM/respaced）<input id="steps" type="number" min="2" max="1000" value="100"></label>
<button onclick="generate()">生成并观察过程</button></section><div id="status" class="status"></div>
<section class="card results"><div><h2>最终样本</h2><img id="final"></div><div><h2>去噪轨迹</h2><div id="timeline" class="timeline"></div></div></section>
<script>async function generate(){let s=document.querySelector('#status');s.textContent='正在反向扩散…';document.querySelector('#timeline').innerHTML='';
let payload={class_id:+document.querySelector('#class').value,cfg_scale:+document.querySelector('#cfg').value,seed:+document.querySelector('#seed').value,trajectory_frames:+document.querySelector('#frames').value,sampler:document.querySelector('#sampler').value,sampling_steps:+document.querySelector('#steps').value};
try{let r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let d=await r.json();if(!r.ok)throw Error(d.error);document.querySelector('#final').src=d.image;
document.querySelector('#timeline').innerHTML=d.trajectory.map((x,i)=>`<div class="frame"><img src="${x}"><small>t=${d.timesteps[i]}</small></div>`).join('');s.textContent=`完成：class=${d.class_name}, seed=${payload.seed}`;}catch(e){s.textContent='错误：'+e.message}}</script></body></html>"""


def image_data(image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def render_page(pipeline: DiTInference) -> bytes:
    """Render checkpoint strings as text, never as executable HTML."""

    options = "".join(
        f'<option value="{i}">{i} · {html.escape(str(name), quote=True)}</option>'
        for i, name in enumerate(pipeline.class_names)
    )
    metadata = " · ".join(
        f"{html.escape(str(key), quote=True)}="
        f"{html.escape(str(value), quote=True)}"
        for key, value in pipeline.metadata.items()
    )
    return HTML.replace("__OPTIONS__", options).replace(
        "__METADATA__", metadata
    ).encode()


def make_handler(pipeline: DiTInference):
    inference_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path != "/":
                self.send_error(404); return
            body = render_page(pipeline)
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/generate":
                self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                with inference_lock:
                    result = pipeline.generate(**request, upscale=8)
                payload = {"image": image_data(result.image), "trajectory": [image_data(x) for x in result.trajectory], "timesteps": result.timesteps, "class_name": pipeline.class_names[request["class_id"]]}
                status = 200
            except Exception as error:
                payload, status = {"error": str(error)}, 400
            body = json.dumps(payload).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

        def log_message(self, format, *args):
            print(f"[web] {format % args}")
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    pipeline = DiTInference.from_checkpoint(
        args.checkpoint, args.device, use_ema=args.use_ema
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline))
    print(f"DiT Studio: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
