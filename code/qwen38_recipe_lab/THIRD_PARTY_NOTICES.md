# Third-party notices

Text model equations, visual encoder and interleaved mRoPE, GR/PLE/hashing semantics and test references derive from Qwen Team / Hugging Face Transformers `qwen4_exp`, copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0. See `licenses/Apache-2.0.txt` and the exact revision and hashes in `sources.lock.json`.

Polar Express coefficient generation is by Noah Amsel et al., MIT License. See `licenses/PolarExpress-MIT.txt`; the exploratory schedule in `polar_coefficients.amsel.json` was generated from the pinned upstream generator and is retained for the superseded exploratory lineage; it is not the current optimizer schedule.

Cached source and tokenizer assets are downloaded separately by `bootstrap.py`; no official model weights are redistributed. The model/tokenizer license is governed by its own model repository; tokenizer files stay in the ignored local cache.

Current optimizer constants, normalization, momentum and mixed-precision update order follow the pinned paper-author Megatron-Canzona `qwen_express` implementation. See `licenses/Canzona-LICENSE.txt` and `sources.lock.json`. The report does not identify the production launch selector, so that remains a documented gap.

CPU pixel/frame resizing, normalization, RGBA compositing, patch layout and timestamp handling follow the locked Hugging Face Transformers Qwen2-VL/Qwen3-VL processor references, Apache-2.0; `sources.lock.json` records each source. Torchvision and Pillow are installed dependencies with their own licenses, not vendored here.

Task-local Colima and Lima release artifacts are downloaded separately and SHA256-verified (`runtime.lock.json`); their respective upstream licenses apply. No VM disk, binary release or container layer is committed. Evaluation datasets retain their publishers' licenses and access restrictions; immutable source URLs and byte hashes are recorded separately in `eval_sources.lock.json`.

MATH answer normalization is loaded from the hash-locked hendrycks/math MIT source (`math_equivalence.py`); see `licenses/MATH-MIT.txt`. EvalPlus is installed separately at published version 0.3.1 under Apache-2.0; the wheel and judge/sanitizer source hashes are recorded in `evalplus_source.lock.json`, and expanded test releases retain their upstream license terms. Neither its package binaries nor benchmark data are committed. Public Redux2-ok is a candidate protocol from the independently locked lm-evaluation-harness template; it is not asserted to be Qwen's unpublished evaluator version.

MultiPL-E translator, cleaned originals and evaluation runners are cached separately at publisher revision3025a531af7450e7df8b96fe0440e9804480bbad; MIT license is preserved as licenses/MultiPL-E-MIT.txt. Dataset and runner revisions/SHA are in multipl_*lock.json. The eight-language runtime uses Debian packages under their respective licenses, plus javatuples1.2 (Apache-2.0, confirmed in its Maven POM); no compiler binaries, jar, dataset or image layers enter Git. The Mac wrapper adds a test-completion marker and uses a native arm64 subset runtime; it is not the publisher's all-language image.

PyAV16.1.0 is an installed BSD-3-Clause dependency, pinned with wheel hashes in uv.lock. Bundled FFmpeg libraries retain their own upstream licensing; their versions are recorded in runs/media-environment.json, and no binary is committed. The added processing_utils.py oracle is the same locked Apache-2.0 Transformers revision as the existing Qwen3-VL processor; sources.lock.json binds its original bytes.

长上下文补充：NVIDIA/RULER，Apache-2.0，固定db43f92ff634bbce1c33d1fb15858aec4bba509a；原LICENSE保留在.cache/long-context-sources/ruler-license。OpenAI MRCR数据卡声明MIT，固定f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d。源码/说明仅作本机协议审计，独立锁见long_context_sources.lock.json；未将来源下载视为模型评测。

MTP proposer/cache/sampling protocol derives from pinned vLLM sources (Apache-2.0, copyright contributors to the vLLM project), revision aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592. Additional cached sources are locked separately in mtp_decode_sources.lock.json and mtp_proposer_sources.lock.json; original copyright headers remain in ignored source files. No CUDA implementation or model weights are redistributed.

Additional generic MTP training documentation, module and tests from the same Megatron-Canzona revision are bound in mtp_training_sources.lock.json; their original headers are retained in cache under the existing licenses/Canzona-LICENSE.txt terms. This does not establish the Qwen3.8 production training selector or hyperparameters.

The additional vLLM Qwen config adaptation and loader sources/tests are separately pinned in mtp_loader_sources.lock.json (Apache-2.0, same revision). The isolated config-branch audit is not an executed vLLM engine.

Qwen3.5 control components and their reference tests use the Qwen/Hugging Face public Transformers reference, revision863d2620210d72fc0a9a03fdfe200bc9699154f6. The upstream source declares Copyright2025 The Qwen Team and The HuggingFace Inc. team, Apache-2.0; see qwen35_reference_sources.lock.json and licenses/Apache-2.0.txt. Tests execute selected pure Torch fallback bodies, excluding hub-dispatch wrappers. Raw source remains in the ignored cache; no official weights are redistributed.
