# Direct C1 training v3: result

This preregistered low-coverage pilot failed its held-composition test. A 23.17M-parameter latent Video DiT was trained from scratch for 30,000 optimizer updates on 102 training videos. Training loss decreased substantially, and IID full-triple accuracy improved to 8/24 (33.3%), but the held `cat|playing|indoors` C1 accuracy was 0/28. C1 action accuracy was also 0/28.

The result is not evidence of unseen compositional generalization. The dataset has only 20 independent held groups rather than the formal 50-group requirement, and the non-held `cat|playing` pair has 6 groups rather than 10. Two independent prompt-hidden raters reviewed all 52 generated videos; disagreements counted as incorrect.

The larger model raises a memorization risk but does not prove visible copying. Frozen CLIP retrieval was only 15.4%, exactly equal to its shuffled-prompt control, while its three nearest-training comparisons had similarities of 0.977–0.979. Those pairs share low-frequency color/composition after downsampling, but the user correctly judged the displayed generation itself to be moving mosaic with no clearly recoverable original clip. Accordingly, the honest conclusion is: more capacity and optimization improved in-distribution fitting, but did not solve recognizable text-conditioned C1 composition; CLIP/PSNR nearest-neighbor evidence must not be overstated as frame-identical copying.

Artifacts:

- `blind-human-evaluation.json`: two-rater IID/C1 factor and triple results with Wilson intervals.
- `condition-alignment.json`: frozen CLIP matched/shuffled diagnostic and nearest training IDs.
- `nearest-audit.json` and `nearest-top*.png`: top-three feature/layout similarity audit; not proof of visible copying.
- `diagnostic-vae-reconstruction.png` / `diagnostic-generated-mosaic.png`: VAE oracle versus fixed free-generation comparison.
- `training-summary.json` and `training-loss-every-100-steps.csv`: training evidence.
- `preregistered-first-generation.mp4`: record 1 of the frozen grid, not a quality-selected sample.
- `evidence.json`: hashes and the complete evidence index.

The local ignored checkpoint is `code/video_dit/runs/c1-direct-train-v3/train/video_dit_inference.pt`, SHA-256 `07b602b865b0afb09094b48efb6b2024ae0e7ca7052982cf70ce2a9257a93adc`.
