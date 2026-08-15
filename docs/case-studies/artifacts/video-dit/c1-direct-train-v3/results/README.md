# Direct C1 training v3: result

This preregistered low-coverage pilot failed its held-composition test. A 23.17M-parameter latent Video DiT was trained from scratch for 30,000 optimizer updates on 102 training videos. Training loss decreased substantially, and IID full-triple accuracy improved to 8/24 (33.3%), but the held `cat|playing|indoors` C1 accuracy was 0/28. C1 action accuracy was also 0/28.

The result is not evidence of unseen compositional generalization. The dataset has only 20 independent held groups rather than the formal 50-group requirement, and the non-held `cat|playing` pair has 6 groups rather than 10. Two independent prompt-hidden raters reviewed all 52 generated videos; disagreements counted as incorrect.

The larger model also shows clear memorization. Frozen CLIP retrieval was only 15.4%, exactly equal to its shuffled-prompt control, while its three nearest-training comparisons had similarities of 0.977–0.979. The contact sheets place generated frames on top and training frames below; two are near-frame-identical reconstructions. Accordingly, the honest conclusion is: more capacity and optimization improved in-distribution fitting, but did not solve text-conditioned C1 composition and amplified training-video reconstruction.

Artifacts:

- `blind-human-evaluation.json`: two-rater IID/C1 factor and triple results with Wilson intervals.
- `condition-alignment.json`: frozen CLIP matched/shuffled diagnostic and nearest training IDs.
- `nearest-audit.json` and `nearest-top*.png`: top-three visual memorization audit.
- `training-summary.json` and `training-loss-every-100-steps.csv`: training evidence.
- `preregistered-first-generation.mp4`: record 1 of the frozen grid, not a quality-selected sample.
- `evidence.json`: hashes and the complete evidence index.

The local ignored checkpoint is `code/video_dit/runs/c1-direct-train-v3/train/video_dit_inference.pt`, SHA-256 `07b602b865b0afb09094b48efb6b2024ae0e7ca7052982cf70ce2a9257a93adc`.
