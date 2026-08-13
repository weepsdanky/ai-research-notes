# DiT demo artifacts

这些文件随仓库交付，因此新 clone 不需要先训练就能复现本学习项目的推理闭环。完整 optimizer、scaler、RNG、日志和周期 checkpoint 仍在本机 `runs/` 中且不纳入 Git；这里的两个 `*_inference.pt` 仅包含模型状态和基础元数据，可由 CLI/UI 通过 PyTorch `weights_only=True` 加载。

## Image DiT

- `code/image_dit/demo_artifacts/image_dit_inference.pt`：MNIST canonical-overfit、v-prediction、3,000 updates、254,276 参数的推理 checkpoint。
- `code/image_dit/demo_artifacts/samples/class_8.png`：请求 class 8 的单样本；当前小模型的类别服从不足，这个样本更像 7，保留它是为了如实展示边界。
- `code/image_dit/demo_artifacts/samples/class_8_denoising.gif`：同一张图沿 diffusion sampling time 的诊断轨迹，不是视频。
- `code/image_dit/demo_artifacts/samples/ema_ddpm1000_same_noise.png`：固定噪声下各类别的完整 DDPM 网格。

## Video DiT

- `code/video_dit/demo_artifacts/video_dit_inference.pt`：canonical-overfit、concept + v-prediction、3,000 updates、2,472,112 参数的推理 checkpoint。
- `code/video_dit/demo_artifacts/samples/*.mp4`：猫向右、狗向左、红衣人物向右的 8 帧模型输出；这些才是视频时间轴。
- `code/video_dit/demo_artifacts/samples/*_denoising-diagnostic.gif`：生成整段视频时沿 diffusion sampling time 的诊断轨迹。
- `code/video_dit/demo_artifacts/samples/contacts/*.png`：每段 MP4 的 8 帧联系表，便于静态检查时间连续性。
- `code/video_dit/demo_artifacts/samples/temporal_metrics.json`：运动质心方向与帧间变化；主体语义仍以联系表人工检查为准。

## SHA-256

```text
f8ebbaff5623cac45232bb3a678bd8d18b8d15fff9a33450094f471ae39ca7fd  output/pdf/module3-dit-math-derivation.pdf
a174742891cd9c94b102c324e6c084ba69c18c7ef1f3a0b898ea56d227ddf8ef  code/image_dit/demo_artifacts/image_dit_inference.pt
2d2ba4c0461cb4b7f45aa01e6afc439d0cb4b280f7438a1cc9df10f1988e1c6a  code/image_dit/demo_artifacts/samples/class_8.png
663022870adc3df8877fde12a36a7ff4c41453d2901c8f4266e904f4a4961182  code/image_dit/demo_artifacts/samples/class_8_denoising.gif
e0a3247af6f095f7d5735ca1e2480c2c4dd55cff15e59d9055ef064cb642391b  code/image_dit/demo_artifacts/samples/ema_ddpm1000_same_noise.png
3409c4184a628f20c75635d15dcff1dea52460a80ab0381bc695ee34a6f629de  code/video_dit/demo_artifacts/video_dit_inference.pt
31bb7ade707e8f140f1bfe7f178ecb2353d19e69cbd56645e376294e52c0b7ad  code/video_dit/demo_artifacts/samples/model_cat_orange_walk_right_snow_seed-42.mp4
e03c78516f99eca88158aa45547d9a78ea6c8afefc1c84128b8496192f80b621  code/video_dit/demo_artifacts/samples/model_dog_brown_walk_left_grass_seed-42.mp4
d576a200059fa1ff5214e158f19aa6a553a94b8aae346fa66a43e3bade1daf11  code/video_dit/demo_artifacts/samples/model_person_red_walk_right_snow_seed-42.mp4
8c4bc71fad7ea3ec3ecd224f1855f6fa395264a7dd410a494653beedd6ef2953  code/video_dit/demo_artifacts/samples/model_cat_orange_walk_right_snow_seed-42_denoising-diagnostic.gif
2f8353cf2c342ecf25d4e72a540dd5f1e07f8dff630ef72c78a4d2a5d5477bd1  code/video_dit/demo_artifacts/samples/model_dog_brown_walk_left_grass_seed-42_denoising-diagnostic.gif
c818e15b8892322dd014473950e44e9a755bb764826b6212922b601d0ab373c4  code/video_dit/demo_artifacts/samples/model_person_red_walk_right_snow_seed-42_denoising-diagnostic.gif
942081dcb085814fefae9c4134a92d40653326250527e4e5ead8ea000e10a639  code/video_dit/demo_artifacts/samples/contacts/cat.png
5373cf174241228d8cb7d91cfaa0b5807980271777ac5e9716b252ce0ab35a93  code/video_dit/demo_artifacts/samples/contacts/dog.png
3b1c9fd57cea8ca1a8c32aa8066a7198a36ee6128e334d4e93da994381b6d2df  code/video_dit/demo_artifacts/samples/contacts/person.png
d5c404b983ab02c699403c72fac1e4f7c1cde6b4541ca5664fd5a2f4cf96ddd6  code/video_dit/demo_artifacts/samples/temporal_metrics.json
```

验证命令：

```bash
shasum -a 256 <path>
```
