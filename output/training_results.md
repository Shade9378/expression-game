# Training results

[Technical details](technical_details.md) | [Setup](../README.md#setup)

## Training setup

| Parameter | CNNs | Feature classifiers |
| --- | --- | --- |
| Train / validation / test | 27,933 / 3,543 / 3,589 | 24,599 / 3,122 / 3,169 paired images |
| Initialization | ImageNet-1K V1 weights; all layers fine-tuned | New MLP; frozen seed-42 ResNet18 for image inputs |
| Seeds | 42, 43, 44 | 42 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 0.0003 | 0.001 |
| Weight decay | 0.01 | 0.01 |
| Batch size | 128 | 256 |
| Maximum epochs | 8 | 20 |
| Scheduler | CosineAnnealingLR, T_max=8, minimum LR=0 | CosineAnnealingLR, T_max=20, minimum LR=0 |
| Early stopping | Patience 3 epochs | Patience 5 epochs |
| Checkpoint selection | Validation macro-F1; minimum improvement 0.001 | Same |
| Loss | Weighted cross-entropy | Weighted cross-entropy |
| Precision | CUDA mixed precision + gradient scaling | FP32 |
| Training GPU | RTX 4060 Laptop, CUDA 12.4 | Same |

Class weights: sqrt(N / (7 * class_count)), normalized to mean 1 using the corresponding training split. Exact duplicate leakage/conflicting training labels are filtered; the official test set is retained. Feature comparisons use the same successfully extracted subset across all five classifiers.

CNN input: grayscale, bilinear resize to 112 x 112, repeated across three channels; mean (0.485, 0.456, 0.406), standard deviation (0.229, 0.224, 0.225).

Training augmentation only: horizontal flip (p=0.5); affine scale 0.95-1.05, translation +/-3%, rotation +/-10 degrees (p=0.6, reflected borders); brightness/contrast +/-0.1 (p=0.3).

## CNN classification

Test set: 3,589 images. PyTorch: mean +/- sample SD, seeds 42/43/44. ONNX: seed 42.

| Model | Format | Accuracy (%) | Macro-F1 |
| --- | --- | ---: | ---: |
| ResNet18 | PyTorch | 68.30 +/- 0.48 (3 seeds) | 0.6691 +/- 0.0047 (3 seeds) |
| MobileNetV3-S | PyTorch | 63.01 +/- 0.41 (3 seeds) | 0.6025 +/- 0.0046 (3 seeds) |
| EfficientNet-B0 | PyTorch | 68.10 +/- 0.11 (3 seeds) | 0.6617 +/- 0.0036 (3 seeds) |
| ResNet18 | FP32 | 68.82 | 0.6745 |
| ResNet18 | FP16 | 68.85 | 0.6748 |
| ResNet18 | INT8 | 68.15 | 0.6683 |
| MobileNetV3-S | FP32 | 62.69 | 0.5980 |
| MobileNetV3-S | FP16 | 62.72 | 0.5976 |
| MobileNetV3-S | INT8 | 17.41 | 0.1357 |
| EfficientNet-B0 | FP32 | 68.07 | 0.6662 |
| EfficientNet-B0 | FP16 | 68.04 | 0.6658 |
| EfficientNet-B0 | INT8 | 42.63 | 0.3718 |

All CNN runs trained for eight epochs. The reported test scores use weights saved after epoch 8, except EfficientNet-B0 seed 43, which uses epoch 7. FP16 and INT8 conversion uses seed-42 checkpoints; INT8 uses static per-channel QDQ quantization calibrated on 256 training-only samples.

## Feature comparison

Seed 42: freeze the trained ResNet18 to extract 512 image features; use MediaPipe for 96 normalized landmark coordinates. Each input combination below is trained on a separated MLP, training-only standardization, no image augmentation during extraction. MLP: input -> 128 -> ReLU -> Dropout(0.2) -> 64 -> ReLU -> 7.

All five classifiers use identical paired splits, including 3,169 test images.

| Model | Format | Accuracy (%) | Macro-F1 |
| --- | --- | ---: | ---: |
| Image embeddings | FP32 | 69.33 | 0.6627 |
| Image embeddings | FP16 | 69.33 | 0.6628 |
| Image embeddings | INT8 | 68.98 | 0.6562 |
| Landmark | FP32 | 59.55 | 0.5034 |
| Landmark | FP16 | 59.48 | 0.5029 |
| Landmark | INT8 | 59.32 | 0.4996 |
| Landmark + blendshape | FP32 | 59.51 | 0.5236 |
| Landmark + blendshape | FP16 | 59.51 | 0.5236 |
| Landmark + blendshape | INT8 | 59.32 | 0.5232 |
| Image + landmark | FP32 | 68.73 | 0.6647 |
| Image + landmark | FP16 | 68.73 | 0.6647 |
| Image + landmark | INT8 | 68.57 | 0.6602 |
| Image + landmark + blendshape | FP32 | 69.01 | 0.6640 |
| Image + landmark + blendshape | FP16 | 69.04 | 0.6642 |
| Image + landmark + blendshape | INT8 | 69.17 | 0.6638 |

### Feature training epochs

The last column identifies the saved model used for the reported test scores, chosen using validation macro-F1.

| Inputs | Epochs trained | Epoch used for test results |
| --- | ---: | ---: |
| Image only | 6 | 1 |
| Landmarks | 20 | 20 |
| Landmarks + blendshapes | 15 | 10 |
| Image + landmarks | 9 | 4 |
| Image + landmarks + blendshapes | 6 | 1 |
