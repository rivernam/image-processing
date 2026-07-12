# ROI Saturation Variation Design

## Goal

Extend ROI-only color augmentation so generated objects can range from fully grayscale to their original saturation while retaining the existing Hue variation and unchanged backgrounds.

## Generation

- Add `saturation_scale_range: tuple[float, float]` to `GenerationSettings` with default `(0.0, 1.0)`.
- Validate two finite ordered values within `[0.0, 2.0]`; values above `1.0` remain available for explicitly requested oversaturation.
- Draw one seeded `saturation_scale` per generated sample.
- In the existing ROI HSV transformation, rotate Hue and multiply the Saturation channel by `saturation_scale`, clipping to `[0, 255]`.
- Apply both transformations to the resized ROI before compositing so background pixels remain unchanged.
- `0.0` produces a grayscale ROI, `1.0` preserves original saturation, and intermediate values produce partial desaturation.

## Metadata and Compatibility

- Add `saturation_scale: float = 1.0` to `TransformRecord`.
- Persist `saturation_scale_range` in project generation settings and `saturation_scale` in sample metadata.
- Legacy project files without the range load as `(1.0, 1.0)`.
- Legacy sample metadata without the value loads as `1.0`.
- These legacy defaults preserve the appearance of previously generated samples.

## UI

- Add `Saturation min` and `Saturation max` double spin boxes below the Hue controls.
- Range: `0.0` through `2.0`; defaults: `0.0` and `1.0`; step: `0.1`; two decimal places.
- Include values in `generation_settings()` and restore them in `_apply_generation_settings()`.

## Testing

- Verify invalid settings ranges are rejected.
- Verify scale `0.0` produces equal BGR channels inside the ROI while the background stays unchanged.
- Verify scale `1.0` preserves saturation when Hue shift is zero.
- Verify seeded reproducibility and transform metadata.
- Verify new and legacy persistence behavior.
- Verify UI defaults, construction, and restoration.
- Run focused generator, persistence, and UI tests followed by the full suite.
