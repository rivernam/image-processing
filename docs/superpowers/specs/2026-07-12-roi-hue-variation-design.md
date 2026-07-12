# ROI Hue Variation Design

## Goal

Generate synthetic samples whose ROI template object has reproducible medium-strength color variation while leaving the background color unchanged.

## Generation

- Add `hue_shift_range: tuple[float, float]` to `GenerationSettings` with a default of `(-60.0, 60.0)` degrees.
- Validate the range as finite, ordered, and contained within `[-180.0, 180.0]`.
- Draw one `hue_shift_degrees` value per sample from the seeded random generator.
- Resize the BGR ROI template, convert only that resized template to HSV, rotate its Hue channel by `hue_shift_degrees / 2` modulo 180, convert it back to BGR, and then composite it onto the background.
- Preserve the current whole-image brightness, contrast, blur, and noise operations after compositing.
- A fixed Seed and identical settings must produce identical images and transform records.

## Metadata and Compatibility

- Add `hue_shift_degrees: float = 0.0` to `TransformRecord`.
- Store `hue_shift_range` in project generation settings.
- Store `hue_shift_degrees` in generated-sample metadata.
- When loading older project or sample files that omit the new fields, use `(0.0, 0.0)` and `0.0` respectively so previously saved work retains its original behavior.

## UI

- Add `Hue min (°)` and `Hue max (°)` double spin boxes to the synthetic test-image preparation panel.
- Allow values from `-180.0` through `180.0`, defaulting to `-60.0` and `60.0`.
- Include their values when constructing `GenerationSettings` and restore them when loading project settings.

## Errors

- Reject a minimum greater than the maximum.
- Reject persisted hue values outside `[-180.0, 180.0]` with a field-specific validation message.

## Testing

- Verify settings validation and seeded reproducibility.
- Verify ROI pixels change while background pixels remain unchanged when all other transforms are neutral.
- Verify zero hue shift preserves ROI pixels.
- Verify new and legacy project/sample persistence.
- Verify UI defaults, settings construction, and restoration.
- Run focused generator, persistence, and main-window tests followed by the full suite.
