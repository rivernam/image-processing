# ROI Template Preview Design

## Goal

Keep the trained ROI template visible in the UI so users can compare the search target with test images without sacrificing the main image workspace.

## Layout

- Add a labeled `ROI Template` preview at the bottom of the left `Train` group.
- Use a compact preview area approximately 240 by 140 pixels within the existing control pane.
- Preserve the template's aspect ratio and never upscale beyond its native pixel size.
- Before training, show centered text `No ROI template`.

## Behavior

- After successful `Train from ROI`, display the cropped trained template.
- After successful `Train from File`, display the full trained template.
- After project loading restores a model, display the restored model template.
- Loading or displaying test/generated images must not replace or clear the ROI preview.
- Clicking a populated preview displays the ROI template in the central `ImageView` for larger inspection.
- Clicking an empty preview has no effect.

## Data Flow

- The preview source is always `TrainModel.color`, ensuring the displayed image exactly matches the color template used by the matcher.
- `_set_model` updates the preview whenever the active model changes.
- Preview updates are presentation-only and do not mutate the model or training image.

## Errors and Compatibility

- Existing training and project-load errors remain unchanged.
- No persistence schema changes are required because the preview is derived from the restored model.
- Existing widget attributes and search/generation flows remain unchanged.

## Testing

- Verify the empty preview state.
- Verify `_set_model` displays `TrainModel.color` in the preview.
- Verify test-image loading does not clear the preview.
- Verify clicking a populated preview sends the template to the central `ImageView`.
- Run the main-window UI suite and the full regression suite.
