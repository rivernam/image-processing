# Compact Image-First UI Design

## Goal

Reduce unused control-panel space and make the image the dominant workspace while retaining every existing workflow and setting.

## Layout

- Replace the top horizontal control strip with a horizontal splitter containing:
  - A compact left control pane with `Train`, `Test Workflow`, and `Search Settings` sections stacked vertically.
  - A right workspace containing the image, collapsible results, and the existing progress footer.
- Give the left pane an initial width of approximately 320 pixels and the image workspace the remaining width.
- Set splitter stretch so window resizing primarily enlarges the image workspace.
- Keep the current three-step test workflow inside the left pane.
- Align controls to the top of their sections so shorter sections do not create large gaps between their contents.

## Image Workspace

- Give `ImageView` all remaining vertical stretch.
- Keep the progress footer visible at the bottom because it communicates active work and cancellation.
- Preserve all image interaction, ROI selection, match overlays, and result highlighting behavior.

## Results

- Add one compact toggle button below the image.
- Results tabs are hidden by default.
- Toggle text is `Show Results` while collapsed and `Hide Results` while expanded.
- Search completion updates the available table data but does not automatically expand the results, preventing unexpected image shrinkage.
- Collapsing results does not clear final or diagnostic rows.

## Diagnostics

- Rename `Show diagnostic candidates` to `Show pre-filter candidates`.
- Place it in a collapsed `Advanced Search Settings` subsection.
- Add explanatory text: `Shows up to 100 matches before overlap removal and result limiting.`
- The option remains disabled by default and retains its existing worker behavior.

## Behavior and Compatibility

- Do not change search, generation, training, persistence, or worker data flow.
- Preserve existing public widget attributes used by tests and signal wiring.
- Loading existing projects continues to restore settings without changing layout state.

## Testing

- Assert the left/right splitter structure and image stretch priority.
- Assert results are initially hidden and the toggle changes visibility without clearing rows.
- Assert diagnostic copy and advanced-section default visibility.
- Run the main-window UI suite and the full regression suite.
