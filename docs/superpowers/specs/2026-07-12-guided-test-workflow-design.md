# Guided Test Workflow Design

## Goal

Present testing as one ordered workflow so users can see that existing images and generated images are alternative inputs to the same search action.

## Layout

Replace the two peer input groups inside `Test / Generator` with three numbered sections:

1. `1. Choose Test Source`
   - `Existing Images`
   - `Synthetic Test Images`
2. `2. Prepare Test Images`
   - Existing path: `Open Images to Search`
   - Synthetic path: `Add Backgrounds for Generation`, Count, Seed, and `Generate Test Images`
3. `3. Run Search`
   - One shared `Run Search` button below both preparation paths

The source choice controls which preparation controls are visible. `Existing Images` is selected initially. Switching source does not delete already loaded paths or backgrounds.

## Behavior

- Opening existing images continues to set them as the current search inputs.
- Generating test images continues to set generated samples as the current search inputs automatically.
- The shared `Run Search` button searches whichever test-image list is current.
- Training remains in the separate `Train` group and remains a prerequisite for generation and search.
- Search, generation, evaluation, persistence, and worker behavior do not change.

## Feedback and Errors

- Preserve existing status-bar messages and validation dialogs.
- The source selector changes presentation only; existing action enablement remains governed by the current model and test-image state.

## Testing

- Assert the three numbered section titles.
- Assert that one shared `Run Search` button is outside both preparation panels.
- Assert that selecting each source shows its matching controls and hides the other controls.
- Run the focused main-window suite and the full regression suite.
