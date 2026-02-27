# SDPB Options Reference (This Workflow)

Options are read from `sdpb_options.json` and passed through `sdp.set_option(key, value)`.
Unsupported keys are filtered out by the Python driver.

Current default keys:
- `checkpointInterval`
  - Seconds between checkpoints.
- `maxIterations`
  - Maximum interior-point iterations.
- `maxRuntime`
  - Max runtime in seconds.
- `dualityGapThreshold`
  - Target duality-gap stopping threshold.
- `primalErrorThreshold`
  - Target primal-error threshold.
- `dualErrorThreshold`
  - Target dual-error threshold.
- `minPrimalStep`
  - Minimum primal step size (when supported).
- `minDualStep`
  - Minimum dual step size (when supported).

## Notes
- Keep this file and `sdpb_options.json` synchronized.
- Option support depends on SDPB/PyCFTBoot version.
- If you see unsupported-option warnings, remove/adjust those keys.
