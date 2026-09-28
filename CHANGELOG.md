# Changelog

All notable public changes to `bistable-cert` are documented here.

## 1.0.0 — 2026-09-24

- First public release of the Theorem A/B certificate layer (Theorem B is
  Theorem 4.3 of the paper and covers every initial state in the certified
  source box, off-diagonal ones included).
- Renamed the temporary pre-release distribution `rlds-network` to
  `bistable-cert`; the Python import is now `bistable_cert`.
- Added public release metadata, citation information, and bundle checksums.
- Preserved the audited certificate implementation from internal version
  0.1.5 without changing formulas or numerical behavior.
- Added input validation to the simulation utilities (`simulate_write`,
  `equilibrium`, `WriteSimulation`), matching the certificate layer's own
  checks, after a review of the release candidate found e.g.
  `target_node=-1` silently applying no pulse, negative durations
  integrating backward in time, and nan/inf inputs hanging. Results for
  valid inputs are unchanged.
- Corrected the description of "Theorem B\*": it is the full-box case of
  Theorem 4.3 (implemented), not robustness over an interval of amplitudes.
- `probes/certify_probe.py` now defaults to the 500 trials its archived
  results were produced with (it defaulted to 400).
- In the release bundle, the package directory is named `bistable-cert/`
  rather than `bistable_cert/`: the same-named directory shadowed an
  editable install as an empty namespace package when Python was started
  from the bundle root.
- Validated 111 tests, the Section 6 reference reproduction, all three
  archived probe results, wheel creation, and installation into a clean
  environment.

## Pre-release history

Internal versions 0.1.0–0.1.5 were used for adversarial review and defect
correction before the public release. Their detailed audit trail is retained
in `README.md` under “Pre-release audit and correction history.”
