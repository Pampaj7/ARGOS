# ARGOS v2 candidate-conditioned stereo-evidence information probe

This directory compares the strict validated 128-channel/8-block selector
with its frozen 13 universal inputs against the same selector augmented with
candidate-conditioned current-frame census correspondence evidence.

The three-seed augmented campaign is launched through
`scripts/launch_utility_selector_information_probe.sh`.  It trains and
calibrates entirely on the strict train/dataset-2 validation split before it
ever reads dataset 7.  The baseline control is the completed three-seed
campaign at `results/utility_selector_capacity_probe/medium_128x8_12ep_clean/`.

Temporary smoke and overfit outputs were deleted after their contracts passed.

## Completed strict result

All three B3/full-census seeds trained on datasets 1/3/6, selected and
calibrated only on dataset 2, and then evaluated once on dataset 7.  No
unseen-backbone or OOD data was loaded because the predeclared seen promotion
gate failed.

| Final dataset-7 metric | 13-map control (mean ± sample std) | + full 37-map matching evidence (mean ± sample std) |
| --- | ---: | ---: |
| raw EPE | 0.533685 | 0.533685 |
| selector EPE | 0.519577 ± 0.002422 | 0.519434 ± 0.000969 |
| EPE gain | 0.014109 ± 0.002422 | 0.014251 ± 0.000969 |
| raw-or-memory oracle gain | 0.052326 | 0.052326 |
| oracle recovery | 26.963% ± 4.629% | 27.235% ± 1.852% |
| intervention coverage | 0.989% ± 0.165% | 1.872% ± 0.722% |
| false-update rate | 0.548% ± 0.143% | 1.287% ± 0.487% |
| clean-pixel degradation | 0.286% ± 0.074% | 0.543% ± 0.188% |
| intervention precision | 47.283% ± 2.910% | 52.501% ± 6.614% |

The treatment changes mean recovery by only +0.272 percentage points, far
below the preregistered 50% promotion gate, while increasing intervention
coverage and both safety costs.  It is therefore a **NO-GO**: this compact
census-based candidate-conditioned matching evidence is not the missing
generalizable observable signal for the strict raw-versus-BiDA-memory selector.

The full treatment reports are in `full_128x8/`; the frozen control is in
`../../utility_selector_capacity_probe/medium_128x8_12ep_clean/`.  `B1` and
`B2` were deliberately not trained: after the predeclared full representation
failed, their narrower subsets could not promote the method and would only add
an uninformative campaign.
