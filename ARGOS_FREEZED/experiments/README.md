# ARGOS v2 experiment workspace

Every future paper experiment lives in its own directory here. Lifecycle:

1. create a new experiment directory;
2. run `scripts/verify_freeze.py`;
3. state one scientific question and hypothesis;
4. freeze train/validation/test protocol;
5. freeze deterministic seeds;
6. record the frozen-core manifest hash;
7. run a small smoke test;
8. remove successful smoke outputs;
9. launch the full run and record infrastructure;
10. freeze the validation decision;
11. open test only after that freeze;
12. write compact CSV/JSON/log/README artifacts;
13. register the verdict in `EXPERIMENT_REGISTRY.csv`.

Experiments may not modify files outside their own directory.
