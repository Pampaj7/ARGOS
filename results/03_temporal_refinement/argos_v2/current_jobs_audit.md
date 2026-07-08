# ARGOS v2 Current Jobs Audit

Generated: 2026-07-07

## Active LSF Jobs Observed

```text
JOBID      USER    STAT  QUEUE      FROM_HOST                         EXEC_HOST   JOB_NAME   SUBMIT_TIME
28881733   leopam  RUN   p1i        hpclogin9.hpccluster.dtu.dk       n-62-12-83  qrsh       Jul  7 08:56
28882775   leopam  RUN   p1i        hpclogin9.hpccluster.dtu.dk       n-62-12-83  qrsh       Jul  7 11:48
28882778   leopam  RUN   p1i        hpclogin9.hpccluster.dtu.dk       n-62-12-83  qrsh       Jul  7 11:49
28883882   leopam  RUN   p1i        hpclogin9.hpccluster.dtu.dk       n-62-12-83  qrsh       Jul  7 15:59
```

## Interpretation

These are interactive `qrsh` jobs. No active submitted NVDS-lite matrix/training job was identified from the current LSF listing. I did not stop or restart any job.

## Relevant Historical NVDS Logs

- `logs/nvds_aux_cache.log`: cache generation completed.
- `logs/nvds_valflow.log`: flow validation completed.
- `logs/nvds_smoke.log`: stale smoke with old identity-collapse setup.
- `logs/nvds_validate_all.log`: stale old setup, A/B only.
- `logs/nvds_diag_pilot.log`: failed after training due to output path error and shell typo.
- `logs/nvds_matrix.log`: terminated.
- `logs/nvds_matrix_g0.log`: loaded shards but produced no run outputs.
- `logs/nvds_matrix_g1.log`: contains D seed0 log metrics but no reusable artifacts were found.
- `logs/argos_v2_nvds_matrix_g*.local.out`: prior local attempt loaded shards only; invalid.

## Operational Issue

The successful GPU work used interactive `p1i` sessions on `n-62-12-83`. A prior local/background launch from the login host was invalid and should not be repeated. The next actual run should use a clean LSF/interactive pattern from a compute allocation, with output-root creation verified before training.
