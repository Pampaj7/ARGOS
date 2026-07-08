# MPC Safety Failure Audit

Checkpoint: `results/03_temporal_refinement/training/magnitude_proposal_critic_refiner/checkpoints/best_pareto.pt`.

Current selected MAE `9.1788`, gap `47.47%`, new-Bad3 frame mean `5.12%`.

Support precision `65.51%`, recall `81.21%`; proposal sign accuracy `72.73%`. Top 1% applied magnitudes cause `0.02%` of new-Bad3.

Conclusion: keep the large proposal branch; train a counterfactual verifier to authorize the proposal by predicted benefit/new-Bad3 risk and safe step size.
