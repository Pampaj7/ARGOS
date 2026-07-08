# ARGOS v2 Safe Causal BiDA Warm-Start Plan

Do not retrain in this diagnostic task.

Next minimal experiment:
1. Load the FaithfulCausalBiDA checkpoint.
2. Copy faithful core weights into SafeCausalBiDA.
3. Initialize gate bias open, around `+2`.
4. Start safe/sparse weights at zero for a short burn-in.
5. Ramp safe/sparse to the target values only after modified ratio and MAE match faithful.
6. Select checkpoint with MAE plus New-Bad3 constraint, not MAE alone.
