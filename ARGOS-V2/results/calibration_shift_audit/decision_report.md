# D0 decision report

## Decision

**B — feature-space support detector is sufficient**

Failed OOD domains are separated from SCARED-C/CRES in the frozen penultimate feature space.

Quantitative frozen-audit indicators:

- CRES feature overlap with the SCARED-C reference NN support: 0.928
- mean SERV-CT/D4D feature overlap: 0.039
- incorrect authorization among authorized GT pixels on SERV-CT/D4D: 0.615
- frozen ultra-safe authorization still covers SERV-CT 0.433 and D4D 0.285

Thus **A is rejected**: the already frozen ultra-safe probability threshold
does not materially abstain on either failed shift. **B is supported** because
CRES remains in support while both failed domains are almost entirely outside
it. This is a diagnostic recommendation, not an OOD threshold fit; the audit
does not alter the frozen pipeline.
