# Dataset audit — planning only

SCARED-C is the only currently supported supervised source, using quality-gated processed
temporal pseudo-GT causally after warmup. StereoMIS has no geometric reference. SERV-CT has
static CT-derived GT only. D4D has sparse anchors and is future zero-shot only. Hamlyn and
EndoSLAM are locally unavailable/unknown. Therefore cross-training, pooled training, and LODO
launches are blocked; see `evidence_audit.json` for file-level provenance.
