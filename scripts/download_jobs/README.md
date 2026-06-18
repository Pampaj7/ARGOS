# Download Jobs

This folder tracks dataset/checkpoint download commands and logs.

## Files

| File | Purpose |
|---|---|
| `download_scared_aria2.sh` | aria2-based SCARED download/resume script. |
| `download_scared_full.py` | Python SCARED download helper. |
| `download_training_extras.py` | Extra surgical stereo training data download helper. |
| `download_monsterpp_large.py` | MonSter++ checkpoint/data download helper. |
| `*_download.log` | Logs from long-running download jobs. |
| `scared_remaining_aria2.txt` | aria2 URL queue for remaining SCARED files. |

Keep long-running downloads in `tmux` or `screen`, and keep logs here so status checks are easy.

