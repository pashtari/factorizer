import glob
import re
from black import format_str, FileMode


for fold in range(5):
    for path in glob.glob("config_brats_fold0_*.py"):
        with open(path, "r") as f:
            cfg = f.read()

        cfg = re.sub('"split": 0', f'"split": {fold}', cfg)
        cfg = re.sub("fold0", f"fold{fold}", cfg)
        cfg = format_str(cfg, mode=FileMode(line_length=79))
        path = re.sub("fold0", f"fold{fold}", path)
        with open(path, "w") as f:
            f.write(cfg)

    # for path in glob.glob("config_btcv_fold0_*.py"):
    #     with open(path, "r") as f:
    #         cfg = f.read()

    #     cfg = re.sub('"split": 0', f'"split": {fold}', cfg)
    #     cfg = re.sub("fold0", f"fold{fold}", cfg)
    #     cfg = format_str(cfg, mode=FileMode(line_length=79))
    #     path = re.sub("fold0", f"fold{fold}", path)
    #     with open(path, "w") as f:
    #         f.write(cfg)
