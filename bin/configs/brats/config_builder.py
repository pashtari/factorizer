import glob
import re


for fold in range(5):
    for path in glob.glob("config_brats_fold0_*.yaml"):
        with open(path, "r") as f:
            cfg = f.read()

        cfg = re.sub("split: 0", f"split: {fold}", cfg)
        cfg = re.sub("fold0", f"fold{fold}", cfg)
        path = re.sub("fold0", f"fold{fold}", path)
        with open(path, "w") as f:
            f.write(cfg)
