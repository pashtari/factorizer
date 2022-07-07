import glob
import re

# read factorizer config
with open("config_brats_fold0_swin-factorizer.yaml", "r") as f:
    cfg = f.read()


# number of iterations
for iters in range(0, 21):
    cfg_new = re.sub("num_iters: 5", f"num_iters: {iters}", cfg)
    cfg_new = re.sub("results.csv", f"results_iters{iters}.csv", cfg_new)
    with open(
        f"config_brats_fold0_swin-factorizer_iters{iters}.yaml", "w"
    ) as f:
        f.write(cfg_new)


# rank (or compression ratio), HALS
for rank in range(1, 8):
    cfg_new = re.sub("rank: 1", f"rank: {rank}", cfg)
    cfg_new = re.sub("results.csv", f"results_hals_rank{rank}.csv", cfg_new)
    with open(
        f"config_brats_fold0_swin-factorizer_hals_rank{rank}.yaml", "w"
    ) as f:
        f.write(cfg_new)


# rank (or compression ratio), MU
for rank in range(1, 8):
    cfg_new = re.sub("rank: 1", f"rank: {rank}", cfg)
    cfg_new = re.sub("solver: hals", "solver: mu", cfg_new)
    cfg_new = re.sub("results.csv", f"results_mu_rank{rank}.csv", cfg_new)
    with open(
        f"config_brats_fold0_swin-factorizer_mu_rank{rank}.yaml", "w"
    ) as f:
        f.write(cfg_new)


# remove only one layer
for layer in range(1, 10):
    cfg_new = re.sub(
        "          - !ft.SegmentationFactorizer",
        f"""          - !ablate
            - !ft.SegmentationFactorizer
            - ["nmf", "factorize"]
            - !lambda "layer, sublayer: layer == {layer-1}" """,
        cfg,
    )
    cfg_new = re.sub("results.csv", f"results_rm{layer}.csv", cfg_new)
    with open(f"config_brats_fold0_swin-factorizer_rm{layer}.yaml", "w") as f:
        f.write(cfg_new)


# keep up to Lth block
for layer in range(1, 10):
    cfg_new = re.sub(
        "          - !ft.SegmentationFactorizer",
        f"""          - !ablate
            - !ft.SegmentationFactorizer
            - ["nmf", "factorize"]
            - !lambda "layer, sublayer: layer > {layer-1}" """,
        cfg,
    )
    cfg_new = re.sub("results.csv", f"results_keeptill{layer}.csv", cfg_new)
    with open(
        f"config_brats_fold0_swin-factorizer_keeptill{layer}.yaml", "w"
    ) as f:
        f.write(cfg_new)


# build 5-folf configs
for fold in range(5):
    for path in glob.glob("*_fold0_*.yaml"):
        with open(path, "r") as f:
            cfg = f.read()

        cfg = re.sub("split: 0", f"split: {fold}", cfg)
        cfg = re.sub("fold0", f"fold{fold}", cfg)
        path = re.sub("fold0", f"fold{fold}", path)
        with open(path, "w") as f:
            f.write(cfg)
