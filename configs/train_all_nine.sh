#!/usr/bin/env bash
# The nine PICS coverage runs of the manuscript (Table 3, Fig. 8): spatial {1, 10, 20} %
# x temporal {1, 10, 20} %. Run from the repository root with Dataset/3D_Pristine.mat in place.
# Each run writes Results/forward_runs/sp<S>_t<T>_silu_F256_h256x256x256_a1_hpc/.
# Every setting below is what the shipped config.json files record. The sp20/t10 run used
# --epochs 2000 and stopped at the 48 h wall limit after 640 epochs; all others used 600.
set -e
COMMON="--activation silu --num-freq 256 --hidden 256 256 256 --balance-alpha 1.0 \
        --data-only-epochs 30 --patience 50 --dtype float64 --batch-size 16384 \
        --ckpt-every 5 --resume --tag-suffix _hpc"

for S in 1 10 20; do
    python Codes/step1_generate_dataset.py --pct $S
    CSV=Dataset/dataset_layerwise_${S}pct_3ply_fullsignal_6001steps.csv
    for T in 0.01 0.10 0.20; do
        EPOCHS=600
        if [ "$S" = "20" ] && [ "$T" = "0.10" ]; then EPOCHS=2000; fi
        python Codes/step2_train.py --csv $CSV --temporal $T --epochs $EPOCHS $COMMON
    done
done
