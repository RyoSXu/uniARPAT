#!/bin/bash
python data/csv2npy.py
python -u train.py \
  --init_method "tcp://127.0.0.1:$PORT" \
  -c ./configs/config.yaml \
  --outdir "./output" \
  --world_size 1 \
  --desc "ARPAT" \
  --resume False \
  --smear 0 \
  --dos_minmax True \
  --dos_zscore False \
  --scale_factor 1.0 \
  --apply_log False \
  --seed 42