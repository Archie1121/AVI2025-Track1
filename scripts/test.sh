#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# replace with your own paths
ARGS_JSON="/data2/heyichao/AVI_track1_code/our_fusion_method/args_log_clip/SFR-Embedding-Mistral-easy-detaild-new-2048_clip_SFR-Embedding-Mistral-easy-detaild-new-2048_clip_SFR-Embedding-Mistral-easy-detaild-new-2048_clip/args_20250702_220655.json"
SUBMISSION_CSV="/data2/heyichao/AVI_track1_code/our_fusion_method/submission.csv"
TEST_CSV="/data2/heyichao/AVI_track1_code/our_fusion_method/data/test_data_basic_information.csv"


TRAIT="Extraversion"  #Honesty-Humility, Extraversion, Agreeableness, Conscientiousness


python /data2/heyichao/AVI_track1_code/our_fusion_method/test_task1.py \
    --args_json "$ARGS_JSON" \
    --submission_csv "$SUBMISSION_CSV" \
    --trait "$TRAIT" \
    --override_test_csv "$TEST_CSV"
