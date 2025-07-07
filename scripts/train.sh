#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

TRAIN_CSV="/data2/heyichao/AVI_track1_code/AVI-track1/data/train_data.csv"
VAL_CSV="/data2/heyichao/AVI_track1_code/AVI-track1/data/val_data.csv"
TEST_CSV="/data2/heyichao/AVI_track1_code/AVI-track1/data/val_data.csv"

AUDIO_DIR="/data2/public_datasets/AVI/AVI_features/text/SFR-Embedding-Mistral-easy-detaild-new-2048"
VIDEO_DIR="/data2/public_datasets/AVI/AVI_features/text/SFR-Embedding-Mistral-easy-detaild-new-2048"
TEXT_DIR="/data2/public_datasets/AVI/AVI_features/text/SFR-Embedding-Mistral-easy-detaild-new-2048"
AUDIO_DIM=4096
VIDEO_DIM=4096
TEXT_DIM=4096

BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=200
NUM_WORKERS=4
PIN_MEMORY="True"
OPTIM="adamw"

HCPdropout_audio=0.2
HCPdropout_video=0.2
HCPdropout_text=0.2
HCPdropout_pure_text=0.2
UNIFIED_DIM=512

HEADS_NUM=4
ATCdropout=0.3
VTCdropout=0.3
HIDDEN_DIM=256

ENHANCER_DIM=256
TFEdropout=0.1

RHdropout=0.2
Target_dim=1
NUM_MODALITIES=3
MODALITIES="audio,text,video"


declare -a QUESTIONS=("q3" "q4" "q5" "q6")
declare -a TRAITS=("Honesty-Humility" "Extraversion" "Agreeableness" "Conscientiousness")

CSV_DIR="./results/demo_results.csv"  # replace with your CSV directory


for i in "${!QUESTIONS[@]}"; do
  QUESTION=${QUESTIONS[$i]}
  LABEL_COL=${TRAITS[$i]}

  AUDIO_NAME=$(basename "$AUDIO_DIR")
  VIDEO_NAME=$(basename "$VIDEO_DIR")
  TEXT_NAME=$(basename "$TEXT_DIR")
  RUN_TIME=$(date +"%Y%m%d_%H%M%S")
  COMBINED_NAME="${QUESTION}_${LABEL_COL}_${AUDIO_NAME}_${VIDEO_NAME}_${TEXT_NAME}"
  OUTPUT_DIR="./save_ckpt/${COMBINED_NAME}/${RUN_TIME}/"
  OUTPUT_MODEL="${OUTPUT_DIR}/best_model.pth"
  LOSS_PLOT_PATH="${OUTPUT_DIR}/loss_plot.png"
  ARGS_DIR="./args_log/${AUDIO_NAME}_${VIDEO_NAME}_${TEXT_NAME}"

  mkdir -p "$OUTPUT_DIR"
  mkdir -p "$ARGS_DIR"

  echo "start training: $QUESTION -> $LABEL_COL"
  python ./train_task1.py \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --test_csv "$TEST_CSV" \
    --question "$QUESTION" \
    --label_col "$LABEL_COL" \
    --audio_dir "$AUDIO_DIR" \
    --video_dir "$VIDEO_DIR" \
    --text_dir "$TEXT_DIR" \
    --audio_dim "$AUDIO_DIM" \
    --video_dim "$VIDEO_DIM" \
    --text_dim "$TEXT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --pin_memory "$PIN_MEMORY" \
    --optim "$OPTIM" \
    --HCPdropout_audio "$HCPdropout_audio" \
    --HCPdropout_video "$HCPdropout_video" \
    --HCPdropout_text "$HCPdropout_text" \
    --HCPdropout_pure_text "$HCPdropout_pure_text" \
    --use_prompt \
    --unified_dim "$UNIFIED_DIM" \
    --heads_num "$HEADS_NUM" \
    --ATCdropout "$ATCdropout" \
    --VTCdropout "$VTCdropout" \
    --hidden_dim "$HIDDEN_DIM" \
    --enhancer_dim "$ENHANCER_DIM" \
    --TFEdropout "$TFEdropout" \
    --RHdropout "$RHdropout" \
    --target_dim "$Target_dim" \
    --num_modalities "$NUM_MODALITIES" \
    --modalities "$MODALITIES" \
    --output_model "$OUTPUT_MODEL" \
    --loss_plot_path "$LOSS_PLOT_PATH" \
    --log_dir "$OUTPUT_DIR" \
    --training_time "$RUN_TIME" \
    --args_dir "$ARGS_DIR"\
    --csv_dir "$CSV_DIR"

  if [ $? -eq 0 ]; then
    echo "train finished: $QUESTION -> $LABEL_COL"
    echo "model saved to: $OUTPUT_MODEL"
  else
    echo "Failed: $QUESTION -> $LABEL_COL, please check the logs."
  fi

done
