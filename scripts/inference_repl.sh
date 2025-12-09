# Inference command for predictions_15.txt
python inference_15.py \
    --config_file config/DINO/DINO_valve_5scale.py \
    --checkpoint_path checkpoints/checkpoint_5scale_24epochs.pth \
    --input_folder data/testing_image \
    --output_file $1 \
    --confidence_threshold 0.15 \
    --batch_size 64 \
    --num_workers 2

python inference_4scale_15.py \
    --config_file config/DINO/DINO_valve_4scale.py \
    --checkpoint_path checkpoints/checkpoint_4scale_24epochs.pth \
    --input_folder data/testing_image \
    --output_file $2 \
    --confidence_threshold 0.15 \
    --batch_size 64 \
    --num_workers 2