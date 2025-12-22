#!/bin/bash

# Base configuration parameters
epochs=500
batch_size=256
patience=50
lr=0.0001
weight_decay=0.01

file_path=('data/RG1.csv' 'data/RG2.csv')

# TRAINING
combinations=(
"R2C _base 2 16 0.3 6 12 1 0"
"R2Cms _base 2 16 0.3 6 12 1 0"
)

for combo in "${combinations[@]}"
do
    read -r arc model_id num_layers hidden_units dropout forecast lookback numfeats ar <<< "$combo"

    echo "******************** ${arc^^} - ${model_id^^} ********************"
    # Run the script for each file with base configuration
    for file in "${file_path[@]}"
    do
        python3 -u sc/freeze_traintest.py \
            --file_path $file \
            --arc $arc \
            --forecast $forecast \
            --lookback $lookback \
            --ar $ar \
            --batch_size $batch_size \
            --epochs $epochs \
            --patience $patience \
            --lr $lr \
            --num_feats $numfeats \
            --weight_decay $weight_decay \
            --dropout $dropout \
            --hidden_units $hidden_units \
            --num_layers $num_layers \
            --model_id $model_id
    done
done

: '
# FOR TESTING
for checkpoint in models/final_checkpoints/*; do
    folder_name=$(basename "$checkpoint")

    echo "******************** Testing checkpoint: $folder_name ********************"

    # Determine which file to use based on prefix (RG1 or RG2)
    if [[ $folder_name == RG1* ]]; then
        file="data/RG1.csv"
    elif [[ $folder_name == RG2* ]]; then
        file="data/RG2.csv"
    else
        echo "Skipping folder $folder_name (does not match RG1* or RG2*)"
        continue
    fi

    python3 -u sc/freeze_test.py \
        --file_path "$file" \
        --seed $seed \
        --batch_size $batch_size \
        --epochs $epochs \
        --epochsinfo $epochsinfo \
        --patience $patience \
        --lr $lr \
        --weight_decay $weight_decay \
        --num_layers $num_layers \
        --hidden_units $hidden_units \
        --checkpoint "$checkpoint"
done
'
