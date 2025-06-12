#!/bin/bash

# Script to extend vocabulary for all languages
echo "Starting vocabulary extension for all languages..."

# Array of language directories and their corresponding ISO codes
declare -A languages=(
    ["afrikaans"]="af"
    ["akan"]="ak"
    ["akuapem_twi"]="tw"
    ["amharic"]="am"
    ["arabic"]="ar"
    ["asante_twi"]="tw"
    ["hausa"]="ha"
    ["igbo"]="ig"
    ["kinyarwanda"]="rw"
    ["luganda"]="lg"
    ["pedi"]="nso"
    ["sesotho"]="st"
    ["shona"]="sn"
    ["swahili"]="sw"
    ["tswana"]="tn"
    ["twi"]="tw"
    ["xhosa"]="xh"
    ["yoruba"]="yo"
    ["zulu"]="zu"
)

# Counter for progress tracking
total=${#languages[@]}
current=0

# Loop through each language
for lang_dir in "${!languages[@]}"; do
    current=$((current + 1))
    lang_code="${languages[$lang_dir]}"
    
    echo "[$current/$total] Processing $lang_dir ($lang_code)..."
    
    python extend_vocab_config.py \
        --output_path=checkpoints/ \
        --metadata_path=dataset-1/$lang_dir/metadata_train.csv \
        --language=$lang_code \
        --extended_vocab_size=1024
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $lang_dir"
    else
        echo "✗ Error processing $lang_dir"
        exit 1
    fi
    
    echo ""
done

echo "All vocabulary extensions completed successfully!"