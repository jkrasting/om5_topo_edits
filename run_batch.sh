#!/bin/bash

# This shell script will loop over all notebooks and execute them in batch mode

# Specify which kernel to use
kernel_name="python3"

for notebook in *.ipynb; do
    filename=$(basename -- "$notebook")
    filename="${filename%.*}"

    output_directory="output/"
    mkdir -p ${output_directory}

    echo "Running $notebook ..."
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=$kernel_name "$notebook" --output "${output_directory}${filename}_executed.ipynb"

done

