#!/bin/bash
# Script to automate systematic weight loading experiments for Recurrent DeBERTa on RTE

LAYERS=(-1 0 4 14 20)

for layer in "${LAYERS[@]}"
do
    echo "=========================================================="
    echo "Starting experiment with Layer Index: $layer"
    echo "=========================================================="
    
    # Run the RTE training script with the specified layer
    ./experiments/glue/rte.sh recurrent-deberta-v3-large $layer
    
    echo "Experiment with Layer $layer finished."
    echo "=========================================================="
done

echo "All experiments completed!"
