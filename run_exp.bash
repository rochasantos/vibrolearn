#!/bin/bash

# Run the experiment with the specified parameters
for i in {1..10}; do
    for setup in dataset/cwru/*.json; do
        python main.py -e $setup -c RandomForestClassifier > results/$(basename $setup .json)_${i}.txt 
        python main.py -e $setup -a -c RandomForestClassifier > results/$(basename $setup .json)_${i}_aug.txt
    done
done
