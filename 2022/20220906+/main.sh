#!/bin/bash

COUNTER=0
while [  $COUNTER -lt 110 ]; do
    python -u main.py train_config.json
    rm -rf results
    let COUNTER=COUNTER+1 
done