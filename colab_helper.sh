#!/bin/bash

mkdir -p tmp && \
mkdir -p logs && \
mkdir -p results/main && \
mkdir -p results/baseline && \
mkdir -p results/colab && \
mkdir -p models/colab

FILE=rus-ukr.tar

if [ ! -f "$FILE" ]; then
    wget https://object.pouta.csc.fi/Tatoeba-Challenge/rus-ukr.tar
    tar -xvf rus-ukr.tar
fi

pip install -r requirements.txt
