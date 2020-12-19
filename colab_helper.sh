#!/bin/bash

# A script to help with launching this project in Colab Notebook.
# First you need to clone the repository, `cd` into it, then launch this script.
# You'll have to launch it twice, because Colab requires restarting kernel
# after installing packages.
# Folders are created just in case. Simply to avoid unnecessary headaches
# with non-existent directories.

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
