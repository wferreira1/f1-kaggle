#!/bin/bash

DATA_DIR="./data/in"

mkdir -p "$DATA_DIR"

kaggle competitions download \
  -c playground-series-s6e5 \
  -p "$DATA_DIR"

unzip "$DATA_DIR/playground-series-s6e5.zip" \
  -d "$DATA_DIR"