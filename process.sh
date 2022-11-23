#!/bin/bash

imgs="$1"
cd detect
python3 detect.py $imgs

cd ../align
python3 align.py $imgs