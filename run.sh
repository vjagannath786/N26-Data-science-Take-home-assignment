#!/bin/bash
pip3 install -r high_requirements.txt;
echo $1
cp $1 ./input/private/;
cd src;
python3 compare.py;
python3 app.py;