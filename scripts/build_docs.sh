#!/bin/bash

#source venv/bin/activate

cd ../docs || exit
#make clean
make html
cd - || exit

