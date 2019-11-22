#!/bin/bash

for  i in $(seq 0.1 0.05 0.7)
do
	./Program.py --parallel auto --relevance ${i}
done
