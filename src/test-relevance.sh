#!/bin/bash

for (( i =0.1; i<=0.7;i+=0.05))
do
	./Program --parallel auto --relevance ${i}
done
