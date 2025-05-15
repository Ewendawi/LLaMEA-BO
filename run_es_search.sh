#!/bin/bash

# define api key
export GEMINI_API_KEY=<YOUR_API_KEY>

export N_POPULATION=40

# -p: number of parents
# -o: number of offspring
# -e: enable elitism
# -k: api key
# -n: number of population
# -f: plot the results. If the flag is set, other parameters are ignored

# run 1+1 experiment
python run_es_search.py -p 1 -o 1 -e -k $GEMINI_API_KEY -n $N_POPULATION

# run 4,16 experiment
python run_es_search.py -p 4 -o 16 -k $GEMINI_API_KEY -n $N_POPULATION

# run 8+16 experiment
python run_es_search.py -p 8 -o 16 -e -k $GEMINI_API_KEY -n $N_POPULATION

# plot the results
# python run_es_search.py -f