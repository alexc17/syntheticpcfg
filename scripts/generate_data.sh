#!/bin/bash
seed=1
n=100
lex=1000
lr=1000
nt=10
sampler="../../syntheticpcfg/syntheticpcfg/sample_grammar.py"
for br in 20 30 40 50 60 70 80
do
    direct="../data/test$br"
    mkdir -p $direct
    python $sampler --seed $seed --nonterminals $nt --numbergrammars $n --lexicalproductions $lr --terminals $lex --binaryproductions $br "${direct}/grammar%d.pcfg"
done
echo "Done"
