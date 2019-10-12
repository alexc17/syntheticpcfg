#!/bin/bash
seed=1
n=100
lex=1000
nt=10

sampler="../syntheticpcfg/sample_fullgrammar.py"
for bd in 0.006 0.012 0.018 0.024 1.0
do
    direct="../data/test$bd"
    mkdir -p $direct
python $sampler --dirichletbinary ${bd} --verbose --seed $seed --nonterminals $nt --numbergrammars $n  --terminals $lex  "${direct}/grammar%d.pcfg"
done
echo "Done"
