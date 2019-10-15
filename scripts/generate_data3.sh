#!/bin/bash
seed=1
n=100
lex=1000
nt=10

sampler="../syntheticpcfg/sample_fullgrammar.py"
for bd in 0.025 0.050 0.075 0.100 
do
    direct="../data/test$bd"
    mkdir -p $direct
	python $sampler --dirichletbinary ${bd} --verbose --seed $seed --nonterminals $nt --numbergrammars $n  --terminals $lex  "${direct}/grammar%d.pcfg" > "${direct}/sampling.log" &
done
echo "Done"
