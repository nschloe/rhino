#!/bin/sh

BINDIR=`dirname $0`
NEWTONBASE="python -u $BINDIR/newton.py ../interesting-27001.vtu -k minres -p cycles -r"

# vanilla
BASENAME="newton-vanilla"
$NEWTONBASE 2> $BASENAME.err | tee $BASENAME.yaml
python $BINDIR/visualize_newton_output.py $BASENAME.yaml -i $BASENAME.png -t $BASENAME.tex

BASENAME="newton-defl01"
$NEWTONBASE -n 1 2> $BASENAME.err | tee $BASENAME.yaml
python $BINDIR/visualize_newton_output.py $BASENAME.yaml -i $BASENAME.png -t $BASENAME.tex

BASENAME="newton-defl12"
$NEWTONBASE -n 12 2> $BASENAME.err | tee $BASENAME.yaml
python $BINDIR/visualize_newton_output.py $BASENAME.yaml -i $BASENAME.png -t $BASENAME.tex

BASENAME="newton-defl00-ix"
$NEWTONBASE -d 2> $BASENAME.err | tee $BASENAME.yaml
python $BINDIR/visualize_newton_output.py $BASENAME.yaml -i $BASENAME.png -t $BASENAME.tex

BASENAME="newton-defl11-ix"
$NEWTONBASE -n 11 -d 2> $BASENAME.err | tee $BASENAME.yaml
python $BINDIR/visualize_newton_output.py $BASENAME.yaml -i $BASENAME.png -t $BASENAME.tex
