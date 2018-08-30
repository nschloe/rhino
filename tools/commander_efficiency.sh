#!/bin/sh

BINDIR=`dirname $0`
FILENAME="${BINDIR}/../initial3d.vtu"
#../interesting-22000.vtu
#python -u $BINDIR/unit_timer.py $FILENAME -t daxpy inner jacobian prec cycles exact -n 100 2> timings.err | tee timings.yaml

for P in `seq 0 20`; do 
    python -u $BINDIR/newton.py $FILENAME -k minres -p cycles -r -n $P    2>> newton.err    | tee -a newton.yaml
    python -u $BINDIR/newton.py $FILENAME -k minres -p cycles -r -n $P -d 2>> newton-ix.err | tee -a newton-ix.yaml
done
#use of already computed timings: timings_andré
python -u $BINDIR/visualize_efficiency.py -v newton.yaml -n newton.yaml    -f timings_andre.yaml -i newton.png    -t newton.tex
python -u $BINDIR/visualize_efficiency.py -v newton.yaml -n newton-ix.yaml -f timings_andre.yaml -i newton-ix.png -t newton-ix.tex
