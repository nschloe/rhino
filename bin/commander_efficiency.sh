#!/bin/sh

BINDIR=`dirname $0`
#python -u $BINDIR/unit_timer.py ../interesting-27001.vtu -t jacobian cycles exact prec inner daxpy -n 100 2> timings.err | tee timings.yaml

for P in `seq 0 1`; do 
    echo $P
    python -u $BINDIR/newton.py ../interesting-27001.vtu -k minres -p cycles -r -n $P 2>> newton.err | tee -a newton.yaml
    python -u $BINDIR/newton.py ../interesting-27001.vtu -k minres -p cycles -r -d -n $P 2>> newton-ix.err | tee -a newton-ix.yaml
done

python -u $BINDIR/visualize_efficiency.py -v newton.yaml -n newton.yaml -f timings.yaml -i newton.png -t newton.tex
python -u $BINDIR/visualize_efficiency.py -v newton.yaml -n newton-ix.yaml -f timings.yaml -i newton-ix.png -t newton-ix.tex

