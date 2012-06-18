#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Visualize (1+T_d/T) k_d/k
'''
# ==============================================================================
import numpy as np

import yaml
import matplotlib
# Use the AGG backend to make sure that we don't need
# $DISPLAY to plot something (to files).
matplotlib.use('agg')
import matplotlib.pyplot as pp
import matplotlib2tikz
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    #TJ = 0.1
    #TMinv = 1.2
    #TM = 0.1
    #Tip = 0.03
    #Tdaxpy = 0.03

    # obtain timings
    timings = yaml.load(open(args.timings_file))
    for run in timings["tests"]:
        min_time = min(run["timings"]) / timings["number"]

        if run["target"] == "jacobian":
            TJ = min_time
        if run["target"] == "amg1":
            TMinv = min_time
        if run["target"] == "prec":
            TM = min_time
        if run["target"] == "inner":
            Tip = min_time
        if run["target"] == "daxpy":
            Tdaxpy = min_time

    # k - number of MINRES iteratios
    # p - number of deflation vectors
    def Tp(p):
        return p*(Tip + Tdaxpy)
    TMip = TM + Tip

    def Tqr(p):
        return (p*(p+1)/2) * (TMip + Tdaxpy)
    def Tget_proj(p):
        return p*TJ + p*(p+1)*Tip + Tp(p) + p*Tdaxpy
    def TMINRES(k, p):
        return k * (2*Tip + 2*TJ + TMinv + Tp(p) + 7*Tdaxpy)
    def Tget_ritz(k, p):
        return p*TMinv + p**2 * Tip + (p+k)*p*Tdaxpy
    def Toverall(k, p):
        return Tqr(p) + Tget_proj(p) + TMINRES(k, p) + Tget_ritz(k, p)

    vanilla_newton_data = list(yaml.load_all(open(args.vanilla_newton_data_file)))[0]
    assert vanilla_newton_data['ix deflation'] == False
    assert vanilla_newton_data['extra deflation'] == 0

    newton_steps = list(range(26))
    for newton_data_file in args.newton_data_files:
        newton_data = list(yaml.load_all(open(newton_data_file)))
        for step in newton_steps:
            x = []
            y = []
            for newton_datum in newton_data:
                if step < len(newton_datum['Newton results']) - 1:
                    num_vanilla_steps = len(vanilla_newton_data['Newton results'][step]['relresvec']) -1 
                    num_steps = len(newton_datum['Newton results'][step]['relresvec']) - 1

                    num_defl_vecs = newton_datum['extra deflation']
                    if newton_datum['ix deflation']:
                        num_defl_vecs += 1

                    x.append(num_defl_vecs)
                    y.append(Toverall(num_steps, num_defl_vecs) / Toverall(num_vanilla_steps, 0))

            pp.plot(x, y, color=str(1.0 - float(step+1)/len(newton_steps)), label='step %d' % step)
            #pp.plot(x, y, '-o', color='0.0')
            #pp.plot(x, y, '-o', color='0.6')

    pp.ylim([0, 2])
    pp.title('bla')
    pp.legend()

    # Write the info out to files.
    if args.imgfile:
        pp.savefig(args.imgfile)
    if args.tikzfile:
        matplotlib2tikz.save(args.tikzfile)

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Visualize Newton output.' )

    parser.add_argument('--vanilla-newton-data-file','-n',
                        metavar = 'FILE',
                        required = True,
                        type    = str,
                        help    = 'File containing vanilla Newton data (without ix-deflation)'
                        )

    parser.add_argument('--newton-data-files','-d',
                        metavar = 'FILE',
                        nargs = '+',
                        required = True,
                        type    = str,
                        help    = 'Files containing Newton data with ix-deflation'
                        )

    parser.add_argument('--timings-file','-f',
                        metavar = 'FILE',
                        required = True,
                        type    = str,
                        help    = 'MINRES timing'
                        )

    parser.add_argument('--imgfile', '-i',
                        metavar = 'IMG_FILE',
                        required = True,
                        type = str,
                        help = 'Image file to store the results'
                        )

    parser.add_argument('--tikzfile', '-t',
                        metavar = 'TIKZ_FILE',
                        required = True,
                        type = str,
                        help = 'TikZ file to store the results'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
