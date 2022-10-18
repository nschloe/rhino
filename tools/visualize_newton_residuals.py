# -*- coding: utf-8 -*-
#
import matplotlib
import numpy as np
import yaml

# Use the AGG backend to make sure that we don't need
# $DISPLAY to plot something (to files).
matplotlib.use("agg")
import matplotlib2tikz
import matplotlib.pyplot as pp


def _main():
    args = _parse_input_arguments()
    # read the file
    handle = open(args.filename)
    data = yaml.load(handle)
    handle.close()

    # Plot Newton residuals.
    # Mind that the last Newton datum only contains the final ||F||.
    num_newton_steps = len(data["Newton results"])
    x = list(range(num_newton_steps))
    # Extract Newton residuals
    y = np.empty(num_newton_steps)
    for k in range(num_newton_steps):
        y[k] = data["Newton results"][k]["Fx_norm"]
    # Plot it.
    pp.semilogy(x, y)

    pp.xlabel("Newton step")
    pp.ylabel("||F||")
    pp.title(
        "Krylov: %s    Prec: %r    ix-defl: %r    extra defl: %r    ExpRes: %r    Newton iters: %d"
        % (
            data["krylov"],
            data["preconditioner type"],
            data["ix deflation"],
            data["extra deflation"],
            data["explicit residual"],
            num_newton_steps,
        )
    )

    # Write the info out to files.
    if args.imgfile:
        pp.savefig(args.imgfile)
    if args.tikzfile:
        matplotlib2tikz.save(args.tikzfile)
    return


def _parse_input_arguments():
    """Parse input arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Newton residuals.")
    parser.add_argument("filename", metavar="FILE", type=str, help="Newton data file")

    parser.add_argument(
        "--imgfile",
        "-i",
        metavar="IMG_FILE",
        required=True,
        default=None,
        const=None,
        type=str,
        help="Image file to store the results",
    )

    parser.add_argument(
        "--tikzfile",
        "-t",
        metavar="TIKZ_FILE",
        required=True,
        default=None,
        const=None,
        type=str,
        help="TikZ file to store the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _main()
