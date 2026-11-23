"""Reweight a simulation towards a target radius of gyration (or any single
observable) read from a GROMACS .xvg file, e.g. the output of gmx gyrate,
optionally together with other data in BME format (chemical shifts, NOEs...).

Examples:

    gmx gyrate -f traj.xtc -s topol.tpr -o gyrate.xvg
    bme-rg gyrate.xvg --target 1.85 --sigma 0.05 --scan
    bme-rg gyrate.xvg --target 1.85 --sigma 0.05 --theta 2 --plot

    # Rg together with chemical shifts computed on every 10th frame
    bme-rg gyrate.xvg --target 1.85 --sigma 0.05 --stride 10 \\
           --data cs_exp.dat cs_calc.dat --scan

The weights are written to <out>.weights.dat (columns: time, weight), a
format that can be read back with numpy.loadtxt and used as initial weights.
"""
import argparse
import os
import sys

import numpy as np

import BME
import BME_tools as bt


def summary(rg, w, target, sigma):
    """Weighted average and standard deviation of rg, and deviation from target in sigma units."""
    avg = float(np.dot(w, rg))
    std = float(np.sqrt(np.dot(w, (rg - avg) ** 2)))
    return avg, std, (avg - target) / sigma


def plot(rg, w0, w, target, sigma, units, outfile):
    plt = BME._pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(rg.min(), rg.max(), 60)
    ax.hist(rg, bins=bins, weights=w0, density=True, histtype="step", color="0.4", lw=1.5, label="prior")
    ax.hist(rg, bins=bins, weights=w, density=True, histtype="step", color="C3", lw=1.5, label="reweighted")
    ax.axvspan(target - sigma, target + sigma, color="C0", alpha=0.2, label="target $\\pm\\sigma$")
    ax.axvline(target, color="C0")
    ax.set_xlabel("Rg (%s)" % units)
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        prog="bme-rg",
        description="BME reweighting of a simulation towards a target radius of gyration "
                    "read from a GROMACS .xvg file (gmx gyrate), optionally together with "
                    "other data in BME format (e.g. chemical shifts).")
    p.add_argument("xvg", help="GROMACS .xvg file, e.g. the output of gmx gyrate")
    p.add_argument("--target", type=float, required=True, help="target (experimental) value")
    p.add_argument("--sigma", type=float, required=True, help="uncertainty of the target value")
    p.add_argument("--target-units", choices=["nm", "A"], default="nm",
                   help="units of --target and --sigma; the .xvg file is assumed to be in nm "
                        "as written by gmx gyrate (default: nm)")
    p.add_argument("--theta", type=float, default=1.0,
                   help="regularization parameter: large theta = stay close to the simulation, "
                        "small theta = fit the data more closely (default: 1)")
    p.add_argument("--scan", action="store_true",
                   help="print a table of results for a range of theta values to help choosing it")
    p.add_argument("--column", type=int, default=1,
                   help="data column of the .xvg file (default 1 = total Rg for gmx gyrate)")
    p.add_argument("--bound", choices=["upper", "lower"], default=None,
                   help="treat the target as an upper or lower bound instead of an average")
    p.add_argument("--begin", type=float, default=None, help="first time to use (xvg time units, ps)")
    p.add_argument("--end", type=float, default=None, help="last time to use (xvg time units, ps)")
    p.add_argument("--stride", type=int, default=1,
                   help="use one xvg frame every STRIDE (after --begin/--end), to match data "
                        "computed on a subsampled trajectory (default: 1)")
    p.add_argument("--data", nargs=2, action="append", default=[], metavar=("EXP", "CALC"),
                   help="additional data in BME format (experimental file, calculated file), "
                        "e.g. chemical shifts; the rows of CALC must be the same frames as the "
                        "selected xvg frames. Can be repeated.")
    p.add_argument("--rg-weight", type=float, default=1.0,
                   help="relative importance of the Rg target with respect to each point of --data: "
                        "the Rg counts as RG_WEIGHT data points (default: 1)")
    p.add_argument("--w0", default=None,
                   help="file with initial weights (last column), one row per frame used")
    p.add_argument("--out", default=None, help="prefix of the output files (default: name of the xvg file)")
    p.add_argument("--plot", action="store_true", help="write <out>.png with the Rg distributions")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    target, sigma = args.target, args.sigma
    if args.target_units == "A":
        target, sigma = target / 10.0, sigma / 10.0
    units = "nm"
    out = args.out if args.out else os.path.splitext(os.path.basename(args.xvg))[0]
    bound = args.bound.upper() if args.bound else None
    what = {None: "average", "UPPER": "upper bound", "LOWER": "lower bound"}[bound]

    # load everything once; the fits for different theta reuse the same data
    datasets = []   # (name, slice of the data columns)
    try:
        w0 = np.loadtxt(args.w0, ndmin=2)[:, -1] if args.w0 else ()
        rew = BME.Reweight(out, w0=w0)
        time = rew.load_xvg(args.xvg, target, sigma, column=args.column, bound=bound,
                            begin=args.begin, end=args.end, stride=args.stride, weight=args.rg_weight)
        datasets.append(("Rg", slice(0, 1)))
        for exp_file, calc_file in args.data:
            n0 = rew.get_ndata()
            rew.load(exp_file, calc_file)
            datasets.append((os.path.basename(exp_file), slice(n0, rew.get_ndata())))
    except (ValueError, OSError) as e:
        print("Error: %s" % e, file=sys.stderr)
        return 1

    exp = rew.get_experiment()
    rg = rew.get_calculated(columns=[0])[:, 0]   # the Rg column (loaded first)
    w0 = rew.get_w0()
    n = len(rg)
    avg0, std0, z0 = summary(rg, w0, target, sigma)

    def chi2_per_dataset(w):
        avg = rew.calc_averages(w)
        return [bt.calc_chi_avg(exp[sl], avg[sl]) for _, sl in datasets]

    print("BME reweighting of %s (column %d)" % (args.xvg, args.column))
    print("  frames             %d (time %g - %g, stride %d)" % (n, time[0], time[-1], args.stride))
    print("  target (%s)   %.4f +/- %.4f %s%s" % (what, target, sigma, units,
          "" if args.rg_weight == 1 else "   (weight %g)" % args.rg_weight))
    print("  sampled range      %.4f - %.4f %s" % (rg.min(), rg.max(), units))
    for name, sl in datasets[1:]:
        print("  additional data    %s: %d data points" % (name, sl.stop - sl.start))
    if not (rg.min() <= target <= rg.max()):
        print("  WARNING: the target is outside the range sampled by the simulation; "
              "reweighting cannot reach it.")

    extra = datasets[1:]
    if args.scan:
        thetas = np.geomspace(0.01, 1000, 11)
        head = "  %10s %10s %10s %12s %10s %8s %10s" % ("theta", "<Rg>", "std", "(<Rg>-t)/s", "chi2", "phi", "N_eff")
        head += "".join(" %14s" % ("chi2 " + name)[:14] for name, _ in extra)
        print()
        print(head)
        c0 = chi2_per_dataset(w0)
        print("  %10s %10.4f %10.4f %12.2f %10.4f %8.3f %10d" % ("-", avg0, std0, z0, c0[0], 1.0, n)
              + "".join(" %14.4f" % c for c in c0[1:]) + "   (prior)")
        for theta in thetas:
            c_before, c_after, ph = rew.fit(theta=theta)
            if not np.isfinite(c_after):
                print("  %10.3g   minimization failed" % theta)
                continue
            w = rew.get_weights()
            avg, std, z = summary(rg, w, target, sigma)
            c1 = chi2_per_dataset(w)
            print("  %10.3g %10.4f %10.4f %12.2f %10.4f %8.3f %10.0f" % (theta, avg, std, z, c1[0], ph, ph * n)
                  + "".join(" %14.4f" % c for c in c1[1:]))
        print("  Choose the largest theta for which <Rg> (and the other data) are acceptably close to the "
              "experiment (e.g. |<Rg>-t|/s < 1), keeping phi (fraction of effective frames) as large as possible.")
        print()

    chi2_before, chi2_after, phi = rew.fit(theta=args.theta)
    if not np.isfinite(chi2_after):
        print("Error: the minimization failed for theta=%g (see %s.log)" % (args.theta, out), file=sys.stderr)
        rew.close()
        return 1

    w = rew.get_weights()
    avg1, std1, z1 = summary(rg, w, target, sigma)
    c0, c1 = chi2_per_dataset(w0), chi2_per_dataset(w)
    print("  theta              %g" % args.theta)
    print("  <Rg> prior         %.4f +/- %.4f %s   ((<Rg>-target)/sigma = %+.2f, chi2 = %.4f)" % (avg0, std0, units, z0, c0[0]))
    print("  <Rg> reweighted    %.4f +/- %.4f %s   ((<Rg>-target)/sigma = %+.2f, chi2 = %.4f)" % (avg1, std1, units, z1, c1[0]))
    for (name, _), a, b in zip(extra, c0[1:], c1[1:]):
        print("  chi2 %-13s %.4f -> %.4f" % (name, a, b))
    if extra:
        print("  chi2 all data      %.4f -> %.4f" % (chi2_before, chi2_after))
    print("  fraction of effective frames phi = %.3f (N_eff = %.0f of %d)" % (phi, phi * n, n))

    wfile = out + ".weights.dat"
    np.savetxt(wfile, np.column_stack([time, w]), fmt=["%.6g", "%.8e"],
               header="time weight   (BME, target %g +/- %g %s, theta %g)" % (target, sigma, units, args.theta))
    print("  weights written to %s, log to %s.log" % (wfile, out))
    if args.plot:
        plot(rg, w0, w, target, sigma, units, out + ".png")
        print("  plot written to %s.png" % out)
    rew.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
