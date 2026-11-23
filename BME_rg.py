"""Reweight a simulation towards a target radius of gyration (or any single
observable) read from a GROMACS .xvg file, e.g. the output of gmx gyrate.

Example:

    gmx gyrate -f traj.xtc -s topol.tpr -o gyrate.xvg
    bme-rg gyrate.xvg --target 1.85 --sigma 0.05 --scan
    bme-rg gyrate.xvg --target 1.85 --sigma 0.05 --theta 2 --plot

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


def fit_one(xvg, target, sigma, theta, name, column=1, bound=None, begin=None, end=None, w0=()):
    rew = BME.Reweight(name, w0=w0)
    time = rew.load_xvg(xvg, target, sigma, column=column, bound=bound, begin=begin, end=end)
    chi2_before, chi2_after, phi = rew.fit(theta=theta)
    return rew, time, (chi2_before, chi2_after, phi)


def theta_scan(xvg, target, sigma, thetas, name, **kw):
    rows = []
    for theta in thetas:
        rew, time, (c0, c1, phi) = fit_one(xvg, target, sigma, theta, name, **kw)
        rg = rew.get_calculated()[:, 0]
        w = rew.get_weights() if np.isfinite(c1) else np.full(len(rg), np.nan)
        avg, std, z = summary(rg, w, target, sigma)
        rows.append((theta, avg, std, z, c1, phi))
        rew.close()
    return rows


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
                    "read from a GROMACS .xvg file (gmx gyrate).")
    p.add_argument("xvg", help="GROMACS .xvg file, e.g. the output of gmx gyrate")
    p.add_argument("--target", type=float, required=True, help="target (experimental) value")
    p.add_argument("--sigma", type=float, required=True, help="uncertainty of the target value")
    p.add_argument("--target-units", choices=["nm", "A"], default="nm",
                   help="units of --target and --sigma; the .xvg file is assumed to be in nm "
                        "as written by gmx gyrate (default: nm)")
    p.add_argument("--theta", type=float, default=1.0,
                   help="regularization parameter: large theta = stay close to the simulation, "
                        "small theta = fit the target more closely (default: 1)")
    p.add_argument("--scan", action="store_true",
                   help="print a table of results for a range of theta values to help choosing it")
    p.add_argument("--column", type=int, default=1,
                   help="data column of the .xvg file (default 1 = total Rg for gmx gyrate)")
    p.add_argument("--bound", choices=["upper", "lower"], default=None,
                   help="treat the target as an upper or lower bound instead of an average")
    p.add_argument("--begin", type=float, default=None, help="first time to use (xvg time units, ps)")
    p.add_argument("--end", type=float, default=None, help="last time to use (xvg time units, ps)")
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
    w0 = np.loadtxt(args.w0, ndmin=2)[:, -1] if args.w0 else ()
    kw = dict(column=args.column, bound=bound, begin=args.begin, end=args.end, w0=w0)

    try:
        rew, time, (chi2_before, chi2_after, phi) = fit_one(args.xvg, target, sigma, args.theta, out, **kw)
    except (ValueError, OSError) as e:
        print("Error: %s" % e, file=sys.stderr)
        return 1

    rg = rew.get_calculated()[:, 0]
    w0 = rew.get_w0()
    n = len(rg)
    avg0, std0, z0 = summary(rg, w0, target, sigma)
    what = {None: "average", "UPPER": "upper bound", "LOWER": "lower bound"}[bound]

    print("BME reweighting of %s (column %d)" % (args.xvg, args.column))
    print("  frames             %d (time %g - %g)" % (n, time[0], time[-1]))
    print("  target (%s)   %.4f +/- %.4f %s" % (what, target, sigma, units))
    print("  sampled range      %.4f - %.4f %s" % (rg.min(), rg.max(), units))
    if not (rg.min() <= target <= rg.max()):
        print("  WARNING: the target is outside the range sampled by the simulation; "
              "reweighting cannot reach it.")

    if args.scan:
        thetas = np.geomspace(0.01, 1000, 11)
        rows = theta_scan(args.xvg, target, sigma, thetas, out + "_scan", **kw)
        print()
        print("  %10s %10s %10s %12s %10s %8s %10s" % ("theta", "<Rg>", "std", "(<Rg>-t)/s", "chi2", "phi", "N_eff"))
        print("  %10s %10.4f %10.4f %12.2f %10.4f %8.3f %10d   (prior)" % ("-", avg0, std0, z0, chi2_before, 1.0, n))
        for theta, avg, std, z, c1, ph in rows:
            print("  %10.3g %10.4f %10.4f %12.2f %10.4f %8.3f %10.0f" % (theta, avg, std, z, c1, ph, ph * n))
        print("  Choose the largest theta whose <Rg> is acceptably close to the target "
              "(e.g. |<Rg>-t|/s < 1), keeping phi (fraction of effective frames) as large as possible.")
        print()

    if not np.isfinite(chi2_after):
        print("Error: the minimization failed for theta=%g (see %s.log)" % (args.theta, out), file=sys.stderr)
        return 1

    w = rew.get_weights()
    avg1, std1, z1 = summary(rg, w, target, sigma)
    print("  theta              %g" % args.theta)
    print("  <Rg> prior         %.4f +/- %.4f %s   ((<Rg>-target)/sigma = %+.2f, chi2 = %.4f)" % (avg0, std0, units, z0, chi2_before))
    print("  <Rg> reweighted    %.4f +/- %.4f %s   ((<Rg>-target)/sigma = %+.2f, chi2 = %.4f)" % (avg1, std1, units, z1, chi2_after))
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
