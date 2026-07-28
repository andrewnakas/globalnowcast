"""Compare training runs epoch by epoch, so differences can be read against the noise.

Almost every comparison in this project has been limited by run-to-run variance rather
than by the change being tested: two runs of an identical configuration have peaked
0.0156 apart, which is larger than most of the differences worth chasing. Comparing
peak numbers alone hides that.

    python ml/compare_runs.py /tmp/train_all32_s1.log /tmp/train_all32_s2.log
"""
import argparse
import re
import statistics
import sys
from pathlib import Path


def read_csi(path):
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^epoch\s+(\d+).*CSI (0\.\d+)", line)
        if m:
            out.append(float(m.group(2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    series = {Path(p).stem: read_csi(p) for p in args.logs}
    series = {k: v for k, v in series.items() if v}
    if len(series) < 2:
        print("need at least two logs with epoch lines")
        return 1

    names = list(series)
    n = min(len(v) for v in series.values())
    print(f"{n} matched epochs\n")
    header = f"{'ep':>5}" + "".join(f"{k[-14:]:>16}" for k in names)
    print(header)
    for e in range(args.every, n + 1, args.every):
        row = "".join(f"{series[k][e - 1]:>16.4f}" for k in names)
        print(f"{e:>5}{row}")

    print()
    for k in names:
        v = series[k][:n]
        print(f"  {k[-24:]:>26}: peak {max(v):.4f}  mean {statistics.mean(v):.4f}")

    if len(names) == 2:
        a, b = (series[k][:n] for k in names)
        d = [b[i] - a[i] for i in range(n)]
        print(f"\n  difference: mean {statistics.mean(d):+.4f}  "
              f"sd {statistics.pstdev(d):.4f}  max |d| {max(abs(x) for x in d):.4f}")
        print(f"  peaks differ by {abs(max(a) - max(b)):.4f}")
        print("\nA change is only readable if it exceeds this spread. Runs of the same"
              "\nconfiguration have differed by 0.0156 at peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
