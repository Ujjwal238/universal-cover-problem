"""Generate the cost-scaling table from scaling.log.

Emits the complete table environment, so main.tex inputs it at top level rather
than inside tabular.  No measured value is typed by hand anywhere.
"""
import os
import re
import sys

HEAD = r"""\begin{table}[t]
\centering
\begin{tabular}{lrr@{\qquad}lrr}
\toprule
\multicolumn{3}{c}{$D,B_3,B_5$ \quad($d=5$)} &
\multicolumn{3}{c}{$D,B_3,B_5,B_7$ \quad($d=8$)}\\
\cmidrule(r){1-3}\cmidrule(l){4-6}
$\tau$ & $\mu$ & boxes & $\tau$ & $\mu$ & boxes\\
\midrule
"""
TAIL = r"""\bottomrule
\end{tabular}
\caption{Boxes examined by the subdivision at a range of targets $\tau$. Every
row is the search of \cref{def:split} pruning at $\tau+10^{-9}$ and writing no
certificate, so the last row is the corroborating run of \cref{sec:computation}
rather than the emitting one. The margin $\mu$ is measured against a numerically
located optimum, \FiveDOptimum{} in the first family and \EightDOptimum{} in the
second, not against the proved bound of \cref{prop:ceiling}; in the first family
the two differ by under $10^{-8}$. Throughput was \ThroughputFive{} boxes per
second and \ThroughputEight{} respectively, on eight cores. Generated from
\texttt{scaling.log}.}
\label{tab:scaling}
\end{table}
"""


def rows(block):
    return [(m.group(1), m.group(2), m.group(3))
            for m in re.finditer(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d,]+)\s", block, re.M)]


def cell(t):
    if t is None:
        return " & & "
    n = f"{int(t[2].replace(',', '')):,}".replace(",", "{,}")
    # targets and margins are printed exactly as the log records them; no rounding
    return f"{t[1].rstrip(chr(48)).rstrip(chr(46)) or t[1]} & {t[0]} & {n}"


def main(log, out):
    txt = open(log).read()
    blocks = re.findall(
        r">>> family \([\d, ]+\), dimension (\d+).*?\n(.*?)(?=\n\n|\n>>>|\Z)", txt, re.S)
    r5, r8 = rows(blocks[0][1]), rows(blocks[1][1])
    body = "".join(f"{cell(r5[i] if i < len(r5) else None)} & "
                   f"{cell(r8[i] if i < len(r8) else None)}\\\\\n"
                   for i in range(max(len(r5), len(r8))))
    open(out, "w").write(HEAD + body + TAIL)
    print(f"  wrote {out}: {max(len(r5), len(r8))} rows from {os.path.basename(log)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../scaling.log",
         sys.argv[2] if len(sys.argv) > 2 else "scaling_table.tex")
