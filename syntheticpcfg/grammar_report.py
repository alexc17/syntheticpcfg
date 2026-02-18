"""
Generate a PDF report summarising the statistical properties of a PCFG.

Usage:
    python -m syntheticpcfg.grammar_report grammar.pcfg report.pdf [options]
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import math
import logging
from collections import Counter, defaultdict

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import pcfg
from . import inside
from . import utility
from .uniformsampler import UniformSampler


def compute_length_distribution(mypcfg, max_length):
    upcfg = mypcfg.make_unary()
    insider = inside.UnaryInside(upcfg)
    table = insider.compute_inside_smart(max_length)
    start = insider.start
    lengths = np.arange(1, max_length)
    probs = np.array([table[l, start] for l in range(1, max_length)])
    return lengths, probs


def compute_rank_frequency(mypcfg):
    try:
        te = mypcfg.terminal_expectations()
    except (np.linalg.LinAlgError, ValueError):
        te = defaultdict(float)
        for prod in mypcfg.productions:
            if len(prod) == 2:
                te[prod[1]] += mypcfg.parameters[prod]
    counts = np.array(list(te.values()))
    total = np.sum(counts)
    if total == 0:
        return np.array([]), np.array([])
    xindices = np.argsort(-counts)
    frequencies = counts[xindices] / total
    ranks = np.arange(1, len(te) + 1)
    return ranks, frequencies


def compute_nonterminal_table(mypcfg):
    nte = mypcfg.nonterminal_expectations()
    ece = mypcfg.entropy_conditional_nonterminals()
    slp = mypcfg.sum_lexical_probs()
    rows = []
    for nt in sorted(mypcfg.nonterminals):
        n_binary = sum(1 for p in mypcfg.productions if p[0] == nt and len(p) == 3)
        n_lexical = sum(1 for p in mypcfg.productions if p[0] == nt and len(p) == 2)
        rows.append((nt, nte.get(nt, 0), ece.get(nt, 0), slp.get(nt, 0), n_binary, n_lexical))
    return rows


def robust_sample_trees(sampler, n_samples, max_attempts_factor=3):
    """Sample trees, silently skipping depth-exceeded failures."""
    trees = []
    attempts = 0
    failures = 0
    max_attempts = n_samples * max_attempts_factor
    while len(trees) < n_samples and attempts < max_attempts:
        attempts += 1
        try:
            trees.append(sampler.sample_tree())
        except ValueError:
            failures += 1
    if len(trees) < n_samples:
        logging.warning("Only obtained %d/%d samples (%d attempts, %d depth failures)",
                        len(trees), n_samples, attempts, failures)
    return trees



def robust_estimate_ambiguity(mypcfg, insider_obj, sampler, n_samples, max_length):
    trees = robust_sample_trees(sampler, n_samples)
    total = 0.0
    n = 0
    for tree in trees:
        s = utility.collect_yield(tree)
        if len(s) > max_length:
            continue
        try:
            lp = insider_obj.inside_log_probability(s)
            lpd = mypcfg.log_probability_derivation(tree)
            total += lp - lpd
            n += 1
        except utility.ParseFailureException:
            pass
    return total / n if n > 0 else float('nan')


def robust_monte_carlo_entropy(mypcfg, insider_obj, sampler, n_samples):
    trees = robust_sample_trees(sampler, n_samples)
    string_e = 0.0
    unlabeled_e = 0.0
    derivation_e = 0.0
    n = 0
    for tree in trees:
        try:
            lp_derivation = mypcfg.log_probability_derivation(tree)
            sentence = utility.collect_yield(tree)
            lp_unlabeled = insider_obj.inside_bracketed_log_probability(tree)
            lp_string = insider_obj.inside_log_probability(sentence)
            string_e -= lp_string
            unlabeled_e -= lp_unlabeled
            derivation_e -= lp_derivation
            n += 1
        except utility.ParseFailureException:
            pass
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    return string_e / n, unlabeled_e / n, derivation_e / n


def compute_density_data(mypcfg, max_length, prng, density_samples):
    """
    Use UniformSampler to compute:
    - ambiguity ratio: derivations / |V|^n at each length (exact)
    - string density: proportion of strings of length n in the language (MC estimate)
    """
    us = UniformSampler(mypcfg, max_length, prng)
    lengths = []
    ambiguity = []
    string_density = []
    for l in range(1, max_length):
        lengths.append(l)
        ambiguity.append(us.density(l))
        if us.get_total(l) > 0:
            try:
                sd = us.string_density(l, density_samples)
            except ValueError:
                sd = 0.0
        else:
            sd = 0.0
        string_density.append(sd)
    return np.array(lengths), np.array(ambiguity), np.array(string_density)


def make_report(grammar_file, output_file, args):
    mypcfg = pcfg.load_pcfg_from_file(grammar_file)
    prng = RandomState(args.seed) if args.seed else RandomState()
    sampler = pcfg.Sampler(mypcfg, random=prng)
    insider_obj = inside.InsideComputation(mypcfg)

    n_binary = sum(1 for p in mypcfg.productions if len(p) == 3)
    n_lexical = sum(1 for p in mypcfg.productions if len(p) == 2)

    nan = float('nan')

    try:
        el = mypcfg.expected_length()
    except (np.linalg.LinAlgError, ValueError):
        logging.warning("Could not compute expected length (singular matrix).")
        el = nan

    try:
        de = mypcfg.derivational_entropy()
    except (np.linalg.LinAlgError, ValueError):
        de = nan

    try:
        de_binary, de_lexical = mypcfg.derivational_entropy_split()
    except (np.linalg.LinAlgError, ValueError):
        de_binary, de_lexical = nan, nan

    try:
        eu = mypcfg.entropy_unigram()
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        eu = nan

    try:
        ep = mypcfg.entropy_preterminal()
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        ep = nan

    try:
        la = mypcfg.compute_lexical_ambiguity()
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        la = nan

    logging.info("Basic statistics computed. Sampling for MC estimates...")

    ambiguity = robust_estimate_ambiguity(
        mypcfg, insider_obj, sampler, args.mc_samples, args.max_length)

    se, ute, lte = robust_monte_carlo_entropy(
        mypcfg, insider_obj, sampler, args.mc_samples)

    density_lengths, ambiguity_values, string_density_values = compute_density_data(
        mypcfg, args.max_length, prng, args.density_samples)

    lengths, length_probs = compute_length_distribution(mypcfg, args.max_length)

    try:
        ranks, frequencies = compute_rank_frequency(mypcfg)
    except (np.linalg.LinAlgError, ValueError):
        ranks, frequencies = np.array([]), np.array([])

    try:
        nt_rows = compute_nonterminal_table(mypcfg)
    except (np.linalg.LinAlgError, ValueError):
        nt_rows = [(nt, nan, nan, nan,
                     sum(1 for p in mypcfg.productions if p[0] == nt and len(p) == 3),
                     sum(1 for p in mypcfg.productions if p[0] == nt and len(p) == 2))
                    for nt in sorted(mypcfg.nonterminals)]

    try:
        pf = mypcfg.compute_partition_function_fast()
    except (ValueError, np.linalg.LinAlgError):
        pf = {nt: nan for nt in mypcfg.nonterminals}

    fmt = lambda v: "N/A" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
    fmt6 = lambda v: "N/A" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}"

    with PdfPages(output_file) as pdf:

        # ===== Page 1: Text and tables =====
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        title = grammar_file.split('/')[-1]
        ax.text(0.5, 0.97, f"Grammar Report: {title}",
                ha='center', va='top', fontsize=14, fontweight='bold')

        # -- Summary statistics (left column) --
        left_stats = [
            ("Grammar size", ""),
            ("  Nonterminals", str(len(mypcfg.nonterminals))),
            ("  Terminals", str(len(mypcfg.terminals))),
            ("  Binary productions", str(n_binary)),
            ("  Lexical productions", str(n_lexical)),
            ("", ""),
            ("  Expected length", fmt(el)),
            ("  Partition fn Z(S)", fmt6(pf.get(mypcfg.start, nan))),
        ]
        right_stats = [
            ("Entropy (nats)", ""),
            ("  Derivational", fmt(de)),
            ("    Binary component", fmt(de_binary)),
            ("    Lexical component", fmt(de_lexical)),
            ("  Unigram", fmt(eu)),
            ("  Preterminal", fmt(ep)),
            ("", ""),
            ("Ambiguity", ""),
            ("  H(NT|terminal)", fmt(la)),
            (f"  H(tree|string) MC", fmt(ambiguity)),
            ("", ""),
            (f"MC estimates (n={args.mc_samples})", ""),
            ("  String entropy", fmt(se)),
            ("  Unlabeled tree H", fmt(ute)),
            ("  Derivation H", fmt(lte)),
        ]

        y_left = 0.92
        for label, value in left_stats:
            if label == "" and value == "":
                y_left -= 0.006
                continue
            weight = 'bold' if value == "" else 'normal'
            ax.text(0.04, y_left, label, ha='left', va='top', fontsize=8,
                    fontweight=weight, family='monospace')
            if value:
                ax.text(0.36, y_left, value, ha='left', va='top', fontsize=8,
                        family='monospace')
            y_left -= 0.018

        y_right = 0.92
        for label, value in right_stats:
            if label == "" and value == "":
                y_right -= 0.006
                continue
            weight = 'bold' if value == "" else 'normal'
            ax.text(0.52, y_right, label, ha='left', va='top', fontsize=8,
                    fontweight=weight, family='monospace')
            if value:
                ax.text(0.82, y_right, value, ha='left', va='top', fontsize=8,
                        family='monospace')
            y_right -= 0.018

        # -- Nonterminal table --
        table_top = min(y_left, y_right) - 0.03
        ax.plot([0.04, 0.96], [table_top + 0.005, table_top + 0.005],
                'k-', linewidth=0.5)
        ax.text(0.5, table_top + 0.02, "Nonterminal Statistics",
                ha='center', va='top', fontsize=10, fontweight='bold')

        col_headers = ["NT", "E[count]", "H(prod|NT)", "P(lex)",
                        "#bin", "#lex", "Z(NT)"]
        col_x = [0.04, 0.15, 0.28, 0.43, 0.56, 0.66, 0.78]
        y = table_top - 0.005
        for cx, h in zip(col_x, col_headers):
            ax.text(cx, y, h, ha='left', va='top', fontsize=7.5,
                    fontweight='bold', family='monospace')
        y -= 0.004
        ax.plot([0.04, 0.96], [y, y], 'k-', linewidth=0.3)
        y -= 0.013

        for nt, exp, ent, plx, nb, nl in nt_rows:
            if y < 0.03:
                ax.text(0.04, y, "...", fontsize=7, family='monospace')
                break
            znt = fmt6(pf.get(nt, nan))
            vals = [nt, f"{exp:.3f}", f"{ent:.3f}", f"{plx:.3f}",
                    str(nb), str(nl), znt]
            for cx, v in zip(col_x, vals):
                ax.text(cx, y, v, ha='left', va='top', fontsize=7,
                        family='monospace')
            y -= 0.015

        pdf.savefig(fig)
        plt.close(fig)

        # ===== Page 2: All graphs (2x3 grid) =====
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

        # (0,0) Rank-frequency (Zipf) with reference slope
        ax = axes[0, 0]
        if len(ranks) > 0:
            ax.loglog(ranks, frequencies, 'b-', linewidth=0.8)
            ref_ranks = np.array([ranks[0], ranks[-1]], dtype=float)
            ref_freq = frequencies[0] * (ref_ranks / ref_ranks[0]) ** (-1)
            ax.loglog(ref_ranks, ref_freq, 'r--', alpha=0.5, label='slope=-1')
            ax.legend(fontsize=7)
        ax.set_xlabel('Rank', fontsize=8)
        ax.set_ylabel('Relative frequency', fontsize=8)
        ax.set_title('Rank-frequency (Zipf)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, which='both')

        # (0,1) Exact length distribution
        ax = axes[0, 1]
        ax.bar(lengths, length_probs, color='steelblue', edgecolor='none', width=0.8)
        if not math.isnan(el):
            ax.axvline(el, color='red', linestyle='--', linewidth=1,
                       label=f'E[L]={el:.1f}')
            ax.legend(fontsize=7)
        ax.set_xlabel('Length', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_title('Length distribution (exact)', fontsize=9)
        ax.tick_params(labelsize=7)

        # (0,2) Global ambiguity (derivations / |V|^n)
        ax = axes[0, 2]
        if len(ambiguity_values) > 0:
            positive = ambiguity_values > 0
            if positive.any():
                ax.semilogy(density_lengths[positive], ambiguity_values[positive],
                            'b-', linewidth=0.8)
                ax.axhline(1.0, color='red', linestyle='--', linewidth=0.7,
                           alpha=0.5, label='ratio=1')
                ax.legend(fontsize=7)
        ax.set_xlabel('String length', fontsize=8)
        ax.set_ylabel('Derivations / |V|^n', fontsize=8)
        ax.set_title('Global ambiguity', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # (1,0) String density (proportion of strings in the language)
        ax = axes[1, 0]
        if len(string_density_values) > 0:
            positive = string_density_values > 0
            if positive.any():
                ax.semilogy(density_lengths[positive],
                            string_density_values[positive],
                            'b-', linewidth=0.8)
        ax.set_xlabel('String length', fontsize=8)
        ax.set_ylabel('P(w parseable)', fontsize=8)
        ax.set_title(f'String density (n={args.density_samples}/len)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # (1,1) Per-terminal lexical ambiguity
        ax = axes[1, 1]
        try:
            pwe = mypcfg.per_word_entropies()
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            pwe = {}
        if pwe:
            entropies = sorted(pwe.values(), reverse=True)
            ax.bar(range(len(entropies)), entropies, color='steelblue',
                   edgecolor='none', width=1.0)
            ax.set_xlim(-0.5, min(len(entropies), 200) - 0.5)
        ax.set_xlabel('Terminal rank', fontsize=8)
        ax.set_ylabel('H(NT|terminal) (nats)', fontsize=8)
        ax.set_title('Lexical ambiguity', fontsize=9)
        ax.tick_params(labelsize=7)

        # (1,2) Production weight distributions (both on one plot)
        ax = axes[1, 2]
        binary_params = sorted(
            [mypcfg.parameters[p] for p in mypcfg.productions if len(p) == 3],
            reverse=True)
        lexical_params = sorted(
            [mypcfg.parameters[p] for p in mypcfg.productions if len(p) == 2],
            reverse=True)
        if binary_params:
            ax.loglog(range(1, len(binary_params) + 1), binary_params,
                      'b-', linewidth=0.8, label=f'Binary ({len(binary_params)})')
        if lexical_params:
            ax.loglog(range(1, len(lexical_params) + 1), lexical_params,
                      'r-', linewidth=0.8, alpha=0.7,
                      label=f'Lexical ({len(lexical_params)})')
        ax.legend(fontsize=7)
        ax.set_xlabel('Rank', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_title('Production weights', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 1], h_pad=2.5, w_pad=2.0)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a PDF report of a PCFG\'s statistical properties.')
    parser.add_argument("inputfilename", help="PCFG grammar file")
    parser.add_argument("outputfilename", help="Output PDF file",
                        nargs='?', default="grammar_report.pdf")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--mc_samples", type=int, default=1000,
                        help="Number of Monte Carlo samples (default 1000)")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum string length for exact computations (default 40)")
    parser.add_argument("--density_samples", type=int, default=100,
                        help="Samples per length for string density estimate (default 100)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    make_report(args.inputfilename, args.outputfilename, args)
