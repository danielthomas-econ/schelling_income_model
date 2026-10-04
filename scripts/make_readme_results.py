"""Regenerates every figure and number in README.md.

    python scripts/make_readme_results.py            # run everything (~40 min on 4 cores), then update README.md
    python scripts/make_readme_results.py --workers 4

The simulation results are cached in results/readme_runs.pkl, so re-running only redraws the figures and tables.
Delete that file (or pass --rerun) after changing the model. The tables in README.md between the
<!-- auto:... --> markers are rewritten; the prose around them is not, so re-read it after a rerun.
"""
import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ------------------------------------------------- run settings -------------------------------------------------- #
N_AGENTS = 100_000          # results are close to population invariant, and this keeps a full rerun under an hour
N_ROUNDS = 100
N_RUNS = 20                 # paired Monte Carlo runs (baseline and policy start from the same city in run r)
N_RUNS_ROBUSTNESS = 10      # runs for the alternative reservation rents
SEED = 0
PREFERENCES = ["status", "homophily"]
MAIN_R = 0.05               # RESERVATION_RENT_SHARE used in the model (src/common.py)
ROBUSTNESS_R = [0.02, 0.10]
GIF_ROUNDS = 40             # sorting has settled well before this

CACHE = os.path.join(ROOT, "results", "readme_runs.pkl")
FIGURES = os.path.join(ROOT, "figures")
README = os.path.join(ROOT, "README.md")

# colors: one per preference everywhere, validated for colorblind separation (dataviz palette slots 1 and 2)
COLOR = {"status": "#2a78d6", "homophily": "#eb6834"}
LABEL = {"status": "Status (want neighbors at or above own bracket)",
         "homophily": "Homophily (want neighbors within ±1 bracket)"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BRACKET_LABEL = "Income bracket (0-8 = deciles, 9 = 90-95th pct, 10 = 95-99th, 11 = top 1%)"


# --------------------------------------------- one paired policy run ---------------------------------------------- #
def run_job(preference, r_share, n_runs):
    # runs in its own process, so overriding the module constant only affects this job
    import src.houses as houses
    from src.policy import evaluate_policy
    houses.RESERVATION_RENT_SHARE = r_share

    with contextlib.redirect_stdout(io.StringIO()):
        mc_b, mc_p, _, w = evaluate_policy(n_agents = N_AGENTS, max_rounds = N_ROUNDS, n_runs = n_runs, seed = SEED,
                                           plot = False, preference = preference)
    runs = np.arange(n_runs)
    final = {}
    for tag, mc in (("base", mc_b), ("policy", mc_p)):
        for k in ["nonmarket_housing", "vacancies", "avg_value", "theil_within", "theil_between", "churn"]:
            final[f"{k}_{tag}"] = mc[k][runs, mc["last_round"]].astype(np.float64)
    keep = ["ev_median_all", "ev_all", "gain_all", "lose_all", "ev_median", "gain_pct", "lose_pct",
            "housed_base", "housed_policy", "rent_share_base", "rent_share_policy",
            "group_share", "group_gain", "group_lose", "group_ev_median"]
    return {"preference": preference, "r_share": r_share, "n_runs": n_runs,
            "welfare": {k: np.asarray(w[k]) for k in keep}, "final": final,
            "theil_between_base": np.nanmean(mc_b["theil_between"], axis = 0)}


def run_all(workers):
    jobs = [(p, MAIN_R, N_RUNS) for p in PREFERENCES] + [(p, r, N_RUNS_ROBUSTNESS) for p in PREFERENCES
                                                         for r in ROBUSTNESS_R]
    start = time.time()
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futures = {job: pool.submit(run_job, *job) for job in jobs}
        results = {}
        for job, fut in futures.items():
            results[job[:2]] = fut.result()
            print(f"  done: {job[0]}, R = {job[1]:.0%} ({time.time() - start:.0f}s)")
    return results


# ------------------------------------------------- summaries ---------------------------------------------------- #
def mean_ci(x):
    from scipy.stats import t
    x = np.asarray(x, dtype = np.float64)
    x = x[np.isfinite(x)]
    half = t.ppf(0.975, x.size - 1) * x.std(ddof = 1) / np.sqrt(x.size)
    return x.mean(), x.mean() - half, x.mean() + half


def summarize(res):
    w, f = res["welfare"], res["final"]
    reserved_occupied = np.nanmean(w["group_share"][:, 0]) # % of agents in a reserved home = % of the stock
    reserved_empty = 20.0 - reserved_occupied # houses_eligible = 0.2 of every neighborhood's homes
    between = f["theil_between_base"] / (f["theil_between_base"] + f["theil_within_base"])
    return {
        "between_share": 100 * between.mean(),
        "nonmarket_base": f["nonmarket_housing_base"].mean(),
        "nonmarket_change": (f["nonmarket_housing_policy"] - f["nonmarket_housing_base"]).mean(),
        "rent_base": f["avg_value_base"].mean(),
        "rent_change_pct": 100 * (f["avg_value_policy"].mean() / f["avg_value_base"].mean() - 1),
        "ev_median": mean_ci(w["ev_median_all"]),
        "worse_off": np.nanmean(w["lose_all"]),
        "better_off": np.nanmean(w["gain_all"]),
        "eligible_worse_off": np.nanmean(w["lose_pct"][:, :3]), # brackets 0-2 are eligible (income_cutoff = 2)
        "reserved_occupied": reserved_occupied,
        "reserved_empty": reserved_empty,
        "market_empty_policy": f["vacancies_policy"].mean() - reserved_empty,
        "winners_ev": np.nanmean(w["group_ev_median"][:, 0]),
    }


# --------------------------------------------------- figures ---------------------------------------------------- #
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.edgecolor": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def fig_policy(results, path):
    plt = style()
    fig, (ax_ev, ax_stock) = plt.subplots(1, 2, figsize = (13, 4.8), gridspec_kw = {"width_ratios": [1.15, 1]})
    brackets = np.arange(12)
    width = 0.4

    # left: welfare effect on the median agent in each bracket
    for i, p in enumerate(PREFERENCES):
        ev = results[(p, MAIN_R)]["welfare"]["ev_median"]
        ax_ev.bar(brackets + (i - 0.5) * width * 1.04, np.nanmean(ev, axis = 0), width, color = COLOR[p],
                  label = LABEL[p])
    ax_ev.axhline(0, color = INK_2, lw = 0.8)
    ax_ev.axvline(2.5, color = INK_2, ls = "--", lw = 1)
    ax_ev.text(1.0, ax_ev.get_ylim()[0] * 0.97, "eligible for\nreserved homes", ha = "center", va = "bottom",
               color = INK_2, fontsize = 9)
    ax_ev.set_xticks(brackets)
    ax_ev.set_xlabel(BRACKET_LABEL)
    ax_ev.set_ylabel("Median welfare effect (% of income)")
    ax_ev.set_title("Welfare effect on the median household", loc = "left", fontweight = "bold", pad = 44)
    ax_ev.grid(axis = "y", color = GRID, lw = 0.8)
    ax_ev.set_axisbelow(True)
    ax_ev.legend(frameon = False, fontsize = 9, ncol = 1, loc = "lower left", bbox_to_anchor = (0, 1.0),
                 borderaxespad = 0.2)

    # right: what happens to the housing stock
    segments = [("Rented at market rent", "#d6d5cf"), ("Reserved, occupied", "#1baf7a"),
                ("Market, empty", "#eda100"), ("Reserved, empty", "#4a3aa7")]
    rows = []
    for p in PREFERENCES:
        s = summarize(results[(p, MAIN_R)])
        f = results[(p, MAIN_R)]["final"]
        base_empty = f["vacancies_base"].mean()
        rows.append((f"{p.capitalize()}\nbaseline", [100 - base_empty, 0, base_empty, 0]))
        rows.append((f"{p.capitalize()}\nwith set-aside",
                     [100 - s["reserved_occupied"] - s["reserved_empty"] - s["market_empty_policy"],
                      s["reserved_occupied"], s["market_empty_policy"], s["reserved_empty"]]))
    y = np.arange(len(rows))[::-1]
    for j, (name, color) in enumerate(segments):
        left = np.array([sum(r[1][:j]) for r in rows])
        vals = np.array([r[1][j] for r in rows])
        ax_stock.barh(y, vals, left = left, color = color, edgecolor = SURFACE, linewidth = 2, label = name,
                      height = 0.62)
        for yy, l, v in zip(y, left, vals):
            if v >= 5: # direct labels: these colors sit below 3:1 contrast, so values are written on them
                ax_stock.text(l + v / 2, yy, f"{v:.0f}%", ha = "center", va = "center", fontsize = 9,
                              color = "#ffffff" if color == "#4a3aa7" else INK)
    ax_stock.set_yticks(y)
    ax_stock.set_yticklabels([r[0] for r in rows], fontsize = 9)
    ax_stock.set_xlim(0, 100)
    ax_stock.set_xlabel("% of all homes (homes = households)")
    ax_stock.set_title("Where the homes go", loc = "left", fontweight = "bold", pad = 44)
    ax_stock.legend(frameon = False, fontsize = 9, ncol = 2, loc = "upper center", bbox_to_anchor = (0.5, -0.16))
    ax_stock.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi = 130)
    plt.close(fig)


def fig_sorting(results, path):
    plt = style()
    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize = (13, 4.4))
    for p in PREFERENCES:
        series = results[(p, MAIN_R)]["theil_between_base"]
        ax_t.plot(np.arange(1, series.size), series[1:], color = COLOR[p], lw = 2, label = LABEL[p])
    ax_t.set_xlabel("Round")
    ax_t.set_ylabel("Between-neighborhood Theil index")
    ax_t.set_title("Income sorting emerges within ~30 rounds", loc = "left", fontweight = "bold")
    ax_t.grid(color = GRID, lw = 0.8)
    ax_t.set_axisbelow(True)
    ax_t.legend(frameon = False, fontsize = 9, loc = "lower right")

    brackets = np.arange(12)
    width = 0.4
    for i, p in enumerate(PREFERENCES):
        housed = np.nanmean(results[(p, MAIN_R)]["welfare"]["housed_base"], axis = 0)
        ax_h.bar(brackets + (i - 0.5) * width * 1.04, 100 - housed, width, color = COLOR[p], label = LABEL[p])
    ax_h.set_xticks(brackets)
    ax_h.set_xlabel(BRACKET_LABEL)
    ax_h.set_ylabel("Priced out of market housing (%)")
    ax_h.set_title("Who gets priced out depends on preferences", loc = "left", fontweight = "bold")
    ax_h.grid(axis = "y", color = GRID, lw = 0.8)
    ax_h.set_axisbelow(True)
    ax_h.legend(frameon = False, fontsize = 9, loc = "upper right")

    fig.tight_layout()
    fig.savefig(path, dpi = 130)
    plt.close(fig)


def make_gif(path):
    style()
    from src.sim import sim_one_round
    from src.plots import create_segregation_animation
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, stats, last_round = sim_one_round(n_agents = N_AGENTS, max_rounds = GIF_ROUNDS, seed = SEED, plot = False)
        create_segregation_animation(stats, last_round, save_path = path)


# ---------------------------------------------- README tables -------------------------------------------------- #
def headline_table(results):
    s = {p: summarize(results[(p, MAIN_R)]) for p in PREFERENCES}
    def row(label, fmt):
        return f"| {label} | " + " | ".join(fmt(s[p]) for p in PREFERENCES) + " |"
    lines = [
        "| | Status preferences | Homophily preferences |",
        "|---|---|---|",
        row("Welfare effect on the median household", lambda x: "**{:+.1f}%** of income (95% CI {:+.1f} to {:+.1f})".format(*x["ev_median"])),
        row("Households worse off / better off", lambda x: f"{x['worse_off']:.0f}% / {x['better_off']:.0f}%"),
        row("Eligible households worse off", lambda x: f"{x['eligible_worse_off']:.0f}%"),
        row("Reserved homes left empty", lambda x: f"{x['reserved_empty']:.0f}% of all homes"),
        row("Average market rent", lambda x: f"{x['rent_change_pct']:+.0f}%"),
        row("Households priced out of market housing", lambda x: f"{x['nonmarket_base']:.1f}% → {x['nonmarket_base'] + x['nonmarket_change']:.1f}%"),
    ]
    return "\n".join(lines)


def robustness_table(results):
    shares = sorted([MAIN_R] + ROBUSTNESS_R)
    lines = ["| Reservation rent (share of median income) | " + " | ".join(f"{r:.0%}" for r in shares) + " |",
             "|---|" + "---|" * len(shares)]
    for p in PREFERENCES:
        cells = []
        for r in shares:
            s = summarize(results[(p, r)])
            cells.append(f"{s['ev_median'][0]:+.1f}% ({s['worse_off']:.0f}% worse off)")
        lines.append(f"| {p.capitalize()}: median welfare effect | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_block(text, name, content):
    start, end = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + "\n" + content + "\n" + text[j:]


# ------------------------------------------------------ main ---------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--rerun", action = "store_true", help = "ignore the cached results")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(CACHE), exist_ok = True)
    os.makedirs(FIGURES, exist_ok = True)
    if os.path.exists(CACHE) and not args.rerun:
        results = pickle.load(open(CACHE, "rb"))
        print(f"using cached results from {CACHE}")
    else:
        print("running the paired baseline/policy simulations...")
        results = run_all(args.workers)
        pickle.dump(results, open(CACHE, "wb"))

    os.chdir(ROOT) # create_segregation_animation writes temporary frames to the working directory
    fig_policy(results, os.path.join(FIGURES, "policy.png"))
    fig_sorting(results, os.path.join(FIGURES, "sorting.png"))
    make_gif(os.path.join(FIGURES, "segregation.gif"))

    text = open(README).read()
    text = replace_block(text, "headline", headline_table(results))
    text = replace_block(text, "robustness", robustness_table(results))
    open(README, "w").write(text)
    print("figures written to figures/, tables updated in README.md")
