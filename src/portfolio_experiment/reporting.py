"""Regenerate every graph and the PDF from saved CSVs"""
from pathlib import Path
from math import ceil, floor, log10
from xml.sax.saxutils import escape
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

LABELS = {"ppo_selected": "Selected PPO", "ppo_initial": "Initial PPO", "ppo_final": "Final PPO",
          "utility_rolling": "Rolling ESG utility", "utility_static": "Static ESG utility",
          "financial_rolling": "Rolling financial utility", "financial_static": "Static financial utility",
          "equal_weight": "Equal weight"}
COLORS = {"ppo_selected": "#1f77b4", "ppo_initial": "#7f7f7f",
          "utility_rolling": "#ff7f0e", "utility_static": "#2ca02c"}
DISPLAY = {"total_return": (100, 2, "Total return difference (pp)"),
           "sharpe_ratio": (1, 3, "Sharpe ratio difference"),
           "preference_utility": (1, 4, "Realised utility difference"),
           "score_risk": (1, 2, "ESG risk difference (points)")}


def shared_bound(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.abs(finite).max()) if len(finite) else 0
    if maximum == 0:
        return 1.0
    step = 10**floor(log10(maximum))
    return ceil(maximum/step)*step


def make_figures(run_folder):
    """Build the thesis result figures, followed by the per-seed diagnostics.

    All figures read completed CSVs. The same helper draws every heatmap, and
    one colour limit per metric covers all references and evaluation periods.
    """
    run = Path(run_folder)
    protocol = json.loads((run/"protocol.json").read_text())
    cfg = protocol["configuration"]
    results = pd.read_csv(run/"all_metrics.csv")
    pairs = pd.read_csv(run/"paired_comparisons.csv")
    summary = pd.read_csv(run/"summary.csv")
    curves = pd.read_csv(run/"value_curves.csv", parse_dates=["date"])
    records = pd.read_csv(run/"training_record.csv")
    history = pd.read_csv(run/"checkpoint_history.csv")
    roles = [r for r in ("training", "validation", "test") if r in set(results.role)]
    role_names = {"training": "Training", "validation": "Validation", "test": "Test"}
    days = results.groupby("role").return_days.first().to_dict()
    a_values, b_values = cfg["risk_aversions"], cfg["esg_appetites"]
    folder = run/"figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    manifest, exported_cells = [], []

    def save(fig, name, title, values=None, thesis_figure=None):
        # Vector exports and the plotted numbers allow edits without retraining.
        for extension in ("png", "svg", "pdf"):
            fig.savefig(folder/f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=.12)
        item = dict(file=f"figures/{name}.png", title=title)
        if values is not None:
            values.to_csv(folder/f"{name}_values.csv", index=False)
            item["values"] = f"figures/{name}_values.csv"
        if thesis_figure:
            item["thesis_figure"] = thesis_figure
        manifest.append(item)
        plt.close(fig)

    # Average model seeds within each basket, then give baskets equal weight.
    group = ["role", "risk_aversion", "esg_appetite", "reference"]
    columns = ["difference_"+m for m in DISPLAY]
    basket = pairs.groupby(group+["basket_digest"])[columns].mean().reset_index()
    means = basket.groupby(group)[columns].mean().reset_index()
    means.to_csv(run/"paired_summary.csv", index=False)
    counts = pairs.groupby(group)[columns].agg(
        lambda x: int(np.isfinite(x).sum()))
    totals = pairs.groupby(group).size()
    selected = summary[summary.strategy.eq("ppo_selected")]
    comparison_role = "test" if "test" in roles else "validation"
    levels = selected.pivot(index=["risk_aversion", "esg_appetite"],
                            columns="role", values="preference_utility")
    period_change = levels[comparison_role]-levels["training"]
    limits = {m: shared_bound(means["difference_"+m]*factor)
              for m, (factor, _, _) in DISPLAY.items()}
    limits["preference_utility"] = shared_bound(np.r_[
        means.difference_preference_utility.to_numpy(), period_change.to_numpy()])
    absolute_utility_limit = shared_bound(selected.preference_utility)

    def grid(frame, column):
        return frame.pivot(index="risk_aversion", columns="esg_appetite", values=column).reindex(
            index=a_values, columns=b_values).to_numpy(float)

    def draw_heatmap(ax, values, limit, title, precision=4, risk=False, xlabel=True):
        cmap = plt.get_cmap("RdBu_r" if risk else "RdBu").with_extremes(bad="#dddddd")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                       norm=Normalize(-limit, limit), aspect="auto")
        ax.set_xticks(range(len(b_values)), [f"{b:g}" for b in b_values], fontsize=8)
        ax.set_yticks(range(len(a_values)), [f"{a:g}" for a in a_values], fontsize=8.5)
        ax.set_xlabel("ESG appetite B" if xlabel else "", fontsize=9)
        ax.set_ylabel("Risk aversion A", fontsize=9)
        ax.set_title(title, fontsize=10, pad=10)
        for i in range(len(a_values)):
            for j in range(len(b_values)):
                value = values[i, j]
                label = ("n.a." if not np.isfinite(value) else f"{0:.{precision}f}"
                         if abs(value) < .5*10**(-precision) else f"{value:+.{precision}f}")
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(value) and abs(value) > .65*limit else "black")
        return im

    def comparison_figure(name, panels, title, thesis_figure=None):
        """Rows are periods, columns are references or measures, as in the thesis."""
        fig = plt.figure(figsize=(7.8 if len(panels) == 2 else 7.4, 1.65*len(roles)+2))
        layout = fig.add_gridspec(len(roles)+1, len(panels),
            height_ratios=[1]*len(roles)+[.065], left=.08, right=.98,
            top=.84, bottom=.075, wspace=.30, hspace=.75)
        cells = []
        for col, (metric, reference) in enumerate(panels):
            factor, precision, unit = DISPLAY[metric]
            for row, role in enumerate(roles):
                data = means[means.role.eq(role)&means.reference.eq(reference)]
                values = grid(data, "difference_"+metric)*factor
                im = draw_heatmap(fig.add_subplot(layout[row, col]), values, limits[metric],
                    f"{role_names[role]}   {int(days[role])} trading days", precision,
                    risk=metric == "score_risk", xlabel=row == len(roles)-1)
                for i, a in enumerate(a_values):
                    for j, b in enumerate(b_values):
                        key = (role, a, b, reference)
                        cells.append(dict(figure=name, metric=metric, role=role, reference=reference,
                            risk_aversion=a, esg_appetite=b, raw_value=values[i, j]/factor,
                            display_value=values[i, j], multiplier=factor,
                            valid_pairs=int(counts.loc[key, "difference_"+metric]),
                            total_pairs=int(totals.loc[key]), lower=-limits[metric], upper=limits[metric]))
            bar = fig.colorbar(im, cax=fig.add_subplot(layout[-1, col]), orientation="horizontal")
            bar.set_label(unit, fontsize=9)
            bar.ax.tick_params(labelsize=8)
            fig.text(.08+.90*(col+.5)/len(panels), .975,
                f"Selected PPO minus {LABELS[reference]}\n{unit}",
                ha="center", va="top", fontweight="bold", fontsize=10.5)
        values = pd.DataFrame(cells)
        exported_cells.extend(cells)
        save(fig, name, title, values, thesis_figure)

    # Figures 4.1-4.5: the combined learning and benchmark comparisons.
    comparison_figure("learning_utility_sharpe", [("preference_utility", "ppo_initial"),
        ("sharpe_ratio", "ppo_initial")], "Selected PPO minus initial PPO: utility and Sharpe", "4.1")
    for metric, number in [("preference_utility", "4.2"), ("sharpe_ratio", "4.3"),
                           ("total_return", "4.4"), ("score_risk", "4.5")]:
        comparison_figure(f"{metric}_controls", [(metric, "utility_rolling"), (metric, "utility_static")],
            f"Selected PPO minus rolling and static utility: {DISPLAY[metric][2]}", number)

    # Figure 4.6: equal y limits across A, with all B values and the initial policy.
    strategies = ["ppo_selected", "ppo_initial", "utility_rolling", "utility_static"]
    styles = {"ppo_selected": "-", "ppo_initial": ":", "utility_rolling": "--", "utility_static": "-."}
    profile = summary[summary.role.eq(comparison_role)&summary.strategy.isin(strategies)]
    specs = [("total_return", 100, "Total return (%)"), ("volatility", 100, "Annualised volatility (%)"),
             ("score_risk", 1, "ESG risk (points)")]
    fig, axes = plt.subplots(3, len(a_values), figsize=(max(5.4, 2.6*len(a_values)), 7.5), squeeze=False)
    for row, (metric, factor, ylabel) in enumerate(specs):
        values = profile[metric]*factor
        lo, hi = float(values.min()), float(values.max())
        margin = (hi-lo)*.1 if hi > lo else .1
        for col, a in enumerate(a_values):
            ax = axes[row, col]
            for strategy in strategies:
                data = profile[profile.risk_aversion.eq(a)&profile.strategy.eq(strategy)].set_index(
                    "esg_appetite").reindex(b_values)
                ax.plot(range(len(b_values)), factor*data[metric], color=COLORS[strategy],
                    ls=styles[strategy], marker="o", ms=3, lw=1.4, label=LABELS[strategy])
            ax.set_ylim(lo-margin, hi+margin)
            ax.set_xticks(range(len(b_values)), [f"{b:g}" for b in b_values], fontsize=7.5)
            ax.set_xlabel("ESG appetite B", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f"A={a:g}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955),
               ncols=2, fontsize=9, frameon=False)
    fig.suptitle(f"{role_names[comparison_role]} outcomes across investor preferences",
                 y=.995, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=.86, bottom=.075, hspace=.5, wspace=.50, left=.09, right=.99)
    save(fig, "preference_profiles", f"{role_names[comparison_role]} preference profiles", profile, "4.6")

    # Figure 4.7: saved curves already average baskets. Average seeds first,
    # then preference means, so no preference receives extra weight.
    overall_strategies = strategies+["equal_weight"]
    overall = curves[curves.strategy.isin(overall_strategies)].groupby(
        ["role", "date", "risk_aversion", "esg_appetite", "strategy"], as_index=False).value.mean()
    path_counts = overall.groupby(["role", "date", "strategy"]).size()
    if not path_counts.eq(len(a_values)*len(b_values)).all():
        raise ValueError("Overall value paths require the complete saved preference grid on each date.")
    overall = overall.groupby(["role", "date", "strategy"], as_index=False).value.mean()
    low = max(0., floor(overall.value.min()/1000)*1000)
    high = max(low+1000, ceil(overall.value.max()/1000)*1000)
    colors = {**COLORS, "equal_weight": "#9467bd"}
    line_styles = {**styles, "equal_weight": (0, (5, 2, 1, 2))}
    fig, axes = plt.subplots(1, len(roles), figsize=(3.5*len(roles), 4.2), squeeze=False)
    for ax, role in zip(axes[0], roles):
        for strategy in overall_strategies:
            data = overall[overall.role.eq(role)&overall.strategy.eq(strategy)].sort_values("date")
            ax.plot(data.date, data.value, color=colors[strategy], ls=line_styles[strategy],
                    lw=1.6, label=LABELS[strategy])
            recorded = summary[summary.role.eq(role)&summary.strategy.eq(strategy)].total_return.mean()
            if not np.isclose(data.value.iloc[-1]/data.value.iloc[0]-1, recorded, atol=1e-10):
                raise ValueError(f"Value-path endpoint differs from saved returns: {role}, {strategy}.")
        ax.set_ylim(low, high)
        ax.set_title(f"{role_names[role]}\n{int(days[role])} trading days", fontsize=10)
        ax.set_ylabel("Average portfolio value (US$)", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
        ax.tick_params(axis="x", rotation=30, labelsize=7.5)
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("Portfolio values across all investor preferences", fontweight="bold")
    fig.subplots_adjust(top=.77, bottom=.28, left=.07, right=.99, wspace=.42)
    save(fig, "overall_value_paths", "Portfolio values across all investor preferences", overall, "4.7")

    # Appendix figures B.1 and B.2. Utility levels and differences have separate scales.
    comparison_figure("learning_esg_risk", [("score_risk", "ppo_initial")],
                      "Selected PPO minus initial PPO: ESG risk", "B.1")
    fig = plt.figure(figsize=(7.8, 6.2))
    for col, role in enumerate(["training", comparison_role]):
        data = selected[selected.role.eq(role)]
        im = draw_heatmap(fig.add_axes([[.09, .59][col], .64, .39, .22]),
            grid(data, "preference_utility"), absolute_utility_limit,
            f"{role_names[role]}   {int(days[role])} trading days", precision=3)
    bar = fig.colorbar(im, cax=fig.add_axes([.09, .505, .89, .022]), orientation="horizontal")
    bar.set_label("Realised utility - common absolute scale", fontsize=9)
    changes = period_change.rename("difference").reset_index()
    im = draw_heatmap(fig.add_axes([.09, .155, .89, .21]), grid(changes, "difference"),
        limits["preference_utility"], f"{role_names[comparison_role]} minus Training: realised utility difference")
    bar = fig.colorbar(im, cax=fig.add_axes([.09, .025, .89, .018]), orientation="horizontal")
    bar.set_label("Realised utility difference", fontsize=9)
    fig.suptitle(f"Selected PPO utility in training and {role_names[comparison_role].lower()}", fontweight="bold")
    save(fig, "training_evaluation_utility", "Selected PPO utility across evaluation periods",
         levels.assign(evaluation_minus_training=period_change).reset_index(), "B.2")

    # Appendix B.3 includes every evaluated candidate step, including zero-choice columns.
    steps = sorted(history.loc[history.stage.ne("initial"), "step"].unique())
    choices = records.groupby(["risk_aversion", "esg_appetite", "selected_step"]).size().unstack(fill_value=0)
    choices = choices.reindex(pd.MultiIndex.from_product([a_values, b_values],
        names=["risk_aversion", "esg_appetite"]), columns=steps, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, len(choices)*.32+1.8)))
    im = ax.imshow(choices.to_numpy(), cmap="Blues", vmin=0, vmax=len(cfg["model_seeds"]), aspect="auto")
    ax.set_xticks(range(len(steps)), [f"{int(s):,}" for s in steps], fontsize=8)
    ax.set_yticks(range(len(choices)), [f"A={a:g}, B={b:g}" for a, b in choices.index], fontsize=8)
    for i in range(len(choices)):
        for j in range(len(steps)):
            v = choices.iloc[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > .65*len(cfg["model_seeds"]) else "black", fontsize=9)
    ax.set_xlabel("Training transitions at selected checkpoint")
    ax.set_title("Validation-selected checkpoints", fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, label="Models selected")
    save(fig, "checkpoint_choices", "Validation-selected checkpoints", choices.reset_index(), "B.3")
    comparison_figure("selection_utility_return", [("preference_utility", "ppo_final"),
        ("total_return", "ppo_final")], "Selected PPO minus final PPO: utility and return", "B.4")

    # Keep the three additional comparisons from the original report.
    for metric, reference in [("total_return", "ppo_initial"), ("sharpe_ratio", "ppo_final"),
                              ("score_risk", "ppo_final")]:
        comparison_figure(f"{metric}_vs_{reference}", [(metric, reference)],
                          f"Selected PPO minus {LABELS[reference]}: {DISPLAY[metric][2]}")

    pd.DataFrame(exported_cells).to_csv(run/"heatmap_values.csv", index=False)
    (run/"figure_scales.json").write_text(json.dumps(dict(palette="RdBu", symmetric_limits=limits,
        absolute_utility_limit=absolute_utility_limit, utility_multiplier=1,
        lower_esg_risk_is_blue=True, scale_scope="All references and roles within this run"), indent=2))

    # Tables used in the results chapter. Utility is compared within preferences,
    # while the across-preference period table reports financial outcomes only.
    period_summary = summary[summary.strategy.isin(overall_strategies)].groupby(
        ["role", "strategy"], as_index=False)[["annual_arithmetic_return", "volatility"]].mean()
    period_summary["return_days"] = period_summary.role.map(days)
    period_summary.to_csv(run/"period_summary.csv", index=False)
    financial_summary = summary.groupby(["role", "strategy"], as_index=False)[
        ["final_value", "total_return", "annual_arithmetic_return", "volatility"]].mean()
    financial_summary["net_gain"] = financial_summary.final_value-cfg["starting_value"]
    financial_summary.to_csv(run/"overall_financial_summary.csv", index=False)
    variation = []
    for (a, b), data in pairs[pairs.role.eq(comparison_role)&pairs.reference.eq("utility_rolling")].groupby(
            ["risk_aversion", "esg_appetite"]):
        gaps = data.groupby("basket_digest").difference_total_return.mean()
        variation.append(dict(role=comparison_role, risk_aversion=a, esg_appetite=b,
            mean_gap_pp=100*gaps.mean(), median_gap_pp=100*gaps.median(),
            positive_baskets=int(gaps.gt(0).sum()), total_baskets=len(gaps),
            matched_win_fraction=float(data.difference_total_return.gt(0).mean()),
            minimum_gap_pp=100*gaps.min(), maximum_gap_pp=100*gaps.max()))
    pd.DataFrame(variation).to_csv(run/"return_variation.csv", index=False)

    # Validation learning history and per-seed paths remain available below.
    initial = history[history.stage.eq("initial")][["risk_aversion", "esg_appetite", "seed", "preference_utility"]]
    history = history.merge(initial, on=["risk_aversion", "esg_appetite", "seed"], suffixes=("", "_initial"), validate="many_to_one")
    history["gain"] = history.preference_utility-history.preference_utility_initial
    fig, axes = plt.subplots(1, len(a_values), figsize=(4.3*len(a_values)+1, 4.2), squeeze=False, sharey=True)
    for ax, a in zip(axes[0], a_values):
        for b in b_values:
            g = history[history.risk_aversion.eq(a)&history.esg_appetite.eq(b)].groupby("step").gain.mean()
            ax.plot(g.index, g.values, marker="o", ms=3, label=f"B={b:g}")
        ax.axhline(0, color="grey", lw=.8)
        ax.set_title(f"A={a:g}")
        ax.set_xlabel("Training transitions")
        ax.set_ylabel("Validation utility gain over initial policy")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.grid(alpha=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(b_values), frameon=False)
    fig.suptitle("Validation learning history under each model's own preference", fontweight="bold")
    fig.subplots_adjust(top=.80, bottom=.24, wspace=.30)
    save(fig, "validation_learning", "Mean validation utility gains over each seed's initial policy. Validation is used for selection.")

    # Show every seed's actual mean portfolio-value path, not just scalar gaps.
    lower = max(0., floor(curves.value.min()/1000)*1000)
    upper = ceil(curves.value.max()/1000)*1000
    if upper <= lower:
        upper = lower+cfg["starting_value"]
    seed_colors = {seed: plt.get_cmap("tab20")(i % 20) for i, seed in enumerate(cfg["model_seeds"])}
    for a in a_values:
        for b in b_values:
            subset = curves[curves.risk_aversion.eq(a)&curves.esg_appetite.eq(b)]
            fig, axes = plt.subplots(1, len(roles), figsize=(4.4*len(roles)+1, 6.1), squeeze=False)
            for ax, role in zip(axes[0], roles):
                d = subset[subset.role.eq(role)]
                for seed in cfg["model_seeds"]:
                    s = d[d.strategy.eq("ppo_selected")&d.seed.eq(seed)]
                    ax.plot(s.date, s.value, color=seed_colors[seed], alpha=.65, lw=.8, label=f"Seed {seed}")
                for strategy, style in [("ppo_selected", "-"), ("ppo_initial", ":"),
                                        ("utility_rolling", "--"), ("utility_static", "-.")]:
                    s = d[d.strategy.eq(strategy)].groupby("date").value.mean()
                    ax.plot(s.index, s.values, color=COLORS[strategy], ls=style, lw=2,
                            label="Selected PPO mean" if strategy == "ppo_selected" else LABELS[strategy])
                ax.set_ylim(lower, upper)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
                ax.set_ylabel("Portfolio value (US$)")
                ax.set_title(role.title())
                ax.tick_params(axis="x", rotation=25, labelsize=8)
                ax.grid(alpha=.22)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            legend_rows = ceil(len(labels)/5)
            fig.subplots_adjust(top=.83, bottom=.21+.04*legend_rows, left=.07, right=.985, wspace=.30)
            fig.legend(handles, labels, loc="lower center", ncols=5, fontsize=8, frameon=False)
            fig.suptitle(f"A = {a:g}, B = {b:g}: complete-period value paths by model seed", fontweight="bold")
            save(fig, f"value_paths_A{a:g}_B{b:g}", f"A={a:g}, B={b:g}: each seed averaged across baskets, with common value limits.")
    (run/"figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_pdf(run_folder, manifest=None):
    """Write parameters, results tables and figures from a completed run."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    run = Path(run_folder)
    protocol = json.loads((run/"protocol.json").read_text())
    cfg = protocol["configuration"]
    manifest = manifest or json.loads((run/"figure_manifest.json").read_text())
    summary = pd.read_csv(run/"summary.csv")
    records = pd.read_csv(run/"training_record.csv")
    output = run/"MAIN_performance_report.pdf"
    pagesize = landscape(A4)
    width = pagesize[0]-70
    doc = SimpleDocTemplate(str(output), pagesize=pagesize,
        leftMargin=35, rightMargin=35, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    story = []

    def heading(text, style="Heading1"):
        story.extend([Paragraph(escape(str(text)), styles[style]), Spacer(1, 7)])

    def make_table(rows, widths):
        formatted = [[Paragraph(escape(str(value)), styles["BodyText"]) for value in row] for row in rows]
        item = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7ebef")),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d9d9d9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        return item

    def paired_parameters(left_rows, right_rows):
        half = width/2-8
        tables = [make_table(rows, [half*.46, half*.54]) for rows in (left_rows, right_rows)]
        item = Table([tables], colWidths=[width/2, width/2])
        item.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        story.append(item)

    completed = records.actual_transitions
    completed_steps = (f"{int(completed.iloc[0]):,}" if completed.nunique() == 1
                       else f"{int(completed.min()):,} to {int(completed.max()):,}")
    heading("Financial and ESG PPO results", "Title")
    heading(run.name, "Heading3")
    experiment = [["Experiment parameter", "Value"],
        ["Profile", cfg["profile"]],
        ["Training fits", len(cfg["risk_aversions"])*len(cfg["esg_appetites"])*len(cfg["model_seeds"])],
        ["Risk aversions A", cfg["risk_aversions"]], ["ESG appetites B", cfg["esg_appetites"]],
        ["Model seeds", cfg["model_seeds"]], ["Baskets", cfg["baskets"]],
        ["Assets per basket", cfg["number_of_assets"]]]
    experiment += [[f"{role.title()} returns", " to ".join(cfg["periods"][role])]
                   for role in ("training", "validation", "test")]
    experiment += [["Requested transitions per fit", f"{cfg['training_steps']:,}"],
        ["Completed transitions per fit", completed_steps],
        ["Checkpoint interval", cfg["checkpoint_frequency"]], ["Episode trading days", cfg["episode_days"]],
        ["Rebalance trading days", cfg["rebalance_days"]]]
    collection = Path(protocol["files"]["training"][0]).parent.name
    data = [["Data parameter", "Value"], ["Data source", cfg["data_source"]],
        ["Collection", collection], ["Refresh WRDS", cfg["refresh_wrds"]],
        ["Financial table", cfg.get("financial_table", "Not recorded")],
        ["ESG table", (f"{cfg['sustainability_library']}.{cfg['esg_table']}"
                       if "sustainability_library" in cfg else "Not recorded")],
        ["ESG field", cfg.get("esg_field_id", "Not recorded")],
        ["Financial history start", cfg["financial_start"]], ["ESG start", cfg["esg_start"]],
        ["Data end", cfg["end_date"]], ["First stock seed", cfg["first_stock_seed"]],
        ["Candidate pool", cfg["candidate_pool_size"]],
        ["Maximum collection attempts", cfg["maximum_collection_attempts"]],
        ["Minimum past observations", cfg["minimum_past_observations"]],
        ["Maximum score age (days)", cfg["maximum_score_age_days"]],
        ["Publication lag (business days)", cfg["publication_lag_business_days"]]]
    paired_parameters(experiment, data)

    story.append(PageBreak())
    heading("Financial and PPO parameters")
    financial = [["Financial / execution parameter", "Value"],
        ["Starting value (US$)", f"{cfg['starting_value']:,.2f}"],
        ["Transaction-cost rate", cfg["transaction_cost"]],
        ["Trading days per year", cfg["periods_per_year"]],
        ["Annual risk-free rate", cfg["annual_risk_free_rate"]],
        ["Reward scale", cfg["reward_scale"]],
        ["Momentum windows", cfg["momentum_days"]], ["Volatility windows", cfg["volatility_days"]],
        ["Correlation window", cfg["correlation_days"]], ["Estimation window", cfg["estimation_days"]],
        ["Utility solver iterations", cfg.get("utility_max_iterations", "Not recorded")],
        ["Utility solver tolerance", cfg.get("utility_tolerance", "Not recorded")],
        ["Shared hidden-layer width", cfg["hidden_size"]], ["Device", cfg["device"]],
        ["CPU threads", cfg["cpu_threads"]], ["Evaluate test period", cfg["run_test"]],
        ["Retain candidate checkpoints", cfg["keep_checkpoints"]]]
    names = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
             "clip_range", "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef",
             "max_grad_norm", "target_kl")
    defaults, overrides = protocol["ppo_defaults"], cfg["ppo_overrides"]
    ppo = [["PPO parameter", "Value used"]]+[[name, overrides.get(name, defaults.get(name, "Not recorded"))]
        for name in names]
    paired_parameters(financial, ppo)
    story.append(Spacer(1, 12))
    software = " | ".join(f"{name}: {version}" for name, version in protocol["software"].items())
    story.append(Paragraph(escape(software), styles["BodyText"]))

    roles = [r for r in ("training", "validation", "test") if r in set(summary.role)]
    for role in roles:
        story.append(PageBreak())
        heading(f"{role.title()} selected PPO outcomes")
        d = summary[summary.role.eq(role)&summary.strategy.eq("ppo_selected")].sort_values(["risk_aversion", "esg_appetite"])
        rows = [["A", "B", "Cumulative return %", "Annual Sharpe", "ESG risk", "Utility", "Annual volatility %", "Drawdown %"]]
        rows += [[f"{r.risk_aversion:g}", f"{r.esg_appetite:g}", f"{100*r.total_return:.2f}",
            f"{r.sharpe_ratio:.3f}", f"{r.score_risk:.2f}", f"{r.preference_utility:+.4f}",
            f"{100*r.volatility:.2f}", f"{100*r.max_drawdown:.2f}"] for r in d.itertuples()]
        fractions = [.055, .075, .135, .115, .12, .15, .18, .17]
        story.append(make_table(rows, [width*x for x in fractions]))

    story.append(PageBreak())
    heading("Financial outcomes across evaluation periods")
    period_summary = pd.read_csv(run/"period_summary.csv")
    rows = [["Period", "Strategy", "Annualised arithmetic return %", "Annual volatility %"]]
    period_strategies = ["ppo_selected", "ppo_initial", "utility_rolling", "utility_static", "equal_weight"]
    for role in roles:
        for strategy in period_strategies:
            r = period_summary[period_summary.role.eq(role)&period_summary.strategy.eq(strategy)].iloc[0]
            rows.append([role.title(), LABELS[strategy], f"{100*r.annual_arithmetic_return:.2f}",
                         f"{100*r.volatility:.2f}"])
    story.append(make_table(rows, [width*x for x in [.15, .35, .28, .22]]))

    role = "test" if "test" in roles else "validation"
    story.append(PageBreak())
    heading(f"{role.title()} financial outcomes across all preferences")
    financial_summary = pd.read_csv(run/"overall_financial_summary.csv")
    rows = [["Strategy", "Mean final value US$", "Mean gain US$", "Cumulative return %",
             "Annualised arithmetic return %", "Annual volatility %"]]
    for strategy in LABELS:
        r = financial_summary[financial_summary.role.eq(role)&financial_summary.strategy.eq(strategy)].iloc[0]
        rows.append([LABELS[strategy], f"{r.final_value:,.2f}", f"{r.net_gain:,.2f}",
            f"{100*r.total_return:.2f}", f"{100*r.annual_arithmetic_return:.2f}", f"{100*r.volatility:.2f}"])
    story.append(make_table(rows, [width*x for x in [.25, .15, .15, .15, .17, .13]]))

    story.append(PageBreak())
    heading(f"{role.title()} return differences: selected PPO minus rolling utility")
    variation = pd.read_csv(run/"return_variation.csv")
    rows = [["A", "B", "Mean gap pp", "Median basket gap pp", "Positive baskets",
             "Matched return wins %", "Minimum basket gap pp", "Maximum basket gap pp"]]
    rows += [[f"{r.risk_aversion:g}", f"{r.esg_appetite:g}", f"{r.mean_gap_pp:+.2f}",
        f"{r.median_gap_pp:+.2f}", f"{r.positive_baskets}/{r.total_baskets}",
        f"{100*r.matched_win_fraction:.1f}", f"{r.minimum_gap_pp:+.2f}",
        f"{r.maximum_gap_pp:+.2f}"] for r in variation.itertuples()]
    story.append(make_table(rows, [width*x for x in [.055, .075, .12, .15, .13, .15, .16, .16]]))

    for figure in manifest:
        story.append(PageBreak())
        path = run/figure["file"]
        fw, fh = ImageReader(str(path)).getSize()
        scale = min(width/fw, (pagesize[1]-80)/fh)
        story.append(Image(str(path), width=fw*scale, height=fh*scale))

    role = "test" if "test" in roles else "validation"
    for a in cfg["risk_aversions"]:
        story.append(PageBreak())
        heading(f"{role.title()} strategy means at A = {a:g}")
        d = summary[summary.role.eq(role)&summary.risk_aversion.eq(a)].sort_values(["esg_appetite", "strategy"])
        rows = [[f"B (A={a:g})", "Strategy", "Cumulative return %", "Annual Sharpe", "ESG risk", "Utility",
                 "Annual variance", "Daily turnover %", "Drawdown %"]]
        rows += [[f"{r.esg_appetite:g}", LABELS[r.strategy], f"{100*r.total_return:.2f}",
            f"{r.sharpe_ratio:.3f}", f"{r.score_risk:.2f}", f"{r.preference_utility:+.4f}",
            f"{r.annual_variance:.5f}", f"{100*r.average_turnover:.2f}", f"{100*r.max_drawdown:.2f}"] for r in d.itertuples()]
        fractions = [.06, .21, .095, .09, .095, .12, .11, .11, .11]
        story.append(make_table(rows, [width*x for x in fractions]))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(pagesize[0]-35, 17, f"Page {document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def make_report(run_folder):
    manifest = make_figures(run_folder)
    report = export_pdf(run_folder, manifest)
    print("PDF report:", report)
    return dict(pdf=report, figures=[Path(run_folder)/m["file"] for m in manifest])
