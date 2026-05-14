"""Matplotlib renderers for the data and performance profiles."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_STYLES = {
    "dfbd_spectral": ("-",  "#1f77b4", "DFBD + spectral design"),
    "dfbd_fd":       ("--", "#ff7f0e", "DFBD + forward differences"),
    "pdfo":          (":",  "#000000", "PDFO (BOBYQA)"),
}


def plot_data_profile(
    kappa_grid, profiles, *,
    tau: float,
    sigma: float,
    savepath: str,
    method_order: list[str] | None = None,
) -> None:
    plt.figure(figsize=(5.8, 3.6))
    order = method_order or list(profiles.keys())
    for s in order:
        if s not in profiles:
            continue
        ls, color, label = METHOD_STYLES.get(s, ("-", "k", s))
        plt.step(kappa_grid, profiles[s], where="post",
                 color=color, linestyle=ls, linewidth=1.8, label=label)
    plt.xlabel(r"Function evaluations / $(n+1)$")
    plt.ylabel("Fraction of problems solved")
    #plt.title(rf"Data profile, $\tau={tau:g}$  ($\sigma{{=}}{sigma:g}$)")
    plt.ylim(-0.02, 1.02); plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(savepath)) or ".", exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_perf_profile(
    alpha_grid, profiles, *,
    tau: float,
    sigma: float,
    savepath: str,
    method_order: list[str] | None = None,
) -> None:
    plt.figure(figsize=(5.8, 3.6))
    order = method_order or list(profiles.keys())
    for s in order:
        if s not in profiles:
            continue
        ls, color, label = METHOD_STYLES.get(s, ("-", "k", s))
        plt.step(alpha_grid, profiles[s], where="post",
                 color=color, linestyle=ls, linewidth=1.8, label=label)
    plt.xlabel(r"Performance ratio $\alpha$")
    plt.ylabel("Fraction of problems")
    plt.title(rf"Performance profile, $\tau={tau:g}$  ($\sigma{{=}}{sigma:g}$)")
    plt.ylim(-0.02, 1.02); plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(savepath)) or ".", exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
