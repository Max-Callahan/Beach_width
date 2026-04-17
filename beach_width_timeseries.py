import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

# ── Data loading ──────────────────────────────────────────────────────────────
weather = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_forcing_with_stormflags.csv")
df      = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv")

weather['date'] = pd.to_datetime(weather['date'])
df['date']      = pd.to_datetime(df['date'])

# filter weather 2017–2023
mask = (weather["date"].dt.year >= 2017) & (weather["date"].dt.year <= 2023)
weather_filtered = weather[mask]

# average ± std beach width per date
avg_width  = df.groupby("date")["width_m"].mean().reset_index()
std_width  = df.groupby("date")["width_m"].std().reset_index()
std_width.columns = ["date", "std_m"]

mask_bw   = (avg_width["date"].dt.year >= 2017) & (avg_width["date"].dt.year <= 2024)
avg_width = avg_width[mask_bw].merge(std_width, on="date", how="left")

output_path = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth"

# ── Design tokens ─────────────────────────────────────────────────────────────
BACKGROUND   = "#F7F5F2"       # warm off-white
PANEL_BG     = "#FFFFFF"
GRID_COLOR   = "#E8E4DE"
SPINE_COLOR  = "#C8C2B8"
TEXT_DARK    = "#1A1714"
TEXT_MID     = "#5A5550"
TEXT_LIGHT   = "#9C958E"

# Panel accent colours — muted, distinct
ACCENT_A     = "#2B5F8E"   # deep ocean blue  — beach width
ACCENT_B     = "#4A7C6F"   # sea-green        — wind speed
ACCENT_C     = "#7B5EA7"   # muted purple     — wave height
ACCENT_D     = "#B5692A"   # warm terracotta  — pressure

FILL_ALPHA   = 0.15
STORM_COLOR  = "#C0392B"
STORM_ALPHA  = 0.3

ACCENT_COLORS = [ACCENT_A, ACCENT_B, ACCENT_C, ACCENT_D]

# ── Storm intervals ───────────────────────────────────────────────────────────
storm_series = weather_filtered.set_index("date")["is_storm_day"]
storm_intervals, in_storm, start, prev_date = [], False, None, None
for date, is_storm in storm_series.items():
    if is_storm and not in_storm:
        start, in_storm = date, True
    elif not is_storm and in_storm:
        storm_intervals.append((start, prev_date))
        in_storm = False
    prev_date = date
if in_storm:
    storm_intervals.append((start, prev_date))

# ── Font setup (Helvetica Neue → fallback to DejaVu Sans) ────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.minor.size":     2,
    "ytick.minor.size":     2,
    "figure.facecolor":     BACKGROUND,
    "axes.facecolor":       PANEL_BG,
    "savefig.facecolor":    BACKGROUND,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           GRID_COLOR,
    "grid.linewidth":       0.5,
    "grid.alpha":           1.0,
    "legend.frameon":       False,
    "legend.fontsize":      8,
    "text.color":           TEXT_DARK,
    "axes.labelcolor":      TEXT_MID,
    "xtick.color":          TEXT_MID,
    "ytick.color":          TEXT_MID,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11), dpi=150)
fig.patch.set_facecolor(BACKGROUND)

# Outer margins
outer = gridspec.GridSpec(
    1, 1,
    left=0.07, right=0.97,
    top=0.91,  bottom=0.06,
)
inner = gridspec.GridSpecFromSubplotSpec(
    4, 1,
    subplot_spec=outer[0],
    hspace=0.32,
    height_ratios=[2.4, 1, 1, 1],
)
axes = [fig.add_subplot(inner[i]) for i in range(4)]

vars_weather = ["daily_max_sust_knots", "WVHT_mean", "daily_min_press_hpa"]
ylabels      = [
    "Beach width (m)",
    "Max wind speed (kn)",
    "Significant wave\nheight (m)",
    "Min SLP (hPa)",
]
panel_letters = ["A", "B", "C", "D"]

# ── Storm shading ─────────────────────────────────────────────────────────────
for ax in axes:
    for (t0, t1) in storm_intervals:
        ax.axvspan(t0, t1, color=STORM_COLOR, alpha=STORM_ALPHA, linewidth=0, zorder=0)

# ── Panel A — beach width ─────────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(
    avg_width["date"],
    avg_width["width_m"] - avg_width["std_m"],
    avg_width["width_m"] + avg_width["std_m"],
    color=ACCENT_A, alpha=FILL_ALPHA, linewidth=0, zorder=2,
)
ax.plot(
    avg_width["date"], avg_width["width_m"],
    color=ACCENT_A, linewidth=1.6, zorder=3,
)
ax.scatter(
    avg_width["date"], avg_width["width_m"],
    color=ACCENT_A, s=14, zorder=4, linewidths=0,
)

# ── Panels B–D — weather variables ───────────────────────────────────────────
for ax, var, color in zip(axes[1:], vars_weather, ACCENT_COLORS[1:]):
    series = weather_filtered[var].values
    dates  = weather_filtered["date"].values
    ax.fill_between(dates, series.min(), series, color=color, alpha=0.08, linewidth=0, zorder=1)
    ax.plot(dates, series, color=color, linewidth=0.9, zorder=2)

axes[3].set_ylim(958, 1042)
axes[3].yaxis.set_major_locator(mticker.MultipleLocator(20))

# ── Spine styling ─────────────────────────────────────────────────────────────
for ax in axes:
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.yaxis.set_tick_params(which="both", right=False)
    ax.xaxis.set_tick_params(which="both", top=False)
    if ax is not axes[-1]:
        ax.xaxis.set_tick_params(bottom=False, labelbottom=False)

# ── Y-axis labels ─────────────────────────────────────────────────────────────
for ax, label in zip(axes, ylabels):
    ax.set_ylabel(label, fontsize=8.5, color=TEXT_MID, labelpad=6,
                  linespacing=1.4)

# ── X-axis ────────────────────────────────────────────────────────────────────
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

# ── Panel letters ─────────────────────────────────────────────────────────────
for ax, letter, color in zip(axes, panel_letters, ACCENT_COLORS):
    ax.text(
        0.012, 0.93, letter,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        color=color, va="top", ha="left",
        zorder=5,
    )

# ── Legend (panel A) ─────────────────────────────────────────────────────────
mean_line = Line2D([0], [0], color=ACCENT_A, linewidth=1.6)
sd_patch  = mpatches.Patch(color=ACCENT_A, alpha=0.3)
storm_patch = mpatches.Patch(color=STORM_COLOR, alpha=0.45)
axes[0].legend(
    handles=[mean_line, sd_patch, storm_patch],
    labels=["Mean width", "±1 SD", "Storm event"],
    loc="upper right",
    fontsize=8,
    handlelength=1.6,
    handleheight=0.9,
    handletextpad=0.5,
    borderpad=0.6,
    labelspacing=0.45,
    labelcolor=TEXT_MID,
)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.07, 0.955,
    "Dynamics of the Dry Upper Beach at Samoa / Ma-Leʼl Dunes",
    fontsize=15, fontweight="bold", color=TEXT_DARK, va="bottom", ha="left",
)
fig.text(
    0.07, 0.935,
    "Meteo-oceanic forcing and morphological response  ·  2017–2024",
    fontsize=9.5, color=TEXT_LIGHT, va="bottom", ha="left",
    fontstyle="italic",
)

# thin rule under title
from matplotlib.lines import Line2D as _L
fig.add_artist(
    _L([0.07, 0.97], [0.928, 0.928],
       transform=fig.transFigure,
       color=SPINE_COLOR, linewidth=0.8)
)

# ── Source note ───────────────────────────────────────────────────────────────
fig.text(
    0.07, 0.005,
    "Data: Samoa / Ma-Leʼl Dunes monitoring programme  ·  NOAA meteorological buoy",
    fontsize=7, color=TEXT_LIGHT, va="bottom", ha="left",
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(
    f"{output_path}/multipanel_timeseries.png",
    dpi=150, bbox_inches="tight",
)
plt.show()