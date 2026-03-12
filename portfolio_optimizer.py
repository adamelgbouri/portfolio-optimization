"""
========================================================
  PORTFOLIO OPTIMIZER — Asset Selector + Markowitz by AEG
========================================================
  Installation:
    pip install yfinance numpy pandas matplotlib scipy financedatabase
  Run:
    python3 portfolio_optimizer.py
========================================================
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox
import time
import sys

# ── GLOBAL PARAMETERS ─────────────────────────────────────────
START = "2019-01-01"
END   = "2026-01-01"
RF    = 0.04
N_MC  = 8000
SEED  = 42

# ═══════════════════════════════════════════════════════════════
#  PART 1 — ASSET SELECTION DIALOG
# ═══════════════════════════════════════════════════════════════

def run_asset_selector():

    # ── Load financedatabase ─────────────────────────────────
    try:
        import financedatabase as fd
        print("Loading asset database...")

        datasets = {
            "Equity"  : fd.Equities().select(market_cap=["Large Cap", "Mega Cap", "Mid Cap"]),
            "ETF"     : fd.ETFs().select(),
            "Fund"    : fd.Funds().select(),
            "Index"   : fd.Indices().select(),
            "Crypto"  : fd.Cryptos().select(),
            "Currency": fd.Currencies().select(),
        }

        allowed_exchanges = ["NMS","NAS","NYQ","NYS","PCX","ARCX","PAR","LSE","SHH","JPX","FRA","XETRA"]

        assets_list = []
        for asset_type, df in datasets.items():
            if "exchange" in df.columns:
                df = df[df["exchange"].isin(allowed_exchanges) | df["exchange"].isna()]
            for symbol, row in df.iterrows():
                name     = str(row.get("name", ""))
                market   = str(row.get("market", ""))
                exchange = str(row.get("exchange", ""))
                industry = str(row.get("industry_group") or row.get("sector", ""))
                text     = f"{name} ({symbol})  |||  {market} ({exchange})  |||  {industry}"
                assets_list.append((text, symbol))

        assets_list = sorted(assets_list, key=lambda x: x[0].lower())
        use_db = True
        print(f"   {len(assets_list)} assets loaded.")

    except Exception as e:
        print(f"   financedatabase not available ({e}). Manual input mode enabled.")
        assets_list = []
        use_db = False

    # ── Final result ─────────────────────────────────────────
    result = {"tickers": [], "start": START, "end": END, "rf": RF}

    # ── Main window ──────────────────────────────────────────
    root = tk.Tk()
    root.title("Portfolio Optimizer — Asset Selection")
    root.geometry("960x700")
    root.configure(bg="#0D1117")
    root.resizable(True, True)

    # ── Colors & fonts ───────────────────────────────────────
    BG        = "#0D1117"
    PANEL     = "#161B22"
    BORDER    = "#30363D"
    TEXT      = "#E6EDF3"
    MUTED     = "#8B949E"
    ACCENT    = "#58A6FF"
    GREEN     = "#238636"
    RED_TEXT  = "#F78166"
    FONT      = ("Consolas", 10)
    FONT_BOLD = ("Consolas", 11, "bold")
    FONT_SM   = ("Consolas", 9)
    FONT_TITLE= ("Consolas", 14, "bold")

    # ── Scrollable main canvas ───────────────────────────────
    outer = tk.Frame(root, bg=BG)
    outer.pack(fill="both", expand=True)

    canvas_main = tk.Canvas(outer, bg=BG, highlightthickness=0)
    vsb = tk.Scrollbar(outer, orient="vertical", command=canvas_main.yview)
    canvas_main.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas_main.pack(side="left", fill="both", expand=True)

    main = tk.Frame(canvas_main, bg=BG, padx=20, pady=16)
    canvas_win = canvas_main.create_window((0, 0), window=main, anchor="nw")

    def on_frame_configure(e):
        canvas_main.configure(scrollregion=canvas_main.bbox("all"))
    def on_canvas_configure(e):
        canvas_main.itemconfig(canvas_win, width=e.width)

    main.bind("<Configure>", on_frame_configure)
    canvas_main.bind("<Configure>", on_canvas_configure)

    # ── Title + trademark ────────────────────────────────────
    title_row = tk.Frame(main, bg=BG)
    title_row.pack(fill="x", anchor="w")
    tk.Label(title_row, text="PORTFOLIO OPTIMIZER",
             bg=BG, fg=ACCENT, font=FONT_TITLE).pack(side="left")
    tk.Label(title_row, text="  ™ by AEG",
             bg=BG, fg=MUTED, font=("Consolas", 10, "italic")).pack(side="left", padx=(6, 0))
    tk.Label(main, text="Search and select your assets, then run the analysis.",
             bg=BG, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(2, 14))

    # ════════════════════════════════════════════════════════
    #  SECTION A — Search (financedatabase)
    # ════════════════════════════════════════════════════════
    if use_db:
        sec_a = tk.LabelFrame(main, text=" Search an asset ",
                              bg=BG, fg=ACCENT, font=FONT_SM,
                              bd=1, relief="solid")
        sec_a.pack(fill="x", pady=(0, 12))

        tk.Label(sec_a, text="Name or ticker:", bg=BG, fg=MUTED, font=FONT_SM).pack(anchor="w", padx=8, pady=(6,0))
        search_var = tk.StringVar()
        search_entry = tk.Entry(sec_a, textvariable=search_var,
                                bg=PANEL, fg=TEXT, insertbackground="white",
                                font=FONT, relief="flat", bd=6)
        search_entry.pack(fill="x", padx=8, pady=4)

        lb_frame = tk.Frame(sec_a, bg=BG)
        lb_frame.pack(fill="x", padx=8, pady=(0,4))
        sb_db = tk.Scrollbar(lb_frame, bg=PANEL)
        sb_db.pack(side="right", fill="y")
        listbox_db = tk.Listbox(lb_frame, height=6,
                                bg=PANEL, fg=TEXT,
                                selectbackground="#1F6FEB",
                                font=FONT_SM, relief="flat",
                                yscrollcommand=sb_db.set)
        listbox_db.pack(fill="x")
        sb_db.config(command=listbox_db.yview)

        def update_db(*args):
            q = search_var.get().lower()
            listbox_db.delete(0, tk.END)
            count = 0
            for text, sym in assets_list:
                if q in text.lower():
                    listbox_db.insert(tk.END, text)
                    count += 1
                    if count >= 150:
                        break
        update_db()
        search_var.trace("w", update_db)

        def add_from_db():
            sel = listbox_db.curselection()
            if sel:
                text   = listbox_db.get(sel[0])
                # Format: "Name (SYMBOL)  ||| market (EXCHANGE) ||| ..."
                # We take the FIRST pair of parentheses = asset symbol
                symbol = text.split("(")[1].split(")")[0].strip().upper()
                add_ticker_row(symbol)

        tk.Button(sec_a, text="  + Add to list  ",
                  bg=PANEL, fg=TEXT, font=FONT_SM,
                  relief="flat", cursor="hand2",
                  command=add_from_db).pack(anchor="w", padx=8, pady=(0,8))

    # ════════════════════════════════════════════════════════
    #  SECTION B — Selected assets list
    # ════════════════════════════════════════════════════════
    sec_b = tk.LabelFrame(main,
                          text=" Selected assets (minimum 2) ",
                          bg=BG, fg=ACCENT, font=FONT_SM,
                          bd=1, relief="solid")
    sec_b.pack(fill="x", pady=(0, 12))

    rows_frame = tk.Frame(sec_b, bg=BG)
    rows_frame.pack(fill="x", padx=8, pady=6)

    ticker_rows = []   # [(frame, var, lbl)]

    def add_ticker_row(default=""):
        idx       = len(ticker_rows) + 1
        row_frame = tk.Frame(rows_frame, bg=PANEL, pady=4)
        row_frame.pack(fill="x", pady=2)

        # Index label
        lbl = tk.Label(row_frame, text=f"  #{idx}",
                       bg=PANEL, fg=MUTED, font=FONT_SM, width=4)
        lbl.pack(side="left")

        # Ticker input
        var   = tk.StringVar(value=default.upper())
        entry = tk.Entry(row_frame, textvariable=var,
                         bg=BG, fg=TEXT,
                         insertbackground="white",
                         font=FONT_BOLD,
                         relief="flat", bd=6, width=12)
        entry.pack(side="left", padx=6)

        # Hint
        hint = tk.Label(row_frame,
                        text="e.g. AAPL  /  GLD  /  SWDA.L",
                        bg=PANEL, fg="#484F58", font=FONT_SM)
        hint.pack(side="left", padx=4)

        # Remove button
        row_data = [row_frame, var, lbl]

        def remove(rd=row_data):
            if len(ticker_rows) <= 2:
                messagebox.showwarning("Minimum", "At least 2 assets are required.")
                return
            ticker_rows.remove(rd)
            rd[0].destroy()
            for i, r in enumerate(ticker_rows):
                r[2].config(text=f"  #{i+1}")

        tk.Button(row_frame, text="✕",
                  bg=PANEL, fg=RED_TEXT,
                  font=FONT_SM, relief="flat",
                  cursor="hand2",
                  command=remove).pack(side="right", padx=8)

        ticker_rows.append(row_data)

    # No default rows — user adds them manually

    tk.Button(sec_b, text="  + Add an asset  ",
              bg=PANEL, fg=TEXT, font=FONT_SM,
              relief="flat", cursor="hand2",
              command=lambda: add_ticker_row()).pack(anchor="w", padx=8, pady=(0, 8))

    # ════════════════════════════════════════════════════════
    #  SECTION C — Parameters
    # ════════════════════════════════════════════════════════
    sec_c = tk.LabelFrame(main, text=" Parameters ",
                          bg=BG, fg=ACCENT, font=FONT_SM,
                          bd=1, relief="solid")
    sec_c.pack(fill="x", pady=(0, 14))

    params_inner = tk.Frame(sec_c, bg=BG)
    params_inner.pack(fill="x", padx=8, pady=8)

    def make_param(parent, label, default, col):
        tk.Label(parent, text=label, bg=BG, fg=MUTED,
                 font=FONT_SM).grid(row=0, column=col*2, padx=(12,4), sticky="w")
        e = tk.Entry(parent, bg=PANEL, fg=TEXT,
                     insertbackground="white",
                     font=FONT, relief="flat", bd=4, width=14)
        e.insert(0, default)
        e.grid(row=0, column=col*2+1, padx=(0,10))
        return e

    start_e = make_param(params_inner, "Start:",            START,         0)
    end_e   = make_param(params_inner, "End:",              END,           1)
    rf_e    = make_param(params_inner, "Risk-free rate (%):", str(RF*100), 2)

    # ════════════════════════════════════════════════════════
    #  RUN BUTTON
    # ════════════════════════════════════════════════════════
    status_var = tk.StringVar(value="")
    tk.Label(main, textvariable=status_var,
             bg=BG, fg=GREEN, font=FONT_SM).pack(anchor="w", pady=(0,6))

    def confirm():
        tickers = []
        for frame, var, lbl in ticker_rows:
            t = var.get().strip().upper()
            if t:
                tickers.append(t)
        tickers = list(dict.fromkeys(tickers))

        if len(tickers) < 2:
            messagebox.showerror("Error", "Please select at least 2 assets.")
            return
        try:
            rf_val    = float(rf_e.get().strip()) / 100
            start_val = start_e.get().strip()
            end_val   = end_e.get().strip()
        except ValueError:
            messagebox.showerror("Error", "Invalid risk-free rate.")
            return

        result["tickers"] = tickers
        result["rf"]      = rf_val
        result["start"]   = start_val
        result["end"]     = end_val
        status_var.set(f"✓ {len(tickers)} assets selected — launching analysis...")
        root.after(400, root.destroy)

    tk.Button(main, text="  RUN ANALYSIS  ",
              bg=GREEN, fg="white",
              font=("Consolas", 13, "bold"),
              relief="flat", pady=12,
              cursor="hand2",
              command=confirm).pack(fill="x", pady=(4, 16))

    root.mainloop()

    if not result["tickers"]:
        print("No assets selected. Exiting.")
        sys.exit(0)

    return result["tickers"], result["rf"], result["start"], result["end"]


# ═══════════════════════════════════════════════════════════════
#  PART 2 — DOWNLOAD + OPTIMIZATION + DASHBOARD
# ═══════════════════════════════════════════════════════════════

def run_optimizer(ASSETS, RF, START, END):

    n = len(ASSETS)

    # Dynamic colors — adapts to any number of assets
    cmap   = matplotlib.colormaps.get_cmap("tab20")
    COLORS = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # ── 1. DOWNLOAD ──────────────────────────────────────────
    print(f"\n1. Downloading data from Yahoo Finance: {ASSETS}")
    time.sleep(3)

    raw_all = yf.download(ASSETS, start=START, end=END,
                          auto_adjust=False, progress=False)

    if raw_all.empty:
        raise ValueError("Download failed. Check tickers or try again in a few minutes.")

    if isinstance(raw_all.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in raw_all.columns.get_level_values(0) else "Close"
        raw = raw_all[col]
    else:
        raw = raw_all

    # Keep only available tickers
    available = [a for a in ASSETS if a in raw.columns]
    missing   = [a for a in ASSETS if a not in raw.columns]
    if missing:
        print(f"   ⚠ Skipped (not found): {missing}")
    if len(available) < 2:
        raise ValueError(f"Fewer than 2 assets available: {available}")

    ASSETS = available
    n      = len(ASSETS)
    cmap   = matplotlib.colormaps.get_cmap("tab20")
    COLORS = [cmap(i / max(n - 1, 1)) for i in range(n)]

    raw = raw[ASSETS].dropna()
    if raw.empty:
        raise ValueError("No data after dropna(). Check tickers and date range.")

    print(f"   {len(raw)} days x {n} assets")
    print(f"   Period: {raw.index[0].date()} → {raw.index[-1].date()}")

    # ── 2. STATISTICS ────────────────────────────────────────
    returns  = np.log(raw / raw.shift(1)).dropna()
    mu_real  = returns.mean().values * 252
    cov_real = returns.cov().values  * 252
    vols     = np.sqrt(np.diag(cov_real))

    print("\n   Individual statistics:")
    for i, a in enumerate(ASSETS):
        sh = (mu_real[i] - RF) / vols[i]
        print(f"   {a:8s}  Ret={mu_real[i]*100:+6.2f}%  "
              f"Vol={vols[i]*100:5.2f}%  Sharpe={sh:.2f}")

    # ── 3. PORTFOLIO METRICS ─────────────────────────────────
    def port_return(w): return float(w @ mu_real)
    def port_vol(w):    return float(np.sqrt(w @ cov_real @ w))
    def sharpe(w):      return (port_return(w) - RF) / port_vol(w)

    # ── 4. MONTE CARLO ────────────────────────────────────────
    print("\n2. Monte Carlo simulation...")
    np.random.seed(SEED)
    mc_rets, mc_vols, mc_sharpes = [], [], []
    for _ in range(N_MC):
        x = np.random.exponential(1, n)
        w = x / x.sum()
        mc_rets.append(port_return(w))
        mc_vols.append(port_vol(w))
        mc_sharpes.append(sharpe(w))
    mc_rets    = np.array(mc_rets)
    mc_vols    = np.array(mc_vols)
    mc_sharpes = np.array(mc_sharpes)
    print(f"   {N_MC} portfolios simulated")

    # ── 5. OPTIMIZATION ───────────────────────────────────────
    print("\n3. Optimization...")
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * n
    w0          = np.ones(n) / n

    mvp   = minimize(lambda w: w @ cov_real @ w, w0,
                     method="SLSQP", bounds=bounds, constraints=constraints)
    w_mvp = mvp.x

    best_sh, w_tan = -np.inf, w0.copy()
    for _ in range(30):
        x0  = np.random.dirichlet(np.ones(n))
        res = minimize(lambda w: -sharpe(w), x0,
                       method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success and -res.fun > best_sh:
            best_sh, w_tan = -res.fun, res.x

    min_ret = port_return(w_mvp)
    max_ret = mu_real.max() * 0.98
    front_vols, front_rets = [], []
    for target in np.linspace(min_ret, max_ret, 80):
        cons = constraints + [{"type": "eq",
                                "fun": lambda w, t=target: port_return(w) - t}]
        res  = minimize(lambda w: w @ cov_real @ w, w0,
                        method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            front_vols.append(port_vol(res.x))
            front_rets.append(target)

    print(f"   Tangent  → Sharpe={sharpe(w_tan):.3f}  "
          f"Ret={port_return(w_tan)*100:.1f}%  Vol={port_vol(w_tan)*100:.1f}%")
    print(f"   Min Var  → Vol={port_vol(w_mvp)*100:.1f}%  "
          f"Ret={port_return(w_mvp)*100:.1f}%")

    # ── 6. DASHBOARD ──────────────────────────────────────────
    print("\n4. Generating dashboard...")

    fig = plt.figure(figsize=(18, 12), facecolor="#0D1117")
    fig.suptitle(
        f"Portfolio Optimizer — {' | '.join(ASSETS)}  ({START} → {END})   ™ by AEG",
        fontsize=13, fontweight="bold", color="white", y=0.985
    )

    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.36,
                           left=0.06, right=0.97, top=0.89, bottom=0.06)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor("#161B22")
        for s in ax.spines.values(): s.set_edgecolor("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=8)
        ax.grid(True, alpha=0.12, color="#30363D")

    # Efficient frontier
    sc = ax1.scatter(mc_vols*100, mc_rets*100, c=mc_sharpes,
                     cmap="RdYlGn", s=6, alpha=0.35, zorder=2)
    cbar = fig.colorbar(sc, ax=ax1, pad=0.01, fraction=0.025)
    cbar.set_label("Sharpe", color="#8B949E", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E", fontsize=7)

    ax1.plot(np.array(front_vols)*100, np.array(front_rets)*100,
             color="#3FB950", lw=2.5, zorder=4, label="Efficient Frontier")
    cml_v = np.linspace(0, max(mc_vols)*100 * 1.2, 100)
    ax1.plot(cml_v, RF*100 + sharpe(w_tan)*cml_v, "--",
             color="#D2A8FF", lw=1.8, zorder=4, label="Capital Market Line")
    ax1.scatter(port_vol(w_tan)*100, port_return(w_tan)*100,
                s=300, marker="*", color="#FF7B72", zorder=7,
                edgecolors="white", lw=1,
                label=f"Tangent  Sharpe={sharpe(w_tan):.2f}")
    ax1.scatter(port_vol(w_mvp)*100, port_return(w_mvp)*100,
                s=200, marker="D", color="#58A6FF", zorder=7,
                edgecolors="white", lw=1, label="Min Variance")
    ax1.scatter(0, RF*100, s=150, marker="o", color="#FFA657", zorder=7,
                label=f"Risk-free rate ({RF*100:.0f}%)")
    ax1.set_title("Markowitz Efficient Frontier",
                  color="white", fontsize=11, pad=10)
    ax1.set_xlabel("Annual Volatility (%)", color="#8B949E")
    ax1.set_ylabel("Annual Return (%)", color="#8B949E")
    ax1.legend(fontsize=8, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="white")

    # Normalized prices — dynamic colors
    norm = raw / raw.iloc[0] * 100
    for i, a in enumerate(ASSETS):
        ax2.plot(norm[a].values, color=COLORS[i],
                 lw=1.2, label=a, alpha=0.85)
    ax2.set_title("Normalized Prices (base 100)", color="white", fontsize=10, pad=10)
    ax2.set_ylabel("Price", color="#8B949E")
    ax2.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
               labelcolor="white", ncol=max(1, n // 4 + 1))

    # Pie charts — dynamic colors
    for ax, w, title in [(ax3, w_tan, "Tangent Portfolio"),
                          (ax4, w_mvp, "Minimum Variance")]:
        mask   = w > 0.01
        labels = [ASSETS[i] for i in range(n) if mask[i]]
        vals   = w[mask]
        cols   = [COLORS[i] for i in range(n) if mask[i]]
        if len(vals) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.set_title(title, color="white", fontsize=9)
            continue
        wedges, _, autotexts = ax.pie(
            vals, colors=cols, autopct="%1.1f%%", pctdistance=0.75,
            wedgeprops=dict(edgecolor="#0D1117", linewidth=2), startangle=90)
        for at in autotexts:
            at.set_color("#0D1117")
            at.set_fontsize(7)
            at.set_fontweight("bold")
        ax.legend(wedges, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18),
                  ncol=max(1, len(labels) // 3 + 1), fontsize=7,
                  facecolor="#161B22", edgecolor="#30363D", labelcolor="white")
        ax.set_title(
            f"{title}\nRet={port_return(w)*100:.1f}%  Vol={port_vol(w)*100:.1f}%",
            color="white", fontsize=9, pad=6)

    # Individual Sharpe ratios — dynamic colors
    sharpes_ind = [(mu_real[i] - RF) / vols[i] for i in range(n)]
    bars = ax5.barh(range(n), sharpes_ind, color=COLORS, edgecolor="#0D1117")
    ax5.axvline(0, color="#8B949E", lw=0.8)
    ax5.axvline(sharpe(w_tan), color="#FF7B72", lw=1.5, linestyle="--",
                label=f"Tangent ({sharpe(w_tan):.2f})")
    for bar, val in zip(bars, sharpes_ind):
        offset = 0.01 if val >= 0 else -0.01
        ha     = "left" if val >= 0 else "right"
        ax5.text(val + offset, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", ha=ha,
                 color="white", fontsize=8)
    ax5.set_title("Individual Sharpe Ratio", color="white", fontsize=10)
    ax5.legend(fontsize=8, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="white")
    ax5.set_xlabel("Sharpe", color="#8B949E")
    ax5.set_yticks(range(n))
    ax5.set_yticklabels(ASSETS, color="#8B949E", fontsize=8)

    # ── Metrics bar — positioned in the gap between suptitle and plots ──
    info = (f"Tangent :  Sharpe={sharpe(w_tan):.3f}  |  "
            f"Ret={port_return(w_tan)*100:.1f}%  |  "
            f"Vol={port_vol(w_tan)*100:.1f}%  |  Weights: " +
            "  ".join(f"{ASSETS[i]}={w_tan[i]*100:.0f}%"
                      for i in range(n) if w_tan[i] > 0.01))
    fig.text(0.5, 0.938, info, ha="center", color="#8B949E", fontsize=8,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#161B22",
                       edgecolor="#30363D", alpha=0.9))

    plt.savefig("portfolio_result.png", dpi=160,
                bbox_inches="tight", facecolor="#0D1117")
    print("   Dashboard saved: portfolio_result.png")

    # ── TERMINAL OUTPUT ───────────────────────────────────────
    print("\n" + "═"*55)
    print(f"  OPTIMAL PORTFOLIO — {n} assets")
    print("═"*55)
    for i, a in enumerate(ASSETS):
        if w_tan[i] > 0.01:
            print(f"  {a:8s} : {w_tan[i]*100:5.1f}%  "
                  f"{'█' * int(w_tan[i] * 40)}")
    print(f"\n  Annualized return     : {port_return(w_tan)*100:+.2f}%")
    print(f"  Annualized volatility : {port_vol(w_tan)*100:.2f}%")
    print(f"  Sharpe ratio          : {sharpe(w_tan):.3f}")
    print("═"*55)
    print("✅ Done!")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assets, rf, start, end = run_asset_selector()

    print(f"\nAssets  : {assets}")
    print(f"Period  : {start} → {end}")
    print(f"RF      : {rf*100:.1f}%")

    run_optimizer(assets, rf, start, end)
