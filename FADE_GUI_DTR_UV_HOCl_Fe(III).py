#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UV/HOCl/Fe(III) AOP - DTR Prediction GUI (Best Model)
Optimized for laptop screen (1366x768 and above)
File: D:\HIT Research\Research Paper 2\ML data\Data2026042605 ONP HOCl.csv
GitHub: https://github.com/asifmarwat778-commits/FADE
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import webbrowser
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_PATH = r"D:\HIT Research\Research Paper 2\ML data\Data2026042605 ONP HOCl.csv"
GITHUB_URL = "https://github.com/asifmarwat778-commits/FADE"

FEATURES = [
    'UV254(mJ cm-2)',
    'HOCl(mg/L)',
    'Fe3+ (mg/L)',
    'pH',
    'NB (micromolar)',
    'MeOH(mM)',
    'ONP(mg/L)',
    'NO2-(mM)',
    'HCO3-(mM)',
    'Cl-(mM)',
    'SO42-(mM)'
]

TARGET = 'Kobs (min-1)'

FEATURE_LABELS = [
    'UV\u2082\u2085\u2084',
    'HOCl',
    'Fe\u00b3\u207a',
    'pH',
    'NB',
    'MeOH',
    'ONP',
    'NO\u2082\u207b',
    'HCO\u2083\u207b',
    'Cl\u207b',
    'SO\u2084\u00b2\u207b'
]

UNITS = [
    'mJ cm\u207b\u00b2',
    'mg/L',
    'mg/L',
    '',
    '\u03bcmolar',
    'mM',
    'mg/L',
    'mM',
    'mM',
    'mM',
    'mM'
]

# ============================================================
# MAIN APPLICATION
# ============================================================
class FADE_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FADE  \u2014  UV/HOCl/Fe(III) AOP DTR Prediction System")
        self.root.geometry("1120x720")
        self.root.minsize(950, 600)
        self.root.configure(bg='#f5f6fa')

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f5f6fa')
        self.style.configure('TLabel', background='#f5f6fa', font=('Times New Roman', 10))
        self.style.configure('TButton', font=('Times New Roman', 10, 'bold'))
        self.style.configure('TEntry', font=('Times New Roman', 10))
        self.style.configure('TCombobox', font=('Times New Roman', 10))
        self.style.configure('TLabelframe', background='#f5f6fa', font=('Times New Roman', 11, 'bold'))
        self.style.configure('TLabelframe.Label', background='#f5f6fa', font=('Times New Roman', 11, 'bold'), foreground='#1a5276')

        # Data storage
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None

        # Build UI
        self._build_ui()

        # Try auto-load
        self._auto_load()

    def _build_ui(self):
        # Main frame with padding
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 5))

        title = tk.Label(header, text="FADE  \u2014  UV/HOCl/Fe(III) AOP  \u2014  Decision Tree Regressor (DTR) Prediction System",
                        font=('Times New Roman', 16, 'bold'), bg='#1a5276', fg='white', pady=8)
        title.pack(fill=tk.X)

        subtitle = tk.Label(header, text="Best-Performing Model for k\u2092\u209b\u209b Prediction",
                           font=('Times New Roman', 11, 'italic'), bg='#f5f6fa', fg='#1a5276')
        subtitle.pack(pady=(2, 0))

        # GitHub link bar
        gh_frame = tk.Frame(header, bg='#e8f4f8', pady=4)
        gh_frame.pack(fill=tk.X, pady=(3, 0))
        tk.Label(gh_frame, text="Model Repository:", font=('Times New Roman', 10, 'bold'), bg='#e8f4f8', fg='#1a5276').pack(side=tk.LEFT, padx=10)
        gh_link = tk.Label(gh_frame, text=GITHUB_URL, font=('Times New Roman', 10, 'underline'),
                          bg='#e8f4f8', fg='#2980b9', cursor='hand2')
        gh_link.pack(side=tk.LEFT, padx=5)
        gh_link.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        tk.Label(gh_frame, text="(Click to open)", font=('Times New Roman', 9, 'italic'),
                bg='#e8f4f8', fg='gray').pack(side=tk.LEFT, padx=5)

        # Notebook tabs
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab 1: Data & Training
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="  1. Data & Training  ")
        self._build_tab1()

        # Tab 2: Prediction
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="  2. Prediction  ")
        self._build_tab2()

        # Tab 3: Results & Plots
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="  3. Results & Visualization  ")
        self._build_tab3()

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load data and train DTR model. | FADE: " + GITHUB_URL)
        status = tk.Label(main, textvariable=self.status_var, font=('Times New Roman', 9),
                         bg='#dfe6e9', fg='#2c3e50', anchor='w', pady=3, padx=5)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    # --------------------------------------------------------
    # TAB 1: DATA & TRAINING
    # --------------------------------------------------------
    def _build_tab1(self):
        # Top: File loading
        file_frame = ttk.LabelFrame(self.tab1, text=" Data Loading ", padding=8)
        file_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Label(file_frame, text="CSV Path:", font=('Times New Roman', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.path_var = tk.StringVar(value=DEFAULT_PATH)
        path_entry = ttk.Entry(file_frame, textvariable=self.path_var, width=75, font=('Consolas', 9))
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Button(file_frame, text="Load", command=self._load_data, width=8).pack(side=tk.LEFT, padx=3)

        # Middle: Two columns
        mid_frame = ttk.Frame(self.tab1)
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        # Left: Data info
        left_frame = ttk.LabelFrame(mid_frame, text=" Dataset Information ", padding=8)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self.data_info = tk.Text(left_frame, width=50, height=14, font=('Consolas', 9),
                                wrap=tk.WORD, state='disabled', bg='#ffffff', relief=tk.SUNKEN, bd=1)
        self.data_info.pack(fill=tk.BOTH, expand=True)

        # Right: Training controls
        right_frame = ttk.LabelFrame(mid_frame, text=" DTR Training ", padding=8)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(3, 0))

        # Parameters grid
        params_frame = ttk.Frame(right_frame)
        params_frame.pack(fill=tk.X, pady=2)

        ttk.Label(params_frame, text="Test Split:", font=('Times New Roman', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(params_frame, from_=0.1, to=0.4, increment=0.05, textvariable=self.test_size_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Random State:", font=('Times New Roman', 10, 'bold')).grid(row=0, column=2, sticky='w', padx=(15, 0), pady=2)
        self.rs_var = tk.IntVar(value=42)
        ttk.Spinbox(params_frame, from_=0, to=100, textvariable=self.rs_var, width=8).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(params_frame, text="max_depth:", font=('Times New Roman', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=2)
        self.max_d_var = tk.IntVar(value=10)
        ttk.Spinbox(params_frame, from_=3, to=20, textvariable=self.max_d_var, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="min_samples_split:", font=('Times New Roman', 10, 'bold')).grid(row=1, column=2, sticky='w', padx=(15, 0), pady=2)
        self.min_split_var = tk.IntVar(value=2)
        ttk.Spinbox(params_frame, from_=2, to=20, textvariable=self.min_split_var, width=8).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(params_frame, text="min_samples_leaf:", font=('Times New Roman', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=2)
        self.min_leaf_var = tk.IntVar(value=1)
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.min_leaf_var, width=8).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="max_features:", font=('Times New Roman', 10, 'bold')).grid(row=2, column=2, sticky='w', padx=(15, 0), pady=2)
        self.max_feat_var = tk.StringVar(value='None')
        ttk.Combobox(params_frame, textvariable=self.max_feat_var, values=['None', 'sqrt', 'log2'], width=8, state='readonly').grid(row=2, column=3, padx=5, pady=2)

        # Train button
        ttk.Button(right_frame, text="\u25b6  TRAIN DTR MODEL", command=self._train_dtr).pack(fill=tk.X, pady=(10, 5))

        # Results text
        self.train_result = tk.Text(right_frame, width=45, height=8, font=('Consolas', 9),
                                   wrap=tk.WORD, state='disabled', bg='#ffffff', relief=tk.SUNKEN, bd=1)
        self.train_result.pack(fill=tk.BOTH, expand=True)

        # Bottom: Feature Importance
        bottom_frame = ttk.LabelFrame(self.tab1, text=" Feature Importance (DTR) ", padding=5)
        bottom_frame.pack(fill=tk.X, padx=5, pady=3)

        self.fig_imp = plt.Figure(figsize=(10, 2.8), dpi=100)
        self.ax_imp = self.fig_imp.add_subplot(111)
        self.canvas_imp = FigureCanvasTkAgg(self.fig_imp, master=bottom_frame)
        self.canvas_imp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------
    # TAB 2: PREDICTION
    # --------------------------------------------------------
    def _build_tab2(self):
        # Two columns
        left = ttk.Frame(self.tab2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right = ttk.Frame(self.tab2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: Input parameters
        input_frame = ttk.LabelFrame(left, text=" Input Parameters ", padding=8)
        input_frame.pack(fill=tk.BOTH, expand=True)

        self.input_vars = {}

        # Create 2-column grid of inputs
        for i, (feat, label, unit) in enumerate(zip(FEATURES, FEATURE_LABELS, UNITS)):
            row = i // 2
            col = (i % 2) * 3

            ttk.Label(input_frame, text=f"{label}:", font=('Times New Roman', 10, 'bold')).grid(
                row=row, column=col, sticky='e', padx=5, pady=3)

            var = tk.DoubleVar(value=0.0)
            self.input_vars[feat] = var
            ttk.Entry(input_frame, textvariable=var, width=12, font=('Times New Roman', 10)).grid(
                row=row, column=col+1, sticky='w', pady=3)

            if unit:
                ttk.Label(input_frame, text=unit, font=('Times New Roman', 9), foreground='gray').grid(
                    row=row, column=col+2, sticky='w', padx=2)

        # Quick buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=6, column=0, columnspan=6, pady=(10, 5))

        ttk.Button(btn_frame, text="Typical UV/HOCl/Fe(III)", command=self._set_typical).grid(row=0, column=0, padx=3)
        ttk.Button(btn_frame, text="UV Only", command=self._set_uv_only).grid(row=0, column=1, padx=3)
        ttk.Button(btn_frame, text="HOCl Only", command=self._set_hocl_only).grid(row=0, column=2, padx=3)
        ttk.Button(btn_frame, text="Clear", command=self._clear_inputs).grid(row=0, column=3, padx=3)

        # Right: Prediction output
        pred_frame = ttk.LabelFrame(right, text=" Prediction ", padding=8)
        pred_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(pred_frame, text="Model Status:", font=('Times New Roman', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.model_status = ttk.Label(pred_frame, text="Not trained", font=('Times New Roman', 10), foreground='red')
        self.model_status.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Separator(pred_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky='ew', pady=8)

        ttk.Button(pred_frame, text="\u25b6  PREDICT k\u2092\u209b\u209b", command=self._predict, width=20).grid(
            row=2, column=0, columnspan=2, pady=5)

        ttk.Label(pred_frame, text="Predicted k\u2092\u209b\u209b:", font=('Times New Roman', 12, 'bold')).grid(
            row=3, column=0, sticky='w', pady=5)
        self.pred_value = ttk.Label(pred_frame, text="---", font=('Times New Roman', 22, 'bold'), foreground='#1a5276')
        self.pred_value.grid(row=4, column=0, columnspan=2, sticky='w', pady=5)

        ttk.Label(pred_frame, text="min\u207b\u00b9", font=('Times New Roman', 12), foreground='gray').grid(
            row=3, column=1, sticky='w', padx=5)

        ttk.Separator(pred_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=8)

        ttk.Label(pred_frame, text="Prediction Log:", font=('Times New Roman', 10, 'bold')).grid(
            row=6, column=0, sticky='w', pady=2)
        self.pred_log = tk.Text(pred_frame, width=40, height=10, font=('Consolas', 9),
                               wrap=tk.WORD, state='disabled', bg='#ffffff', relief=tk.SUNKEN, bd=1)
        self.pred_log.grid(row=7, column=0, columnspan=2, sticky='nsew', pady=2)
        pred_frame.grid_rowconfigure(7, weight=1)

    # --------------------------------------------------------
    # TAB 3: VISUALIZATION
    # --------------------------------------------------------
    def _build_tab3(self):
        ctrl = ttk.Frame(self.tab3)
        ctrl.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl, text="Plot:", font=('Times New Roman', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.plot_var = tk.StringVar(value='parity')
        plot_combo = ttk.Combobox(ctrl, textvariable=self.plot_var, state='readonly', width=18,
                                  values=['Parity Plot (Actual vs Predicted)',
                                          'Residual Plot',
                                          'Actual vs Predicted (Indexed)',
                                          'Feature Importance'])
        plot_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Generate Plot", command=self._generate_plot).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT, padx=5)

        plot_frame = ttk.Frame(self.tab3)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_viz = plt.Figure(figsize=(10, 5.5), dpi=100)
        self.ax_viz = self.fig_viz.add_subplot(111)
        self.canvas_viz = FigureCanvasTkAgg(self.fig_viz, master=plot_frame)
        self.canvas_viz.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_viz, plot_frame, pack_toolbar=False)
        toolbar.pack(fill=tk.X, side=tk.BOTTOM)

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _log(self, widget, text, clear=False):
        widget.configure(state='normal')
        if clear:
            widget.delete(1.0, tk.END)
        widget.insert(tk.END, text + "\n")
        widget.configure(state='disabled')

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _auto_load(self):
        import os
        if os.path.exists(DEFAULT_PATH):
            self._load_data(silent=True)

    def _set_typical(self):
        values = [1434, 10, 4.48, 3, 0, 0, 10, 0, 0, 0, 0]
        for feat, val in zip(FEATURES, values):
            self.input_vars[feat].set(val)

    def _set_uv_only(self):
        values = [1434, 0, 0, 3, 0, 0, 10, 0, 0, 0, 0]
        for feat, val in zip(FEATURES, values):
            self.input_vars[feat].set(val)

    def _set_hocl_only(self):
        values = [0, 10, 0, 3, 0, 0, 10, 0, 0, 0, 0]
        for feat, val in zip(FEATURES, values):
            self.input_vars[feat].set(val)

    def _clear_inputs(self):
        for feat in FEATURES:
            self.input_vars[feat].set(0.0)

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------
    def _load_data(self, silent=False):
        path = self.path_var.get().strip()
        try:
            self.df = pd.read_csv(path)

            missing = [c for c in FEATURES + [TARGET] if c not in self.df.columns]
            if missing:
                messagebox.showerror("Column Error", f"Missing columns: {missing}")
                return False

            info = f"File: {path}\n"
            info += f"Samples: {len(self.df)} | Features: {len(FEATURES)}\n"
            info += f"Target: {TARGET}\n"
            info += f"k_obs range: [{self.df[TARGET].min():.6f}, {self.df[TARGET].max():.6f}]\n"
            info += f"Mean k_obs: {self.df[TARGET].mean():.6f}\n"
            info += f"Std k_obs: {self.df[TARGET].std():.6f}\n\n"
            info += "Features:\n"
            for f in FEATURES:
                info += f"  \u2022 {f}\n"

            self._log(self.data_info, info, clear=True)
            self.status_var.set(f"Data loaded: {len(self.df)} samples | {len(FEATURES)} features | FADE")

            if not silent:
                messagebox.showinfo("Success", f"Data loaded!\n{len(self.df)} samples, {len(FEATURES)} features.")
            return True

        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            return False

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    def _train_dtr(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load data first.")
            return

        try:
            X = self.df[FEATURES].values
            y = self.df[TARGET].values

            test_size = self.test_size_var.get()
            rs = self.rs_var.get()
            max_d = self.max_d_var.get()
            min_split = self.min_split_var.get()
            min_leaf = self.min_leaf_var.get()
            max_feat = self.max_feat_var.get()
            if max_feat == 'None':
                max_feat = None

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=rs
            )

            X_train_s = self.scaler.fit_transform(self.X_train)
            X_test_s = self.scaler.transform(self.X_test)

            self.model = DecisionTreeRegressor(
                max_depth=max_d,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                max_features=max_feat,
                random_state=rs
            )

            self.model.fit(X_train_s, self.y_train)

            self.y_pred_train = self.model.predict(X_train_s)
            self.y_pred_test = self.model.predict(X_test_s)

            # Metrics
            r2_train = r2_score(self.y_train, self.y_pred_train)
            r2_test = r2_score(self.y_test, self.y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
            mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
            mae_test = mean_absolute_error(self.y_test, self.y_pred_test)

            result = f"DTR Training Complete\n"
            result += "=" * 35 + "\n"
            result += f"Parameters:\n"
            result += f"  max_depth={max_d}, min_samples_split={min_split}\n"
            result += f"  min_samples_leaf={min_leaf}, max_features={max_feat}\n"
            result += f"  test_size={test_size}, random_state={rs}\n"
            result += "-" * 35 + "\n"
            result += f"Train samples: {len(self.X_train)}\n"
            result += f"Test samples:  {len(self.X_test)}\n"
            result += "-" * 35 + "\n"
            result += f"R\u00b2 (Train):  {r2_train:.6f}\n"
            result += f"R\u00b2 (Test):   {r2_test:.6f}\n"
            result += f"RMSE (Train): {rmse_train:.6f}\n"
            result += f"RMSE (Test):  {rmse_test:.6f}\n"
            result += f"MAE (Train):  {mae_train:.6f}\n"
            result += f"MAE (Test):   {mae_test:.6f}\n"

            self._log(self.train_result, result, clear=True)
            self.model_status.config(text=f"DTR Ready (R\u00b2={r2_test:.4f})", foreground='green')
            self.status_var.set(f"DTR trained | Test R\u00b2={r2_test:.4f} | RMSE={rmse_test:.6f} | FADE")

            # Feature importance
            self._plot_importance()

            messagebox.showinfo("Training Complete", f"DTR trained successfully!\nTest R\u00b2 = {r2_test:.4f}\nRMSE = {rmse_test:.6f}")

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def _plot_importance(self):
        self.ax_imp.clear()

        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            self.ax_imp.text(0.5, 0.5, "Train model to see feature importance", ha='center', va='center', fontsize=11)
            self.canvas_imp.draw()
            return

        importances = self.model.feature_importances_
        idx = np.argsort(importances)[::-1]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importances)))

        self.ax_imp.barh(range(len(importances)), importances[idx], color=colors, edgecolor='black', linewidth=0.5)
        self.ax_imp.set_yticks(range(len(importances)))
        self.ax_imp.set_yticklabels([FEATURE_LABELS[i] for i in idx], fontsize=10, fontname='Times New Roman')
        self.ax_imp.set_xlabel('Importance', fontsize=11, fontname='Times New Roman')
        self.ax_imp.set_title('DTR Feature Importance', fontsize=12, fontname='Times New Roman', fontweight='bold')
        self.ax_imp.invert_yaxis()
        self.fig_imp.tight_layout()
        self.canvas_imp.draw()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    def _predict(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please train DTR model first.")
            return

        try:
            inputs = [self.input_vars[f].get() for f in FEATURES]
            X_input = np.array(inputs).reshape(1, -1)
            X_input_s = self.scaler.transform(X_input)
            pred = self.model.predict(X_input_s)[0]

            self.pred_value.config(text=f"{pred:.6f}")

            log = f"[DTR Prediction - FADE]\n"
            log += "-" * 30 + "\n"
            for label, feat, val in zip(FEATURE_LABELS, FEATURES, inputs):
                log += f"{label}: {val}\n"
            log += "-" * 30 + "\n"
            log += f"k_obs = {pred:.6f} min\u207b\u00b9\n"

            self._log(self.pred_log, log, clear=True)
            self.status_var.set(f"Prediction: k_obs = {pred:.6f} min\u207b\u00b9 | FADE")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------
    def _generate_plot(self):
        if self.model is None or self.y_test is None:
            messagebox.showwarning("No Model", "Train DTR model first.")
            return

        plot_type = self.plot_var.get()
        self.ax_viz.clear()

        if plot_type == 'Parity Plot (Actual vs Predicted)':
            self.ax_viz.scatter(self.y_test, self.y_pred_test, c='#1a5276', alpha=0.7,
                               edgecolors='black', s=70, label='Predictions', zorder=3)
            min_val = min(self.y_test.min(), self.y_pred_test.min())
            max_val = max(self.y_test.max(), self.y_pred_test.max())
            margin = (max_val - min_val) * 0.05
            self.ax_viz.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin],
                            'r--', lw=2, label='1:1 Line', zorder=2)
            self.ax_viz.set_xlabel('Actual k$_{obs}$ (min$^{-1}$)', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_ylabel('Predicted k$_{obs}$ (min$^{-1}$)', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_title('DTR Parity Plot  |  FADE', fontsize=14, fontname='Times New Roman', fontweight='bold')
            self.ax_viz.legend(fontsize=10, loc='upper left')
            r2 = r2_score(self.y_test, self.y_pred_test)
            self.ax_viz.text(0.05, 0.95, f'R$^2$ = {r2:.4f}', transform=self.ax_viz.transAxes,
                            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif plot_type == 'Residual Plot':
            residuals = self.y_test - self.y_pred_test
            self.ax_viz.scatter(self.y_pred_test, residuals, c='#1a5276', alpha=0.7,
                               edgecolors='black', s=70, zorder=3)
            self.ax_viz.axhline(y=0, color='r', linestyle='--', lw=2, zorder=2)
            self.ax_viz.set_xlabel('Predicted k$_{obs}$ (min$^{-1}$)', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_ylabel('Residuals (min$^{-1}$)', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_title('DTR Residual Plot  |  FADE', fontsize=14, fontname='Times New Roman', fontweight='bold')

        elif plot_type == 'Actual vs Predicted (Indexed)':
            x_idx = np.arange(len(self.y_test))
            self.ax_viz.scatter(x_idx, self.y_test, c='#1a5276', alpha=0.7, edgecolors='black',
                               s=70, label='Actual', zorder=3)
            self.ax_viz.scatter(x_idx, self.y_pred_test, c='#e74c3c', alpha=0.7, edgecolors='black',
                               s=70, marker='x', label='Predicted', zorder=3)
            self.ax_viz.set_xlabel('Sample Index', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_ylabel('k$_{obs}$ (min$^{-1}$)', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_title('DTR Actual vs Predicted  |  FADE', fontsize=14, fontname='Times New Roman', fontweight='bold')
            self.ax_viz.legend(fontsize=10)

        elif plot_type == 'Feature Importance':
            importances = self.model.feature_importances_
            idx = np.argsort(importances)[::-1]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importances)))
            self.ax_viz.barh(range(len(importances)), importances[idx], color=colors,
                            edgecolor='black', linewidth=0.5)
            self.ax_viz.set_yticks(range(len(importances)))
            self.ax_viz.set_yticklabels([FEATURE_LABELS[i] for i in idx], fontsize=11, fontname='Times New Roman')
            self.ax_viz.set_xlabel('Importance', fontsize=12, fontname='Times New Roman')
            self.ax_viz.set_title('DTR Feature Importance  |  FADE', fontsize=14, fontname='Times New Roman', fontweight='bold')
            self.ax_viz.invert_yaxis()

        self.fig_viz.tight_layout()
        self.canvas_viz.draw()

    def _save_plot(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
            if path:
                self.fig_viz.savefig(path, dpi=900, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FADE_GUI(root)
    root.mainloop()