import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PW AML 2026-2027 — Stress detection from physiological signals (WESAD)

    Applied Machine Learning for Real-World Data (AML)

    **Pipeline:** `windowed features -> exploration -> preprocessing -> modelling -> evaluation -> interpretation`

    This practical work takes place in one session. Read the guidelines (`AML_StressDetection_guidelines2627.md`) first. In this notebook, the preprocessing, the validation scheme and a first random forest are provided: you explore the data, **tune and extend the models**, evaluate them and interpret the results.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    1. Explore a real-world physiological dataset (missing values, imbalance, inter-subject variability).
    2. Understand a preprocessing and validation strategy that avoids **data leakage**.
    3. Tune a random forest, add at least one other classifier, and compare them with a baseline for **Base vs TSST (stress)**.
    4. Evaluate with subject-independent validation and appropriate metrics.
    5. Interpret the results (features, modalities, errors).
    6. *(Optional)* Extend to three classes (with **Fun**) and try feature selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports

    Add the imports you need as you go.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.model_selection import ParameterGrid
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        recall_score,
        ConfusionMatrixDisplay,
    )

    # '%matplotlib inline' command supported automatically in marimo

    RANDOM_STATE = 0
    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        ParameterGrid,
        RANDOM_STATE,
        accuracy_score,
        f1_score,
        np,
        pd,
        plt,
        recall_score,
        roc_auc_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("./data/part-2/stress_detection_dataset.csv")
    print(df.shape)
    df.head()
    return (df,)


@app.cell
def _(df):
    # Columns that describe the window or the target: they are NOT features.
    meta_cols = [
        "subject",
        "window_id",
        "window_start",
        "window_end",
        "window_duration_sec",
        "condition",
        "label",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    feature_cols
    return (feature_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Exploration

    In this first part we work on the binary problem **Base (0) vs TSST (1)**. The *Fun* condition is kept aside until section 6.
    """)
    return


@app.cell
def _(df):
    # Binary dataset used in sections 1 to 5.
    # Keep the Fun condition aside until section 6.
    df_bin = df[df["condition"].isin(["Base", "TSST"])].reset_index(drop=True)

    df_bin
    return (df_bin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.1 Class distribution and missing values

    Show the class distribution (overall and per subject) and the missing values (per feature, and per subject if relevant).
    """)
    return


@app.cell
def _(df_bin, feature_cols, mo, pd):
    class_overall = df_bin["condition"].value_counts().to_frame("n_windows")
    class_overall["pct"] = (
        100 * class_overall["n_windows"] / len(df_bin)
    ).round(1)

    class_per_subject = pd.crosstab(df_bin["subject"], df_bin["condition"])
    class_per_subject["TSST_pct"] = (
        100 * class_per_subject["TSST"] / class_per_subject.sum(axis=1)
    ).round(1)

    na_per_feature = df_bin[feature_cols].isna().sum()
    na_per_feature = (
        na_per_feature[na_per_feature > 0]
        .sort_values(ascending=False)
        .to_frame("n_missing")
    )
    na_per_feature["pct"] = (
        100 * na_per_feature["n_missing"] / len(df_bin)
    ).round(1)

    _na_rows = df_bin[feature_cols].isna().any(axis=1)
    na_per_subject = (
        _na_rows.groupby(df_bin["subject"])
        .sum()
        .to_frame("n_windows_with_NaN")
    )
    na_per_subject["pct"] = (
        100
        * na_per_subject["n_windows_with_NaN"]
        / df_bin.groupby("subject").size()
    ).round(1)

    mo.vstack(
        [
            mo.md("**Overall class distribution** (`pct` = % of all windows)"),
            class_overall,
            mo.md(
                "**Per subject** (`TSST_pct` = % of the subject's windows that are TSST)"
            ),
            class_per_subject,
            mo.md("**Missing values per feature** (`pct` = % of all windows)"),
            na_per_feature,
            mo.md(
                "**Windows with ≥ 1 missing value, per subject** (`pct` = % of the subject's windows)"
            ),
            na_per_subject,
        ]
    )
    return


@app.cell
def _(df_bin, feature_cols, mo):
    s8_complete = df_bin[
        (df_bin["subject"] == "S8") & ~df_bin[feature_cols].isna().any(axis=1)
    ]

    s8_dist = s8_complete["condition"].value_counts().to_frame("n_windows")
    s8_dist["pct"] = (100 * s8_dist["n_windows"] / len(s8_complete)).round(1)

    mo.vstack(
        [
            mo.md(
                "**Subject 8 class distribution, complete windows only** "
                f"({len(s8_complete)} windows without missing features, "
                f"out of {(df_bin['subject'] == 8).sum()} total)"
            ),
            s8_dist,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Are the classes balanced overall? Is it the same for every subject? What are the consequences for the choice of the metric?
    - How many missing values are there, and in which features or subjects are they concentrated? What could explain them?
    - What do these observations imply for the preprocessing (section 2)?

    **Your answer:**

    - No, the Base and TSST are not balanced, there is 63.5% of Base vs 36.5% of TSST. This is the same for all the subjects, with the TSST percentage fluacting between 35% and 40%. The metric choice should take this into account, with either selecting the F1-score or the weighted-F1-score.
    - There are a total of 7 features that each have 28 missing values. Those are all concentrated within the S8 subject and derived from a respiratory sensor. They could be explained by a missing respiratory sensore during a 28 window session.
    - Those observation imply that we should probably balance the data within the sets (or chose an appropriate loss function) and remove those S8 subject 28 windows
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.2 Base vs TSST: feature distributions

    TODO: compare feature distributions between Base and TSST. Try several features. Keep at least one feature per modality.
    """)
    return


@app.cell
def _():
    import math

    return (math,)


@app.cell
def _(math, plt):
    def plot_base_vs_tsst(
        data,
        features,
        title="",
        color="tab:blue",
        conds=("Base", "TSST"),
        ncols=None,
    ):

        n = len(features)
        ncols = ncols or math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False
        )
        axes = axes.ravel()
        for ax, f in zip(axes, features):
            values = [
                data.loc[data["condition"] == c, f].dropna() for c in conds
            ]
            bp = ax.boxplot(
                values,
                tick_labels=list(conds),
                showfliers=False,
                patch_artist=True,
            )
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.4)
            ax.set_title(f, fontsize=9)
        for ax in axes[n:]:  # hide unused cells
            ax.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    return (plot_base_vs_tsst,)


@app.function
def modality_of(feature):
    if feature.startswith(("ECG", "HRV")):
        return "ECG"
    if feature.startswith(("EDA", "SCL", "SCR")):
        return "EDA"
    return "RSP"


@app.cell
def _(df_bin, feature_cols, mo, plot_base_vs_tsst):
    features_by_modality = {
        _mod: [f for f in feature_cols if modality_of(f) == _mod]
        for _mod in ["ECG", "EDA", "RSP"]
    }
    modality_colors = {"ECG": "tab:red", "EDA": "tab:blue", "RSP": "tab:green"}

    {_mod: len(_fs) for _mod, _fs in features_by_modality.items()}

    mo.vstack(
        [
            plot_base_vs_tsst(
                df_bin,
                _feats,
                title=f"{_mod} – Base vs TSST",
                color=modality_colors[_mod],
            )
            for _mod, _feats in features_by_modality.items()
        ]
    )
    return (modality_colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.3 Correlations between features

    TODO: visualise the correlation matrix of the features.
    """)
    return


@app.cell
def _():
    # TODO: visualise the correlation matrix of the features.
    return


@app.cell
def _(df_bin, feature_cols, modality_colors, plt):
    corr = df_bin[feature_cols].corr(method="pearson")

    _fig, _ax = plt.subplots(figsize=(11, 9))
    _im = _ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(
        range(len(feature_cols)), feature_cols, rotation=90, fontsize=7
    )
    _ax.set_yticks(range(len(feature_cols)), feature_cols, fontsize=7)

    # Colour the tick labels by modality and draw lines between modality blocks
    for _lbl in _ax.get_xticklabels() + _ax.get_yticklabels():
        _lbl.set_color(modality_colors[modality_of(_lbl.get_text())])
    _mods = [modality_of(f) for f in feature_cols]
    for _i in range(1, len(_mods)):
        if _mods[_i] != _mods[_i - 1]:
            _ax.axhline(_i - 0.5, c="black", lw=1)
            _ax.axvline(_i - 0.5, c="black", lw=1)

    _fig.colorbar(_im, ax=_ax, shrink=0.8, label="Pearson ρ")
    _ax.set_title("Correlation between features (Base + TSST windows)")
    _fig.tight_layout()
    _fig
    return (corr,)


@app.cell
def _(corr, np):
    _upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    high_corr_pairs = (
        _upper.stack()
        .rename("pearson_r")
        .rename_axis(["feature_1", "feature_2"])
        .reset_index()
        .loc[lambda d: d["pearson_r"].abs() > 0.8]
        .sort_values("pearson_r", key=abs, ascending=False)
        .round(2)
        .reset_index(drop=True)
    )
    return (high_corr_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which groups of features are redundant?

    **Your answer:**
    """)
    return


@app.cell
def _(high_corr_pairs):
    high_corr_pairs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.4 Hypothesis

    Which features do you expect to be the most discriminative between baseline and stress, and why ?

    **Your hypothesis:**

    We think that the 2 most important features will be:
        - ECG_Rate_Mean
        - SCR_Count

    Both of these features show a very significant difference between the Base and TSST data, as well as not being highly correlated to a lot of other features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Preprocessing

    The preprocessing is implemented in the cell below. Read it carefully.

    - **Feature set:** only the feature columns are kept in `X` (no metadata, no label). `y` is binary: Base = 0, TSST = 1.
    - **Missing values:** windows with at least one missing feature are dropped.
    - **Normalisation:** per-subject z-score. Each feature is standardised with the mean and standard deviation of the windows of the subject itself.
    - **Class imbalance:** handled in the models with `class_weight="balanced"`, and by reporting balanced accuracy.

    > **Data leakage.** Anything that is learnt from the data must be fitted on the **training set only**. Per-subject z-scores are computed from the windows of each subject only. Keep this in mind for the discussion below, and for every new step you add (e.g. imputation, feature selection).
    """)
    return


@app.cell
def _(df_bin, feature_cols):
    # Per-subject z-score: each feature is standardised with the mean/std of the subject's own windows.
    X_raw = df_bin[feature_cols]
    mean_by_subject = X_raw.groupby(df_bin["subject"]).transform("mean")
    std_by_subject = X_raw.groupby(df_bin["subject"]).transform("std")
    X_z = ((X_raw - mean_by_subject) / std_by_subject).mask(
        std_by_subject == 0, 0.0
    )  # constant feature -> 0

    # Missing values: drop the windows with at least one missing feature.
    keep = X_z.notna().all(axis=1)

    # X, y and subjects are aligned row by row.
    # `subjects` tells the validation scheme (section 3.1) which subject each window belongs to.
    X = X_z[keep].reset_index(drop=True)
    y = df_bin.loc[keep, "label"].reset_index(drop=True)  # Base = 0, TSST = 1
    subjects = df_bin.loc[keep, "subject"].reset_index(drop=True)

    print(f"X: {X.shape} | subjects: {subjects.nunique()}")
    print("Class counts:", y.value_counts().sort_index().to_dict())
    return X, subjects, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion of the preprocessing choices:**

    - Per-subject normalisation: what are its pros and cons, in particular in a real-world deployment (what do you need from a *new* subject)?
    - Dropping the windows with missing values: what are the advantages and the drawbacks? What are the alternatives?
    - Class imbalance: why use `class_weight` and balanced accuracy rather than only accuracy?

    **Your answers:**
    -
       - Pros: Z-normalizations puts all the features on the same scale and smooth-out outliers
       - Cons: New data also has to be Z-normalized (meaning a new subject needs to be Z-normalized with all the already acquired data, it makes adding data 1 by 1 a bit troublesome)

    -
      - Pros: We keep our data clean and don't make the model bias towards the missing values (their replacement)
      - Cons: We remove some of the other feature values, we need to have enough data
      - Alternatives: Put the mean / median of the values of the subject (or of the subjects if we don't have any) where the features aren't missing when it's missing

    - This takes into account the fact that the original data is unbalanced, meaning it will handle the model bias toward the most-dominant class (Base)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Modelling

    For Base vs TSST, a **random forest** is provided as a starting point. Your tasks:

    1. **tune** the random forest (one or two hyperparameters, with a small search);
    2. **implement at least one other classifier** (e.g. logistic regression, SVM, k-NN, gradient boosting, MLP) and tune one or two of its hyperparameters;
    3. add a simple **baseline** for context;
    4. **compare** all the models.

    ### 3.1 Validation strategy

    Windows come from the same subjects and overlap by 50 s (one window every 10 s, each 60 s long). A random split would put almost identical windows in both the training and validation sets, so we use a **subject-independent** scheme: **leave-one-subject-out (LOSO)**. The scheme is already implemented below.
    """)
    return


@app.cell
def _(X, subjects, y):
    from sklearn.model_selection import LeaveOneGroupOut

    # Leave-one-subject-out: at each split, all windows of one subject form the validation set
    # and the windows of the 14 other subjects form the training set.
    # `subjects` (defined in section 2) tells the splitter which subject each row belongs to.
    # It must stay aligned with X and y.
    loso = LeaveOneGroupOut()
    splits = list(
        loso.split(X, y, groups=subjects)
    )  # list of (train_idx, val_idx) positions
    return (splits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explain what leave-one-subject-out is and what are its advantages.

    **Your answer:**

    By leaving an entire subject and not just some part of each subject, we get a nice way to do cross validation, and also remove the possiblity of our model over-fitting on the subject data it's given
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Models

    A random forest with default settings is evaluated below with LOSO. The helper `loso_predict` returns the out-of-fold predictions: each window is predicted once, by a model that did **not** see its subject during training. Predictions are stored in the dictionary `results` and reused in section 4.
    """)
    return


@app.cell
def _(RANDOM_STATE, X, np, pd, splits, subjects, y):
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    def loso_predict(model, X, y, splits):
        """Fit a fresh copy of `model` on each training set of `splits` and predict the held-out subject.

        Returns a DataFrame (same index as X) with the true label, the predicted label,
        the predicted probability of stress (if available) and the subject.
        """
        y_pred = np.zeros(len(y), dtype=int)
        y_prob = np.full(len(y), np.nan)
        for train_idx, val_idx in splits:
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred[val_idx] = fitted.predict(X.iloc[val_idx])
            if hasattr(fitted, "predict_proba"):
                y_prob[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        return pd.DataFrame(
            {
                "y_true": y.to_numpy(),
                "y_pred": y_pred,
                "y_prob": y_prob,
                "subject": subjects.to_numpy(),
            },
            index=X.index,
        )

    # Random forest provided as a starting point.
    rf = RandomForestClassifier(
        n_estimators=1,  # number of trees
        max_depth=1,  # fully grown trees
        min_samples_leaf=1000,  # minimum number of windows in a leaf
        class_weight="balanced",  # compensates for the Base / TSST imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = {}  # model name -> out-of-fold predictions
    results["Random forest (default)"] = loso_predict(rf, X, y, splits)
    res = results["Random forest (default)"]
    print(
        f"Random forest (default) - LOSO balanced accuracy: {balanced_accuracy_score(res.y_true, res.y_pred):.3f}"
    )
    return (
        RandomForestClassifier,
        balanced_accuracy_score,
        clone,
        loso_predict,
        results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tune the random forest.** Choose one or two hyperparameters (e.g. `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) and try a few values each. Keep the search small.
    """)
    return


@app.cell
def _(
    ParameterGrid,
    X,
    balanced_accuracy_score,
    clone,
    loso_predict,
    pd,
    splits,
    y,
):
    def grid_search_loso(base_model, param_grid):

        rows, best = [], (-1.0, None, None)
        for params in ParameterGrid(param_grid):
            model = clone(base_model).set_params(**params)
            preds = loso_predict(model, X, y, splits)
            score = balanced_accuracy_score(preds.y_true, preds.y_pred)
            rows.append({**params, "balanced_accuracy": round(score, 3)})
            if score > best[0]:
                best = (score, model, preds)
        return pd.DataFrame(rows), best[1], best[2]

    return (grid_search_loso,)


@app.cell
def _(plt):
    def plot_grid_search(table, x, hue=None, title=""):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x_labels = list(
            dict.fromkeys(table[x].astype(str))
        )  # keep the grid order, None -> "None"
        groups = (
            table.groupby(hue, dropna=False, sort=False)
            if hue
            else [(None, table)]
        )
        for key, g in groups:
            ax.plot(
                g[x].astype(str).map(x_labels.index),
                g["balanced_accuracy"],
                marker="o",
                label=f"{hue} = {key}" if hue else None,
            )
        ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
        ax.set_xticks(range(len(x_labels)), x_labels)
        ax.set_xlabel(x)
        ax.set_ylabel("LOSO balanced accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    return (plot_grid_search,)


@app.cell
def _(
    RANDOM_STATE,
    RandomForestClassifier,
    grid_search_loso,
    mo,
    plot_grid_search,
):
    # TODO: tune the random forest (one or two hyperparameters, small search).
    # Store the predictions of your best random forest in `results`, e.g. results["Random forest (tuned)"] = ...
    rf_grid, rf_tuned, res_rf_tuned = grid_search_loso(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20]},
    )

    mo.vstack(
        [
            mo.md(
                f"**Best random forest:** `max_depth={rf_tuned.max_depth}`, "
                f"`min_samples_leaf={rf_tuned.min_samples_leaf}`"
            ),
            plot_grid_search(
                rf_grid,
                x="max_depth",
                hue="min_samples_leaf",
                title="Random forest – LOSO tuning",
            ),
            rf_grid.sort_values("balanced_accuracy", ascending=False),
        ]
    )
    return res_rf_tuned, rf_tuned


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Add another classifier** (e.g. logistic regression, SVM, k-NN, gradient boosting, MLP) and tune one or two of its hyperparameters. Some models need the features to be scaled: the features are already standardised per subject.
    """)
    return


@app.cell
def _(LogisticRegression, grid_search_loso, mo, plot_grid_search):
    # TODO: implement at least one other classifier, tune one or two hyperparameters,
    # and store its predictions in `results`.
    lr_grid, lr_tuned, res_lr = grid_search_loso(
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"C": [0.001, 0.01, 0.1, 1, 10]},
    )

    mo.vstack(
        [
            mo.md(f"**Best logistic regression:** `C={lr_tuned.C}`"),
            plot_grid_search(
                lr_grid, x="C", title="Logistic regression – LOSO tuning"
            ),
            lr_grid.sort_values("balanced_accuracy", ascending=False),
        ]
    )
    return lr_tuned, res_lr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Add a baseline** (e.g. a classifier that always predicts the most frequent class) to put the scores in perspective.
    """)
    return


@app.cell
def _(X, loso_predict, splits, y):
    from sklearn.dummy import DummyClassifier

    res_dummy = loso_predict(
        DummyClassifier(strategy="most_frequent"), X, y, splits
    )
    return (res_dummy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** if you select the hyperparameters with the same LOSO scores that you report, why is the reported performance slightly optimistic? What could you do to prevent it?

    **Your answer:**

    Because we fine tune the model based on the performance of the "test set", meaning that our model will probably have better results that a truely unseen subject
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Evaluation

    - Report several metrics (e.g. accuracy, balanced accuracy, F1, ROC-AUC) and say which one matters most here and why.
    - Show a confusion matrix for your best model.
    - Report the **per-subject** performance.
    - Build one summary table comparing all the models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Which metric matters most?**

    - **Accuracy** is misleading here: about 64 % of the windows are Base, so the baseline that always predicts Base already gets about 0.64 accuracy without detecting any stress.
    - **Balanced accuracy** (mean of the recall of Base and the recall of TSST) is the **main metric**. It is not affected by the imbalance (chance = 0.5 whatever the class proportions), and it gives the same weight to detecting stress and to recognising rest. It is also the metric used for tuning, so the comparison is consistent.
    - **F1 (TSST)** focuses on the stress class only: it combines how many stress windows are found (recall) and how many alarms are correct (precision). It is useful if missed stress or false alarms matter more, but it depends on the class proportions.
    - **ROC-AUC** measures how well the predicted probability ranks TSST windows above Base windows, independently of the 0.5 threshold. It shows whether a better threshold could improve the model.
    - **Per-subject balanced accuracy (mean ± std)** is what LOSO is designed to measure: performance on a *new* person. The std shows whether the model works equally well for everyone.
    """)
    return


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mo,
    np,
    pd,
    res_dummy,
    res_lr,
    res_rf_tuned,
    results,
    roc_auc_score,
):
    # TODO: build a summary table comparing all models in `results` and report several metrics.
    all_results = {
        **results,  # "Random forest (default)"
        "Random forest (tuned)": res_rf_tuned,
        "Logistic regression (tuned)": res_lr,
        "Baseline (most frequent)": res_dummy,
    }

    _rows = {}
    for _name, _r in all_results.items():
        _per_subj = _r.groupby("subject")[["y_true", "y_pred"]].apply(
            lambda g: balanced_accuracy_score(g.y_true, g.y_pred)
        )
        _rows[_name] = {
            "accuracy": accuracy_score(_r.y_true, _r.y_pred),
            "balanced_accuracy": balanced_accuracy_score(_r.y_true, _r.y_pred),
            "f1_TSST": f1_score(_r.y_true, _r.y_pred),
            "roc_auc": roc_auc_score(_r.y_true, _r.y_prob)
            if _r.y_prob.notna().all()
            else np.nan,
            "bal_acc_subject_mean": _per_subj.mean(),
            "bal_acc_subject_std": _per_subj.std(),
        }

    summary = (
        pd.DataFrame(_rows)
        .T.rename_axis("model")
        .sort_values("balanced_accuracy", ascending=False)
        .round(3)
    )
    best_model_name = summary.drop(
        index=["Random forest (default)", "Baseline (most frequent)"]
    )["balanced_accuracy"].idxmax()

    mo.vstack([mo.md(f"**Best model:** {best_model_name}"), summary])
    return all_results, best_model_name


@app.cell
def _(ConfusionMatrixDisplay, all_results, best_model_name, plt):
    # TODO: show a confusion matrix for your best model (use its predictions in `results`).
    _r = all_results[best_model_name]

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_predictions(
        _r.y_true,
        _r.y_pred,
        display_labels=["Base", "TSST"],
        ax=_axes[0],
        colorbar=False,
    )
    ConfusionMatrixDisplay.from_predictions(
        _r.y_true,
        _r.y_pred,
        display_labels=["Base", "TSST"],
        normalize="true",
        values_format=".2f",
        ax=_axes[1],
        colorbar=False,
    )
    _axes[0].set_title("Counts")
    _axes[1].set_title("Normalised per true class (recall)")
    _fig.suptitle(best_model_name)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    all_results,
    balanced_accuracy_score,
    best_model_name,
    mo,
    pd,
    plt,
    recall_score,
):
    # TODO: report per-subject performance for your best model (hint: `results[name].groupby("subject")`).
    per_subject = (
        all_results[best_model_name]
        .groupby("subject")[["y_true", "y_pred"]]
        .apply(
            lambda g: pd.Series(
                {
                    "n_windows": len(g),
                    "TSST_pct": 100 * g.y_true.mean(),
                    "balanced_accuracy": balanced_accuracy_score(
                        g.y_true, g.y_pred
                    ),
                    "recall_TSST": recall_score(
                        g.y_true, g.y_pred, pos_label=1, zero_division=0
                    ),
                    "recall_Base": recall_score(
                        g.y_true, g.y_pred, pos_label=0, zero_division=0
                    ),
                }
            )
        )
        .round(3)
        .sort_values("balanced_accuracy")
    )

    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.bar(
        per_subject.index.astype(str),
        per_subject["balanced_accuracy"],
        color="tab:blue",
        alpha=0.6,
    )
    _ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    _ax.axhline(
        per_subject["balanced_accuracy"].mean(),
        c="tab:blue",
        lw=1,
        label="mean",
    )
    _ax.set_xlabel("held-out subject")
    _ax.set_ylabel("balanced accuracy")
    _ax.set_ylim(0, 1.05)
    _ax.set_title(f"Per-subject performance – {best_model_name}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([_fig, per_subject])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your comments:** Which model is the best, and is the difference with the others meaningful? Is the model equally good for all subjects?

    **Your answer:**

    For us the best model is the Logistic regression. It has the best weighted accuracy and the lowest bal_acc_subject_std.

    Interestingly, S8 and S2 are the ones with the worst performance (didn't look why yet)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Interpretation

    #### 5.1 Feature importance

    TODO: determine which features matter most for your best model and interpret the results.
    """)
    return


@app.cell
def _(RANDOM_STATE, clone, modality_colors, pd, plt):
    from sklearn.inspection import permutation_importance

    def loso_permutation_importance(model, X, y, splits, n_repeats=5):
        """Permutation importance (drop in balanced accuracy) measured on each held-out subject,
        then averaged over the LOSO folds. Returns a DataFrame: mean, std, modality per feature."""
        imps = []
        for tr, va in splits:
            fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
            pi = permutation_importance(
                fitted,
                X.iloc[va],
                y.iloc[va],
                scoring="balanced_accuracy",
                n_repeats=n_repeats,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            imps.append(pi.importances_mean)
        imp = (
            pd.DataFrame(imps, columns=X.columns)
            .agg(["mean", "std"])
            .T.sort_values("mean", ascending=False)
            .round(4)
        )
        imp["modality"] = imp.index.map(modality_of)
        return imp

    def plot_importance(imp, title="", top=15):
        """Horizontal bar plot of the `top` most important features, coloured by modality."""
        top_imp = imp.head(top)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(
            top_imp.index,
            top_imp["mean"],
            xerr=top_imp["std"],
            color=top_imp["modality"].map(modality_colors),
            alpha=0.7,
        )
        ax.axvline(0, c="grey", lw=0.8)
        ax.set_xlabel("drop in balanced accuracy when the feature is permuted")
        ax.set_title(title)
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7)
                for c in modality_colors.values()
            ],
            labels=list(modality_colors),
            fontsize=8,
            loc="lower right",
        )
        fig.tight_layout()
        return fig

    return loso_permutation_importance, plot_importance


@app.cell
def _(
    X,
    best_model_name,
    loso_permutation_importance,
    lr_tuned,
    mo,
    pd,
    plot_importance,
    rf_tuned,
    splits,
    y,
):
    tuned_models = {
        "Random forest (tuned)": rf_tuned,
        "Logistic regression (tuned)": lr_tuned,
    }

    importances = {
        _name: loso_permutation_importance(_m, X, y, splits)
        for _name, _m in tuned_models.items()
    }

    # Side-by-side comparison tables
    importance_all = pd.DataFrame(
        {_name: _imp["mean"] for _name, _imp in importances.items()}
    ).sort_values(best_model_name, ascending=False)
    importance_all.insert(0, "modality", importance_all.index.map(modality_of))

    importance_by_modality = pd.DataFrame(
        {
            _name: _imp.groupby("modality")["mean"].sum()
            for _name, _imp in importances.items()
        }
    ).round(4)

    mo.vstack(
        [
            plot_importance(
                _imp, title=f"Permutation importance (LOSO) – {_name}"
            )
            for _name, _imp in importances.items()
        ]
        + [
            mo.md("**Total importance per modality**"),
            importance_by_modality,
            mo.md(
                f"**Mean importance per feature, all models** (sorted by {best_model_name})"
            ),
            importance_all,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 5.2 Discussion

    Answer in your own words, supported by your results:

    1. Do the most important features agree with your hypothesis (1.4) ? If not, what are the differences ?
    2. Which modality seems to be the most informative?
    3. Which subjects are misclassified, why could that be ?
    4. What are the limitations of your system? Would you trust it in the real world?

    **Your answers:**

    1. Our features both showed in the top 10 of both models (ECG_Rate_Mean 3, RSP_Rate_Mean 10). This shows that doing the EDA can already give a good grasp at which features could work.
    2. It depends of the model, for the random forest, it seems to be the ECG, where in the logistic regression it's the EDA
    3. Subject 8 and 2 are the ones with the most misclassification. It would need to be looked further, maybe they have a stress disorder
    4. The limitations are that we always need the 31 same features, a subject missing one of those wouldn't work with our current model. This model could probably be tested in an exploratory analysis, but not diagnostic wise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Extension (optional) — adding the *Fun* condition

    *For students who finish early.* Use the full dataset `df` (Base / TSST / Fun) and turn the problem into a **three-class** classification. Re-evaluate your best model(s) with the same subject-independent scheme.

    - Show the confusion matrix. Which classes are confused and why?
    - Which features help to separate stress from amusement?
    """)
    return


@app.cell
def _():
    # TODO
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your comments:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus) Feature selection (optional)

    Reduce the feature set with a method of your choice (filter, wrapper, embedded, ...) and study the effect on performance.

    > The selection must be fitted on the **training set only**, inside each cross-validation fold.
    """)
    return


@app.cell
def _():
    # TODO
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your comments:**
    """)
    return


if __name__ == "__main__":
    app.run()
