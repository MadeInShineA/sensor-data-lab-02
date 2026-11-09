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

    # '%matplotlib inline' command supported automatically in marimo

    RANDOM_STATE = 0
    return RANDOM_STATE, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("stress_detection_dataset.csv")
    print(df.shape)
    df.head()
    return (df,)


@app.cell
def _(df):
    # Columns that describe the window or the target: they are NOT features.
    meta_cols = ["subject", "window_id", "window_start", "window_end", "window_duration_sec", "condition", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
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
    return (df_bin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.1 Class distribution and missing values

    Show the class distribution (overall and per subject) and the missing values (per feature, and per subject if relevant).
    """)
    return


@app.cell
def _():
    # TODO: show class distribution (overall and per subject) and missing values (per feature, and per subject if relevant)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Are the classes balanced overall? Is it the same for every subject? What are the consequences for the choice of the metric?
    - How many missing values are there, and in which features or subjects are they concentrated? What could explain them?
    - What do these observations imply for the preprocessing (section 2)?

    **Your answer:**
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
    # TODO: compare distributions for at least one feature per modality.
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which groups of features are redundant?

    **Your answer:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.4 Hypothesis

    Which features do you expect to be the most discriminative between baseline and stress, and why ?

    **Your hypothesis:**
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
    X_z = ((X_raw - mean_by_subject) / std_by_subject).mask(std_by_subject == 0, 0.0)  # constant feature -> 0

    # Missing values: drop the windows with at least one missing feature.
    keep = X_z.notna().all(axis=1)

    # X, y and subjects are aligned row by row.
    # `subjects` tells the validation scheme (section 3.1) which subject each window belongs to.
    X = X_z[keep].reset_index(drop=True)
    y = df_bin.loc[keep, "label"].reset_index(drop=True)   # Base = 0, TSST = 1
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
    splits = list(loso.split(X, y, groups=subjects))  # list of (train_idx, val_idx) positions
    return (splits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explain what leave-one-subject-out is and what are its advantages.

    **Your answer:**
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
        return pd.DataFrame({"y_true": y.to_numpy(), "y_pred": y_pred, "y_prob": y_prob,
                             "subject": subjects.to_numpy()}, index=X.index)


    # Random forest provided as a starting point.
    rf = RandomForestClassifier(
        n_estimators=1,          # number of trees
        max_depth=1,            # fully grown trees
        min_samples_leaf=1000,        # minimum number of windows in a leaf
        class_weight="balanced",   # compensates for the Base / TSST imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = {}   # model name -> out-of-fold predictions
    results["Random forest (default)"] = loso_predict(rf, X, y, splits)
    res = results["Random forest (default)"]
    print(f"Random forest (default) - LOSO balanced accuracy: {balanced_accuracy_score(res.y_true, res.y_pred):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tune the random forest.** Choose one or two hyperparameters (e.g. `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) and try a few values each. Keep the search small.
    """)
    return


@app.cell
def _():
    # TODO: tune the random forest (one or two hyperparameters, small search).
    # Store the predictions of your best random forest in `results`, e.g. results["Random forest (tuned)"] = ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Add another classifier** (e.g. logistic regression, SVM, k-NN, gradient boosting, MLP) and tune one or two of its hyperparameters. Some models need the features to be scaled: the features are already standardised per subject.
    """)
    return


@app.cell
def _():
    # TODO: implement at least one other classifier, tune one or two hyperparameters,
    # and store its predictions in `results`.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Add a baseline** (e.g. a classifier that always predicts the most frequent class) to put the scores in perspective.
    """)
    return


@app.cell
def _():
    # TODO: add a simple baseline and store its predictions in `results`.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** if you select the hyperparameters with the same LOSO scores that you report, why is the reported performance slightly optimistic? What could you do to prevent it?

    **Your answer:**
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
def _():
    # TODO: build a summary table comparing all models in `results` and report several metrics.
    return


@app.cell
def _():
    # TODO: show a confusion matrix for your best model (use its predictions in `results`).
    return


@app.cell
def _():
    # TODO: report per-subject performance for your best model (hint: `results[name].groupby("subject")`).
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your comments:** Which model is the best, and is the difference with the others meaningful? Is the model equally good for all subjects?

    **Your answer:**
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
def _():
    # TODO: determine which features matter most for your best model.
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
