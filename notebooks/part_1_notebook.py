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
    # PW AML 2026-2027 — Classifying physical activities and gestures

    Applied Machine Learning for Real-World Data (AML)

    **Pipeline:** `time series -> windowing -> features -> dimensionality reduction (PCA, t-SNE, UMAP) -> classification (KNN, Random Forest, ANN)`

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    By the end of the two sessions you should be able to:

    1. Record acceleration with **Phyphox** and understand the exported format.
    2. Visualise and inspect raw time series.
    3. Split continuous signals into **windows**.
    4. Extract **time-domain and frequency-domain features**.
    5. Explore high-dimensional features with **PCA, t-SNE and UMAP**.
    6. Train and compare **KNN, Random Forest and an ANN (MLP)**.
    7. Evaluate (cross-validation, confusion matrix) and interpret the performances.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    # '%pip install numpy pandas matplotlib' command supported automatically in marimo

    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        GroupKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # '%matplotlib inline' command supported automatically in marimo
    return (
        ConfusionMatrixDisplay,
        GridSearchCV,
        GroupKFold,
        KNeighborsClassifier,
        MLPClassifier,
        PCA,
        RandomForestClassifier,
        StandardScaler,
        TSNE,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        make_pipeline,
        np,
        os,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell gives access to your Google Drive from the notebook. Uncomment it only if you
    run the notebook on Google Colab and your data are stored on Drive; adapt `base_dir`
    accordingly. If you run locally, leave it commented and keep your data in the `data/` folder.
    """)
    return


@app.cell
def _():
    # from google.colab import drive
    # drive.mount('/content/drive')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read raw data for each activity
    """)
    return


@app.cell
def _(os, pd):
    # TODO: point `base_dir` to the folder that contains one sub-folder per activity.
    # The provided example is stored in `data/example_activity`.
    base_dir = "./data/part-1/"
    recordings = {}
    folders = [
        f
        for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and (not f.startswith("."))
    ]
    print("List of activities:")
    for this_folder in folders:
        print(this_folder)
        data_csv = os.path.join(base_dir, this_folder, "Raw Data.csv")
        if os.path.isfile(data_csv):
            try:
                _data_df = pd.read_csv(data_csv, sep="\t")
                _data_df.set_index("Time (s)", inplace=True)
            except KeyError as e:
                _data_df = pd.read_csv(data_csv, sep=",")
                _data_df.set_index("Time (s)", inplace=True)
            finally:
                recordings[this_folder] = _data_df
        else:  # Check that the file exists before reading it
            print(f"Warning: '{data_csv}' not found. Skipping...")
    labels = list(recordings.keys())
    print("\nChannels:", list(recordings[labels[0]].columns))
    return labels, recordings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation of the dataframe
    """)
    return


@app.cell
def _(recordings):
    for key, dataset in recordings.items():
        dataset.sort_values(by="Time (s)", inplace=True)
    return


@app.cell
def _(labels, recordings):
    print(labels[0] + ":")
    recordings[labels[0]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Data visualisation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 1: Accelerations for one activity
    """)
    return


@app.cell
def _(plt, recordings):
    # TODO: set these two variables for your data
    activity_of_interest = "subway"  # must be one of `labels`
    _data_df = recordings[activity_of_interest]
    _colors = ["blue", "orange", "green", "red"]
    _fig, axs = plt.subplots(
        len(_data_df.columns), 1, figsize=(15, 8), sharex=True, sharey=True
    )
    # A colour for each subplot
    for _i, (column, color) in enumerate(zip(_data_df.columns, _colors)):
        axs[_i].plot(_data_df.index, _data_df[column], label=column, color=color)
        axs[_i].set_ylabel("(m/s^2)")
        axs[_i].legend(loc="upper right")
        axs[_i].grid(True)
    axs[-1].set_xlabel("Time (s)")
    plt.suptitle("Acceleration vs Time. " + activity_of_interest)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 2: One acceleration channel for all activities
    """)
    return


@app.cell
def _(plt, recordings):
    # TODO: choose the channel you want to compare across activities
    measurement_of_interest = (
        "Linear Acceleration x (m/s^2)"  # one of the channels above
    )
    _fig, axes = plt.subplots(
        len(recordings), 1, figsize=(15, 2 * len(recordings)), sharex=True, sharey=True
    )
    for _i, (activity, _data_df) in enumerate(recordings.items()):
        ax = axes[_i]
        ax.plot(_data_df.index, _data_df[measurement_of_interest], color="purple")
        ax.set_title(f"{activity}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid(True)
    _fig.suptitle(measurement_of_interest + " for all activities", fontsize=16)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(duration_seconds, mo):
    mo.md(f"""
    Remove the first and last {duration_seconds} seconds of each recording to avoid noise and transitions between activities.
    """)
    return


@app.cell
def _(recordings):
    recordings_cleaned = recordings.copy()

    for _label, _df in recordings_cleaned.items():
        _df.drop(_df[_df.index < 5].index, inplace=True)
        _df.drop(_df[_df.index > (_df.index.max() - 5)].index, inplace=True)
    return (recordings_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remove the first and last
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Windowing

    A classifier cannot consume a full 1-minute recording directly. You will split each
    recording into short **windows** (e.g. 1–2 s), with an optional **overlap**. Each window
    becomes one data point, labelled with the activity name.

    Trade-off to keep in mind: shorter windows give more data points but noisier features;
    longer windows give more stable features but fewer samples.
    """)
    return


@app.cell
def _(np, recordings_cleaned):
    # TODO: choose these values and justify your choice in the reflection cell below.
    # Remove the first and last 5 seconds of each recording to avoid noise and transitions between activities.

    duration_seconds = 5  # window length in seconds
    overlap = 0.5  # fraction of overlap (0 = no overlap)
    if duration_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("Choose a positive duration and overlap between 0 and 1.")
    n_sec_recordings = {}
    for _label, df in recordings_cleaned.items():
        sampling_frequency = 1 / np.mean(np.diff(df.index.values))
        rows_per_window = int(duration_seconds * sampling_frequency)
        step = max(1, int(rows_per_window * (1 - overlap)))
        values = df.values
        starts = range(0, len(values) - rows_per_window + 1, step)
        if rows_per_window < 1 or len(starts) == 0:
            raise ValueError(f"Recording {_label!r} is too short for this window size.")
        n_sec_recordings[_label] = np.stack(
            [values[start : start + rows_per_window] for start in starts], axis=2
        )
        print(
            f"{_label}: {n_sec_recordings[_label].shape}"
        )  # Shape: [rows/window, channels, number of windows]
    return duration_seconds, n_sec_recordings, overlap, sampling_frequency


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which window length and overlap did you choose, and why? What happens to
    the number of windows when you increase the overlap? What could be the risk of using windows that
    overlap, and what consequence does this have for evaluating your classifier?

    *Write your answer here*

    We chose a window of 5 seconds because it covers all the timeframe of the activities (hammer, violin, subway start and stop etc)

    We also applied an overlap of 0.5 seconds because this represents 10% of the window length and seemed reasonable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Feature extraction

    Each window is summarised by a compact **feature vector**. Start with time-domain features
    and then add at least one frequency-domain feature. This step has an important impact on the final
    accuracy.

    Each window contains **four channels**: the linear acceleration along the **x**, **y** and **z**
    axes, and the **absolute acceleration** (the norm of the three axes). Each feature is likely
    to be computed **for each channel**, so with *k* features you obtain a vector of
    *4 × k* values per window (e.g. 10 features → 40 values).
    """)
    return


@app.cell
def _():
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew

    return kurtosis, skew, welch


@app.cell
def _(kurtosis, n_sec_recordings, np, sampling_frequency, skew, welch):
    def extract_window_features(X, fs):
        """
        X: np.array of shape [n_samples, n_channels, n_windows]
        Returns:
            F: np.array of shape [n_windows, n_features]
            names: list of feature names
        """
        n_samples, n_channels, n_windows = X.shape
        eps = 1e-12

        feats = []
        names = []

        def add(name, arr_wc):
            # arr_wc has shape [n_windows, n_channels]
            feats.append(arr_wc)
            for c in range(n_channels):
                names.append(f"{name}_ch{c}")

        # ---------- Time-domain features ----------
        add("mean", np.mean(X, axis=0).T)
        add("std", np.std(X, axis=0).T)
        add("min", np.min(X, axis=0).T)
        add("max", np.max(X, axis=0).T)

        # Higher-order statistics
        sk = np.empty((n_windows, n_channels))
        ku = np.empty((n_windows, n_channels))
        pk = np.empty((n_windows, n_channels))
        for w in range(n_windows):
            for c in range(n_channels):
                x = X[:, c, w]
                sk[w, c] = skew(x)
                ku[w, c] = kurtosis(x)
                pk[w, c] = np.max(x) - np.min(x)
        add("skewness", sk)
        add("kurtosis", ku)
        add("peak_to_peak", pk)

        # ---------- Frequency-domain features ----------
        dom_f = np.empty((n_windows, n_channels))
        spec_centroid = np.empty((n_windows, n_channels))
        spec_entropy = np.empty((n_windows, n_channels))
        band_low = np.empty((n_windows, n_channels))
        band_mid = np.empty((n_windows, n_channels))
        band_high = np.empty((n_windows, n_channels))

        nperseg = min(n_samples, 256)
        for w in range(n_windows):
            x = X[:, :, w]  # [n_samples, n_channels]
            f, P = welch(x, fs=fs, axis=0, nperseg=nperseg)
            P = P + eps
            Pn = P / np.sum(P, axis=0, keepdims=True)

            # Ignore DC for dominant frequency
            dom_idx = np.argmax(P[1:, :], axis=0) + 1
            dom_f[w, :] = f[dom_idx]

            spec_centroid[w, :] = np.sum(f[:, None] * Pn, axis=0)
            spec_entropy[w, :] = -np.sum(Pn * np.log(Pn), axis=0) / np.log(len(f))

            # Adjust bands if your activities have different main frequencies
            low_mask = (f >= 0.5) & (f < 3.0)
            mid_mask = (f >= 3.0) & (f < 8.0)
            high_mask = (f >= 8.0) & (f <= 15.0)

            band_low[w, :] = np.sum(P[low_mask, :], axis=0)
            band_mid[w, :] = np.sum(P[mid_mask, :], axis=0)
            band_high[w, :] = np.sum(P[high_mask, :], axis=0)

        add("dominant_freq", dom_f)
        add("spectral_centroid", spec_centroid)
        add("spectral_entropy", spec_entropy)
        add("power_0.5_3Hz", band_low)
        add("power_3_8Hz", band_mid)
        add("power_8_15Hz", band_high)

        F = np.concatenate(feats, axis=1)
        return F, names

    features = {}
    feature_names = None

    for _label, X in n_sec_recordings.items():
        F, names = extract_window_features(X, sampling_frequency)

        features[_label] = F
        if feature_names is None:
            feature_names = names

        print(_label, features[_label].shape)
    return feature_names, features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** List the features you implemented. Which ones do you expect to be the
    most discriminative for your activities, and why?

    *Write your answer here.*

    We divided our features into 2 different categories:

     - Time domain features
     - Frequency domain features

    Our time domain features are the following:

     - Min
     - Max
     - Mean
     - STD
     - Skewness
     - Kurtosis
     - Peak to peak

    Our freqency domain features are the following:

     - Dominant frequency
     - Spectral centroid
     - Spectral entropy
     - Power between 0.5 and 3 HZ
     - Power between 3 and 8 Hz
     - Power between 8 and 15 HZ

    We expect the discriminating feature to be the ** Dominant frequency** as it's probably the one that carecterize each activity the best overall
    """)
    return


@app.cell(hide_code=True)
def _(
    kurtosis,
    mo,
    n_sec_recordings,
    np,
    plt,
    sampling_frequency,
    skew,
    welch,
):
    import base64 as _base64
    import io as _io

    # Illustrate skewness, kurtosis, peak to peak and each frequency-domain feature
    # on real windows (absolute acceleration channel)
    _abs_ch = 3

    def _window_psd(label):
        _X = n_sec_recordings[label]
        _x = _X[:, _abs_ch, _X.shape[2] // 2]  # window in the middle of the recording
        _f, _P = welch(_x, fs=sampling_frequency, nperseg=min(_X.shape[0], 256))
        return _f, _P + 1e-12

    def _to_img(fig):
        _buf = _io.BytesIO()
        fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return f'<img src="data:image/png;base64,{_base64.b64encode(_buf.getvalue()).decode()}" width="380">'

    # Every illustration compares two activities: the one with the lowest and the one with the
    # highest value of the feature (window in the middle of each recording).
    _pair_colors = ["tab:blue", "tab:orange"]

    def _two_panels(sharey=False):
        return plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True, sharey=sharey)

    def _psd_extremes(stat):
        """Activities with the lowest and highest value of `stat(f, P)`."""
        _order = sorted(n_sec_recordings, key=lambda _l: stat(*_window_psd(_l)))
        return _order[0], _order[-1]

    def _value_box(ax, text, color):
        ax.text(
            0.98,
            0.92,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=color),
        )

    def _window_signal(label):
        _X = n_sec_recordings[label]
        _x = _X[:, _abs_ch, _X.shape[2] // 2]
        return np.arange(len(_x)) / sampling_frequency, _x

    def _extremes(stat):
        """Activities with the lowest and highest value of `stat` (middle window)."""
        _order = sorted(n_sec_recordings, key=lambda _l: stat(_window_signal(_l)[1]))
        return _order[0], _order[-1]

    _rows = []

    # Skewness: histogram of the values, mean vs median
    _fig, _axes = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=False)
    for _ax, _label, _color in zip(_axes, _extremes(skew), ["tab:blue", "tab:orange"]):
        _x = _window_signal(_label)[1]
        _ax.hist(_x, bins=40, color=_color, alpha=0.85)
        _ax.axvline(
            np.mean(_x), color="black", linestyle="--", linewidth=1.2, label="mean"
        )
        _ax.axvline(
            np.median(_x), color="red", linestyle=":", linewidth=1.5, label="median"
        )
        _ax.set_title(
            f"{_label}: skewness = {skew(_x):.2f}",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.set_ylabel("Number of\nsamples")
        _ax.grid(True, alpha=0.4)
        _ax.legend(fontsize=7)
    _axes[-1].set_xlabel("Absolute acceleration (m/s^2)")
    _fig.tight_layout()
    _rows.append(
        (
            "Skewness",
            _to_img(_fig),
            "How lopsided the distribution of the values is: close to 0 when symmetric, large and positive when there is a long tail of high values (mean pulled away from the median).",
        )
    )

    # Kurtosis: smooth signal vs signal with rare large spikes
    _fig, _axes = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True)
    for _ax, _label, _color in zip(
        _axes, _extremes(kurtosis), ["tab:blue", "tab:orange"]
    ):
        _t, _x = _window_signal(_label)
        _ax.plot(_t, _x, color=_color, linewidth=1)
        _ax.axhspan(
            np.mean(_x) - 3 * np.std(_x),
            np.mean(_x) + 3 * np.std(_x),
            color="grey",
            alpha=0.2,
            label="mean ± 3 std",
        )
        _ax.set_title(
            f"{_label}: kurtosis = {kurtosis(_x):.2f}",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.set_ylabel("Acceleration\n(m/s^2)")
        _ax.grid(True, alpha=0.4)
        _ax.legend(fontsize=7, loc="upper right")
    _axes[-1].set_xlabel("Time in the window (s)")
    _fig.tight_layout()
    _rows.append(
        (
            "Kurtosis",
            _to_img(_fig),
            "How much of the signal comes from rare extreme peaks: around 0 or below for a smooth, regular signal, high when short spikes stick out far from the rest (outside the grey band). scipy returns the excess kurtosis, so a normal distribution gives 0.",
        )
    )

    # Peak to peak
    _fig, _axes = _two_panels(sharey=True)
    for _ax, _label, _color in zip(_axes, _extremes(np.ptp), _pair_colors):
        _t, _x = _window_signal(_label)
        _ax.plot(_t, _x, color=_color, linewidth=1)
        _ax.axhline(_x.max(), color="red", linestyle="--", linewidth=1)
        _ax.axhline(_x.min(), color="green", linestyle="--", linewidth=1)
        _tx = _t[len(_t) // 2]
        _ax.annotate(
            "",
            xy=(_tx, _x.max()),
            xytext=(_tx, _x.min()),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5),
        )
        _ax.set_title(
            f"{_label}: peak to peak = {np.ptp(_x):.1f} m/s^2",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.set_ylabel("Acceleration\n(m/s^2)")
        _ax.grid(True, alpha=0.4)
    _axes[-1].set_xlabel("Time in the window (s)")
    _fig.tight_layout()
    _rows.append(
        (
            "Peak to peak",
            _to_img(_fig),
            "The full range of the signal in the window (max − min, between the dashed lines): how violent the strongest movement is.",
        )
    )

    # Dominant frequency
    def _fdom(f, P):
        return f[np.argmax(P[1:]) + 1]  # ignore the constant (0 Hz) component

    _fig, _axes = _two_panels()
    for _ax, _label, _color in zip(_axes, _psd_extremes(_fdom), _pair_colors):
        _f, _P = _window_psd(_label)
        _ax.plot(_f, _P, color=_color, linewidth=1.5)
        _ax.axvline(_fdom(_f, _P), color="red", linestyle="--", linewidth=1.2)
        _ax.set_title(
            f"{_label}: dominant frequency = {_fdom(_f, _P):.2f} Hz",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.set_ylabel("Power spectral\ndensity")
        _ax.grid(True, alpha=0.4)
    _axes[-1].set_xlim(0, 20)
    _axes[-1].set_xlabel("Frequency (Hz)")
    _fig.tight_layout()
    _rows.append(
        (
            "Dominant frequency",
            _to_img(_fig),
            "The highest peak of the spectrum (red line): the main rhythm of the movement.",
        )
    )

    # Spectral centroid
    def _centroid(f, P):
        return np.sum(f * P / P.sum())

    _fig, _axes = _two_panels()
    _cmax = 0
    for _ax, _label, _color in zip(_axes, _psd_extremes(_centroid), _pair_colors):
        _f, _P = _window_psd(_label)
        _c = _centroid(_f, _P)
        _cmax = max(_cmax, _c)
        _ax.plot(_f, _P, color=_color, linewidth=1.5)
        _ax.fill_between(_f, 0, _P, color=_color, alpha=0.25)
        _ax.axvline(_c, color="green", linestyle="--", linewidth=1.5)
        _ax.set_title(
            f"{_label}: spectral centroid = {_c:.2f} Hz",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.set_ylabel("Power spectral\ndensity")
        _ax.grid(True, alpha=0.4)
    _axes[-1].set_xlim(0, max(20, _cmax + 5))
    _axes[-1].set_xlabel("Frequency (Hz)")
    _fig.tight_layout()
    _rows.append(
        (
            "Spectral centroid",
            _to_img(_fig),
            "The balance point of the area under the spectrum (green line): low for slow movements, high for fast or vibrating ones.",
        )
    )

    # Spectral entropy: compare the most rhythmic and the most noisy activity
    def _entropy(label):
        _f, _P = _window_psd(label)
        _Pn = _P / _P.sum()
        return _f, _Pn, -np.sum(_Pn * np.log(_Pn)) / np.log(len(_f))

    _by_entropy = sorted(n_sec_recordings, key=lambda _l: _entropy(_l)[2])
    # One panel per example: the normalised spectrum drawn as bars (share of the power at each
    # frequency) next to a flat spectrum, which is the maximum entropy case (entropy = 1).
    _fig, _axes = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True, sharey=True)
    _cases = [
        (
            _by_entropy[0],
            "tab:blue",
            "LOW entropy: power concentrated\nin a few frequencies (clear rhythm)",
        ),
        (
            _by_entropy[-1],
            "tab:orange",
            "HIGH entropy: power spread over\nmany frequencies (noise-like)",
        ),
    ]
    for _ax, (_label, _color, _text) in zip(_axes, _cases):
        _f, _Pn, _H = _entropy(_label)
        _top3 = np.sort(_Pn)[::-1][:3].sum()
        _ax.bar(_f, _Pn, width=(_f[1] - _f[0]) * 0.9, color=_color, alpha=0.85)
        _ax.axhline(
            1 / len(_f),
            color="black",
            linestyle="--",
            linewidth=1,
            label="flat spectrum (entropy = 1)",
        )
        _ax.set_title(
            f"{_label}: entropy = {_H:.2f}",
            fontsize=10,
            color=_color,
            fontweight="bold",
        )
        _ax.text(
            0.98,
            0.92,
            f"{_text}\ntop 3 frequencies = {100 * _top3:.0f}% of the power",
            transform=_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor=_color
            ),
        )
        _ax.set_ylabel("Share of\nthe power")
        _ax.grid(True, alpha=0.4)
    _axes[0].legend(fontsize=7, loc="center right")
    _axes[-1].set_xlim(0, 20)
    _axes[-1].set_xlabel("Frequency (Hz)")
    _fig.tight_layout()
    _rows.append(
        (
            "Spectral entropy",
            _to_img(_fig),
            "Low when the power is concentrated in a few peaks, high when it is spread like noise.",
        )
    )

    # Band powers: same y-axis in both panels so the amounts can be compared directly
    for _lo, _hi in [(0.5, 3.0), (3.0, 8.0), (8.0, 15.0)]:

        def _band_mask(f, lo=_lo, hi=_hi):
            return (f >= lo) & ((f < hi) if hi < 15 else (f <= hi))

        def _band_power(f, P, lo=_lo, hi=_hi):
            return P[_band_mask(f, lo, hi)].sum()

        # Log scale: the two activities can differ by several orders of magnitude
        _pair = _psd_extremes(_band_power)
        _floor = (
            min(_window_psd(_l)[1][_window_psd(_l)[0] <= 20].min() for _l in _pair) / 2
        )
        _top = (
            max(_window_psd(_l)[1][_window_psd(_l)[0] <= 20].max() for _l in _pair) * 2
        )
        _fig, _axes = _two_panels(sharey=True)
        for _ax, _label, _color in zip(_axes, _pair, _pair_colors):
            _f, _P = _window_psd(_label)
            _ax.plot(_f, _P, color=_color, linewidth=1.5)
            _ax.fill_between(
                _f,
                _floor,
                _P,
                where=_band_mask(_f),
                color=_color,
                alpha=0.45,
                interpolate=True,
            )
            _ax.set_yscale("log")
            _ax.set_ylim(_floor, _top)
            _ax.axvspan(_lo, _hi, color="grey", alpha=0.12)
            _ax.set_title(f"{_label}", fontsize=10, color=_color, fontweight="bold")
            _value_box(
                _ax, f"power {_lo:g}-{_hi:g} Hz = {_band_power(_f, _P):.3g}", _color
            )
            _ax.set_ylabel("Power spectral\ndensity")
            _ax.grid(True, alpha=0.4)
        _axes[-1].set_xlim(0, 20)
        _axes[-1].set_xlabel("Frequency (Hz)")
        _fig.tight_layout()
        _rows.append(
            (
                f"Power between {_lo:g} and {_hi:g} Hz",
                _to_img(_fig),
                "The shaded area under the spectrum inside the grey band: how much of the movement happens at these frequencies (log scale, each grid line is ×10).",
            )
        )

    _table = "".join(
        f"<tr><td style='padding:6px 12px'><b>{_name}</b></td>"
        f"<td style='padding:6px 12px'>{_img}</td>"
        f"<td style='padding:6px 12px;max-width:280px'>{_desc}</td></tr>"
        for _name, _img, _desc in _rows
    )
    mo.Html(
        "<table><thead><tr><th>Feature</th><th>Illustration</th><th>What it shows</th></tr></thead>"
        f"<tbody>{_table}</tbody></table>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Compilation of a single dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All windows from all activities are stacked into a single design matrix `data_input`
    (`n_datapoints x n_features`) with the corresponding integer labels `data_output`.
    """)
    return


@app.cell
def _(features, np):
    data_input = []
    data_output = []
    label_dict = {}
    for _i, _label in enumerate(features.keys()):
        data_input.append(features[_label])
        data_output.append(np.repeat(_i, features[_label].shape[0]))
        label_dict[_i] = _label
    data_input = np.vstack(data_input)
    data_output = np.hstack(data_output)
    print("Shape of input data:", data_input.shape)
    print("Number of classes:", len(label_dict))
    return data_input, data_output, label_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Dimensionality reduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 PCA

    Principal Component Analysis projects the data onto orthogonal directions of maximum
    variance. It is fast and linear, and it tells you how much information (variance) is kept
    when you reduce to two dimensions.
    """)
    return


@app.cell
def _(PCA, StandardScaler, data_input, data_output, label_dict, np, plt):

    scaler = StandardScaler()
    data_input_norm = scaler.fit_transform(data_input)

    pca = PCA(n_components=min(data_input_norm.shape))
    results_pca = pca.fit_transform(data_input_norm)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    print(
        f"The first two components keep {100 * cumulative_variance[1]:.1f}% of the variance."
    )
    print(f"{n95} components are needed to keep 95% of the variance.")

    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))

    fig = plt.figure(figsize=(10, 7))
    _ax = fig.add_subplot(111)

    for _i in range(len(label_dict)):
        _points = results_pca[data_output == _i, :]
        _ax.scatter(
            _points[:, 0],
            _points[:, 1],
            s=12,
            alpha=0.7,
            c=[_colors[_i]],
            label=label_dict[_i],
        )

    _ax.set_title("PCA of the window features")
    _ax.set_xlabel("PC 1")
    _ax.set_ylabel("PC 2")
    _ax.legend()
    fig
    return (data_input_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** How many components do you keep and why? Do the activities form
    separable clusters on the first two components? Which activities overlap?

    *Write your answer here.*
    We kept 2 components because it's the easiest combination to visualize, and it also explains quite a lot of the variance (~70%)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 t-SNE
    """)
    return


@app.cell
def _(TSNE, data_input_norm, data_output, label_dict, np, plt):
    # TODO: try different perplexity value
    perplexity = 10
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
    results_tsne = tsne.fit_transform(data_input_norm)
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))
    # Plot the embedding with one colour for each activity.
    plt.figure(figsize=(10, 6))
    for _i in range(len(label_dict)):
        _points = results_tsne[data_output == _i, :]
        plt.scatter(
            _points[:, 0],
            _points[:, 1],
            s=12,
            alpha=0.7,
            c=[_colors[_i]],
            label=label_dict[_i],
        )
    plt.title(f"t-SNE of the window features (perplexity = {perplexity})")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which perplexity gives the most meaningful visualisation? How does t-SNE
    compare with PCA for your data?

    *Write your answer here.*

    We tried several values for the perplexity, and think that 10 is a good one as it clearly separates all the classes except for the elevator and subway which are at their core quite similar (pretty much nothing hapens except at stop and start)

    Interestingly we see that 4 of the subway points are very close to the violin, this can be explained by the fact that the subway was recorded on 2 stops meaning 2 stop start acceleration and 2 stop stop deceleration (which itself looks quite similar to violin)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 UMAP
    """)
    return


@app.cell
def _(data_input_norm, data_output, label_dict, np, plt):
    # If needed, install UMAP once in a separate cell with: %pip install umap-learn
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Install umap-learn with %pip install umap-learn, then rerun this cell."
        ) from exc
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
    results_umap = reducer.fit_transform(data_input_norm)
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))
    plt.figure(figsize=(10, 6))
    for _i in range(len(label_dict)):
        _points = results_umap[data_output == _i, :]
        # Plot the embedding with one colour for each activity.
        plt.scatter(
            _points[:, 0],
            _points[:, 1],
            s=12,
            alpha=0.7,
            c=[_colors[_i]],
            label=label_dict[_i],
        )
    plt.title("UMAP of the window features")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Compare PCA, t-SNE and UMAP on your data. Which one best separates the
    activities? Why might that be (linear vs non-linear, local vs global structure)?

    *Write your answer here.*

    we clearly see that the linear structure of the PCA makes it hard to separate the subway and elevator classes as well as side steps and running,

    t-SNE seems to be the best overall as even if the elevator and subway are still very close we clearly see 2 different groups (the subway surrounding the elevetor by the outside)

    UMAP seems to be the best regaridng the speration of all the classes except the elevator and subway which are overlapping

    To sum it up:
        - t-SNE seems to be the best if the goal is to direrentiate each class on it's own
        - UMAP seems to be the best if we don't mind the similarities between elevator and subway
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Classification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.0 Train / test split

    Consecutive windows overlap, so a random split would put almost identical windows in both
    sets. Instead, for each activity the **last 20 % of the windows (in time)** form the test
    set, and the windows overlapping the boundary are dropped. Hyper-parameters and the best model
    are selected with a **grouped cross-validation on the training set only** (each fold holds out
    one contiguous block of every activity, and acts as the validation set). The test set is never
    used for any choice, only for the final evaluation.
    """)
    return


@app.cell
def _(GroupKFold, data_input, data_output, np, overlap):
    test_fraction = 0.2
    n_folds = 5
    # number of windows that still overlap with the first test window
    gap_windows = int(np.ceil(overlap / (1 - overlap))) if overlap > 0 else 0

    _train_idx, _test_idx, _groups = [], [], []
    for _cls in np.unique(data_output):
        _idx = np.flatnonzero(
            data_output == _cls
        )  # windows are stored in temporal order
        _n_test = max(1, int(round(test_fraction * len(_idx))))
        _train = _idx[: len(_idx) - _n_test - gap_windows]
        _train_idx.append(_train)
        _test_idx.append(_idx[len(_idx) - _n_test :])
        # contiguous blocks in time -> CV group id (same ids for every class)
        _groups.append(np.arange(len(_train)) * n_folds // len(_train))

    _train_idx = np.concatenate(_train_idx)
    _test_idx = np.concatenate(_test_idx)
    X_train, y_train = data_input[_train_idx], data_output[_train_idx]
    X_test, y_test = data_input[_test_idx], data_output[_test_idx]
    cv_groups = np.concatenate(_groups)
    cv = GroupKFold(n_splits=n_folds)

    print(
        f"Train: {X_train.shape[0]} windows | Test: {X_test.shape[0]} windows | gap: {gap_windows} window(s) per class"
    )
    return X_test, X_train, cv, cv_groups, y_test, y_train


@app.cell
def _(
    GridSearchCV,
    X_test,
    X_train,
    accuracy_score,
    cv,
    cv_groups,
    f1_score,
    np,
    plt,
    y_test,
    y_train,
):
    def _short(key):
        return key.split("__")[-1]

    def plot_search(name, search, x_param=None):
        """CV accuracy (± std) vs one hyper-parameter, one line per combination of the others."""
        res = search.cv_results_
        keys = list(res["params"][0].keys())
        x_param = x_param or keys[0]
        others = [k for k in keys if k != x_param]

        # Hyper-parameter values can be None or tuples -> use categorical x positions
        x_values = list(dict.fromkeys(p[x_param] for p in res["params"]))
        x_pos = {v: i for i, v in enumerate(x_values)}

        lines = {}
        for i, p in enumerate(res["params"]):
            lines.setdefault(tuple(p[k] for k in others), []).append(i)

        # Lines often have identical scores (e.g. max_depth=10/20/None) -> shift each line
        # slightly along x and give it its own colour, marker and line style so all stay visible.
        n_lines = len(lines)
        width = min(0.3, 0.06 * (n_lines - 1)) if n_lines > 1 else 0.0
        offsets = [
            (j / (n_lines - 1) - 0.5) * width if n_lines > 1 else 0.0
            for j in range(n_lines)
        ]
        colors = (
            plt.cm.tab10(range(n_lines))
            if n_lines <= 10
            else plt.cm.viridis(np.linspace(0, 1, n_lines))
        )
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
        linestyles = ["-", "--", "-.", ":"]

        fig, ax = plt.subplots(figsize=(12, 4.5))
        best = search.best_index_
        best_xy = None
        for j, (combo, idx) in enumerate(lines.items()):
            idx = sorted(idx, key=lambda i: x_pos[res["params"][i][x_param]])
            xs = np.array([x_pos[res["params"][i][x_param]] for i in idx]) + offsets[j]
            label = ", ".join(f"{_short(k)}={v}" for k, v in zip(others, combo)) or name
            ax.errorbar(
                xs,
                res["mean_test_score"][idx],
                yerr=res["std_test_score"][idx],
                color=colors[j],
                marker=markers[j % len(markers)],
                linestyle=linestyles[j % len(linestyles)],
                linewidth=1.8,
                markersize=6,
                capsize=3,
                elinewidth=1,
                label=label,
            )
            if best in idx:
                best_xy = (xs[idx.index(best)], res["mean_test_score"][best])

        ax.plot(
            *best_xy,
            marker="*",
            markersize=20,
            color="red",
            markeredgecolor="black",
            linestyle="none",
            zorder=5,
            label="best: "
            + ", ".join(f"{_short(k)}={v}" for k, v in search.best_params_.items()),
        )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_xlabel(_short(x_param))
        ax.set_ylabel("CV accuracy (train set)")
        ax.set_title(f"{name}: cross-validated accuracy vs {_short(x_param)}")
        ax.grid(True)
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # avoid marimo displaying the still-open figure a second time
        return fig

    def tune_and_evaluate(name, estimator, param_grid, x_param=None):
        """Grid search with grouped CV on the training set, then refit and evaluate on the test set.

        Also plots the CV accuracy vs `x_param` (default: the first hyper-parameter of the grid).
        """
        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
        )
        search.fit(X_train, y_train, groups=cv_groups)
        y_pred = search.predict(X_test)
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        return {
            "figure": plot_search(name, search, x_param),
            "model": name,
            "search": search,
            "best_params": search.best_params_,
            "cv_accuracy": search.best_score_,
            "cv_std": cv_std,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_macro_f1": f1_score(y_test, y_pred, average="macro"),
            "y_pred": y_pred,
        }

    return (tune_and_evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 K-Nearest Neighbours (KNN)

    KNN predicts the majority class among the K nearest training points. It is simple and
    interpretable, and the choice of K controls the bias/variance trade-off. Test different values of K and select one giving good performance.
    """)
    return


@app.cell
def _(KNeighborsClassifier, StandardScaler, make_pipeline, tune_and_evaluate):
    knn_result = tune_and_evaluate(
        "KNN",
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": list(range(1, 31))},
    )
    print(f"Best K: {knn_result['best_params']['kneighborsclassifier__n_neighbors']}")
    print(f"CV accuracy: {knn_result['cv_accuracy']:.3f} ± {knn_result['cv_std']:.3f}")
    print(
        f"Test accuracy: {knn_result['test_accuracy']:.3f} | macro F1: {knn_result['test_macro_f1']:.3f}"
    )
    return (knn_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** What is the role of K? What happens with K = 1? With K = (number of
    data points − 1)? What is the potential **methodological problem** of choosing the K (or any parameter)
      that simply gives the best score on the whole dataset? How do you fix it?

    *Write your answer here.*

    K determines the number of neighbors considered to assign a new point class.

    With K = 1 we just look at the class of the closes point to assign a new point class

    With K = number of data points - 1 it just assigns the overall majority class (which is very bad)

    We applied cross validation with different K values ranging from 1 to 30

    For us 3 seemed to be the best
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Random Forest
    """)
    return


@app.cell
def _(RandomForestClassifier, tune_and_evaluate):
    rf_result = tune_and_evaluate(
        "Random Forest",
        RandomForestClassifier(random_state=0),
        {"n_estimators": [50, 100, 200, 400], "max_depth": [None, 5, 10, 20]},
        x_param="n_estimators",
    )
    print(f"Best parameters: {rf_result['best_params']}")
    print(f"CV accuracy: {rf_result['cv_accuracy']:.3f} ± {rf_result['cv_std']:.3f}")
    print(
        f"Test accuracy: {rf_result['test_accuracy']:.3f} | macro F1: {rf_result['test_macro_f1']:.3f}"
    )
    return (rf_result,)


@app.cell
def _(feature_names, np, plt, recordings, rf_result):
    # Replace the generic "chN" suffix with the actual channel name
    _channels = [c.split(" (")[0] for c in next(iter(recordings.values())).columns]
    _pretty_names = np.array(
        [
            n.rsplit("_ch", 1)[0] + " | " + _channels[int(n.rsplit("_ch", 1)[1])]
            for n in feature_names
        ]
    )

    rf_importances = rf_result["search"].best_estimator_.feature_importances_
    _top = np.argsort(rf_importances)[::-1][:20]

    _fig, _ax = plt.subplots(figsize=(10, 7))
    _ax.barh(_pretty_names[_top][::-1], rf_importances[_top][::-1])
    _ax.set_xlabel("Mean decrease in impurity")
    _ax.set_title("Random Forest: top 20 feature importances")
    _ax.grid(True, axis="x")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which features are the most important according to the Random Forest?
    Do they match your intuition from section 3?

    *Write your answer here.*

    Our initial guess was the dominant frequency which appears as the 4th most discriminative feature.

    It seems that the most important ones are actually the different power bands, which makes sens because they caracetrize the signal between different feequency ranges which matches our activities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Artificial Neural Network (MLP)
    """)
    return


@app.cell
def _(MLPClassifier, StandardScaler, make_pipeline, tune_and_evaluate):
    ann_result = tune_and_evaluate(
        "ANN (MLP)",
        make_pipeline(StandardScaler(), MLPClassifier(random_state=0)),
        {
            "mlpclassifier__hidden_layer_sizes": [
                (16,),
                (32,),
                (64,),
                (32, 16),
                (64, 32),
            ],
            "mlpclassifier__alpha": [1e-4, 1e-2, 1e-1],
            "mlpclassifier__max_iter": [500, 2000],
        },
        x_param="mlpclassifier__hidden_layer_sizes",
    )
    print(f"Best parameters: {ann_result['best_params']}")
    print(f"CV accuracy: {ann_result['cv_accuracy']:.3f} ± {ann_result['cv_std']:.3f}")
    print(
        f"Test accuracy: {ann_result['test_accuracy']:.3f} | macro F1: {ann_result['test_macro_f1']:.3f}"
    )
    return (ann_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** How does the ANN compare with KNN and Random Forest? Is a more complex
    model always better on this dataset? Why or why not?

    *Write your answer here.*

    The ANN model is better than the other models regarding the test set, however it's not the case on the validation set.

    We have to be carefule when evaluating a complex model as it tends to overfit on the training data the more complex it gets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Evaluation and comparison
    """)
    return


@app.cell
def _(ann_result, knn_result, mo, rf_result):
    model_results = [knn_result, rf_result, ann_result]

    def _fmt_params(params):
        return ", ".join(f"`{k.split('__')[-1]}={v}`" for k, v in params.items())

    _rows = "\n".join(
        f"| {r['model']} | {_fmt_params(r['best_params'])} "
        f"| {r['cv_accuracy']:.3f} ± {r['cv_std']:.3f} "
        f"| {r['test_accuracy']:.3f} | {r['test_macro_f1']:.3f} |"
        for r in model_results
    )
    # Best model chosen on the CV score (training data only), not on the test set
    best_result = max(model_results, key=lambda r: r["cv_accuracy"])

    comparison_table = (
        "| Model | Best hyper-parameters | CV accuracy (train) | Test accuracy | Test macro F1 |\n"
        "|---|---|---|---|---|\n"
        f"{_rows}\n\n"
        f"Best model (highest CV accuracy): **{best_result['model']}**"
    )
    mo.md(comparison_table)
    return (best_result,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    best_result,
    classification_report,
    confusion_matrix,
    label_dict,
    plt,
    y_test,
):
    _class_names = [label_dict[i] for i in sorted(label_dict)]
    _cm = confusion_matrix(y_test, best_result["y_pred"], labels=sorted(label_dict))
    _fig, _ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(_cm, display_labels=_class_names).plot(
        ax=_ax, xticks_rotation=45, colorbar=False
    )
    _ax.set_title(f"Confusion matrix on the test set — {best_result['model']}")
    plt.tight_layout()
    print(
        classification_report(
            y_test,
            best_result["y_pred"],
            labels=sorted(label_dict),
            target_names=_class_names,
            zero_division=0,
        )
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which activities are frequently confused? Does this match what you saw
    on the PCA / t-SNE / UMAP plots?

    Would the pipeline be usable on a real phone (latency,
    memory, robustness)?

    *Write your answer here.*

    We selected the random forest as our best model since it was the model with the greatest performance on the validation set.

    Surpisingly it's very good at discerning all the classes (even subway and elevator)

    This shows that even if thos 2 activities were always very similar on the different PCA, t-SNE and UMAP plots, a model can still effectivly learn how to differentiate them

    I think it could be deployed for on device inference as random forest isn't very computationally heavy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8) General questions (answer in your own words)

    1. Compare the three classifiers (accuracy, robustness, interpretability, cost).

    |Model|Accuracy|Robustness|Interpretability|Cost|
    | --- | --- | --- | --- | --- |
    | KNN | 0.9 |  +- 0.05    | Easy to interpret | Very low|
    | Random Forest | The best (0.98) | The best (+- 0.018) | The best | Mid|
    | MLP | 0.93 | +- 0.06 | Hard | High |

    3. What are the limitations of this data collection protocol (phone placement, subject,
       environment)? How would you improve it?

       The main issue is that the samepling frequency isn't the same depending of the mobile phone

       The phone also has to be in the same position and place (which can be troublesome dpeending of the acitivity)

       The records have a small amount of noise at the start and end (time to place and collect the phone)

       It could be improved by having a clear SOP (standard operating proceedure) or having profesional sensores
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Write your answers here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus) Feature selection

    If you still have time and motivation, try to improve your pipeline by **adding more
    features** and perform **feature selection**.

    1. **Add features.** Extend your feature set, for example with
       [TSFEL](https://tsfel.readthedocs.io), which can compute
       hundreds of temporal, statistical and spectral features per window and per channel.
    2. **Feature selection.** A large feature set contains many redundant or useless
       features, which makes models slower and can hurt them. Implement a method to keep only the useful ones.

    > **Note** Feature selection must use the **training set only**. If features are
    > chosen with the validation data, the validation score is optimistic (data leakage).
    > Fit the selection on the training windows, then apply it to the validation windows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Discuss in a few lines: Which "feature selection" method(s) did you try ? Did it improve the models ? How many features are really needed for optimal results ? Which ones were selected, and does it match your intuition?

    *Write your answers here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deliverables

    1. This notebook completed (all `TODO` cells filled, all questions answered in your words).
    2. Your `data` folder with your own raw recordings.
    3. Be ready to explain your code, your parameters and your results.
    """)
    return


if __name__ == "__main__":
    app.run()
