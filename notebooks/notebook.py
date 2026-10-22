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

    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd

    from matplotlib import pyplot as plt

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    # '%matplotlib inline' command supported automatically in marimo
    return PCA, StandardScaler, TSNE, np, os, pd, plt


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
    base_dir = 'data'
    recordings = {}
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and (not f.startswith('.'))]
    print('List of activities:')
    for this_folder in folders:
        print(this_folder)
        data_csv = os.path.join(base_dir, this_folder, 'Raw Data.csv')
        if os.path.isfile(data_csv):
            _data_df = pd.read_csv(data_csv)
            _data_df.set_index('Time (s)', inplace=True)
            recordings[this_folder] = _data_df
        else:  # Check that the file exists before reading it
            print(f"Warning: '{data_csv}' not found. Skipping...")
    labels = list(recordings.keys())
    print('\nChannels:', list(recordings[labels[0]].columns))  # Use the time column as index
    return labels, recordings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation of the dataframe
    """)
    return


@app.cell
def _(labels, recordings):
    print(labels[0] + ':')
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
    activity_of_interest = 'example_activity'  # must be one of `labels`
    _data_df = recordings[activity_of_interest]
    _colors = ['blue', 'orange', 'green', 'red']
    _fig, axs = plt.subplots(len(_data_df.columns), 1, figsize=(15, 8), sharex=True, sharey=True)
    # A colour for each subplot
    for _i, (column, color) in enumerate(zip(_data_df.columns, _colors)):
        axs[_i].plot(_data_df.index, _data_df[column], label=column, color=color)
        axs[_i].set_ylabel('(m/s^2)')
        axs[_i].legend(loc='upper right')
        axs[_i].grid(True)
    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Acceleration vs Time. ' + activity_of_interest)
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
    measurement_of_interest = ''  # one of the channels above
    _fig, axes = plt.subplots(len(recordings), 1, figsize=(15, 2 * len(recordings)), sharex=True, sharey=True)
    for _i, (activity, _data_df) in enumerate(recordings.items()):
        ax = axes[_i]
        ax.plot(_data_df.index, _data_df[measurement_of_interest], color='purple')
        ax.set_title(f'{activity}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s^2)')
        ax.grid(True)
    _fig.suptitle(measurement_of_interest + ' for all activities', fontsize=16)
    plt.tight_layout()
    plt.show()
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
def _(np, recordings):
    # TODO: choose these values and justify your choice in the reflection cell below.
    duration_seconds = 20  # window length in seconds
    overlap = 0.5  # fraction of overlap (0 = no overlap)
    if duration_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError('Choose a positive duration and overlap between 0 and 1.')
    n_sec_recordings = {}
    for _label, df in recordings.items():
        sampling_frequency = 1 / np.mean(np.diff(df.index.values))
        rows_per_window = int(duration_seconds * sampling_frequency)
        step = max(1, int(rows_per_window * (1 - overlap)))
        values = df.values
        starts = range(0, len(values) - rows_per_window + 1, step)
        if rows_per_window < 1 or len(starts) == 0:
            raise ValueError(f'Recording {_label!r} is too short for this window size.')
        n_sec_recordings[_label] = np.stack([values[start:start + rows_per_window] for start in starts], axis=2)
        print(f'{_label}: {n_sec_recordings[_label].shape}')  # Shape: [rows/window, channels, number of windows]
    return (n_sec_recordings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which window length and overlap did you choose, and why? What happens to
    the number of windows when you increase the overlap? What could be the risk of using windows that
    overlap, and what consequence does this have for evaluating your classifier?

    *Write your answer here.*
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
def _(n_sec_recordings):
    features = {}
    for _label in n_sec_recordings.keys():
        X = n_sec_recordings[_label]  # [rows_per_window, n_channels, n_windows]
        features[_label] = None  # --- TODO (you write this) ---  # Create a feature vector for each window by extracting features from every channel.  #  # Choose features that describe each signal window, for example from:  #   - the time domain (statistics of the amplitude values),  #   - the frequency domain (content of the spectrum),  #   - or any other representation you find relevant.  #  # You can either:  #   (a) implement the features yourself with NumPy/SciPy, or  #   (b) use a dedicated library such as TSFEL (https://tsfel.readthedocs.io),  #       e.g. tsfel.get_features_by_domain() + tsfel.time_series_features_extractor(...)  #  # Concatenate the features of all channels so that:  #     features[label].shape == (n_windows, n_features)
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** List the features you implemented. Which ones do you expect to be the
    most discriminative for your activities, and why?

    *Write your answer here.*
    """)
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
    print('Shape of input data:', data_input.shape)
    print('Number of classes:', len(label_dict))
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
    # Fit PCA using the largest valid number of components to inspect the full curve.
    results_pca = pca.fit_transform(data_input_norm)
    plt.figure(figsize=(10, 5))
    plt.bar(x=np.arange(len(pca.explained_variance_ratio_)), height=pca.explained_variance_ratio_)
    plt.title('Explained variance per component')
    plt.ylabel('Variance ratio')
    plt.xlabel('Component')
    plt.grid(axis='y')
    plt.show()
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    print(f'The first two components keep {100 * cumulative_variance[1]:.1f}% of the variance.')
    print(f'{n95} components are needed to keep 95% of the variance.')
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))
    plt.figure(figsize=(10, 6))
    for _i in range(len(label_dict)):
        _points = results_pca[data_output == _i, :]
        plt.scatter(_points[:, 0], _points[:, 1], s=12, alpha=0.7, c=[_colors[_i]], label=label_dict[_i])
    # Plot the first two components, with one colour for each activity.
    plt.title('PCA of the window features')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid()
    plt.show()
    return (data_input_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** How many components do you keep and why? Do the activities form
    separable clusters on the first two components? Which activities overlap?

    *Write your answer here.*
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
    perplexity = 30
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init='pca')
    results_tsne = tsne.fit_transform(data_input_norm)
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))
    # Plot the embedding with one colour for each activity.
    plt.figure(figsize=(10, 6))
    for _i in range(len(label_dict)):
        _points = results_tsne[data_output == _i, :]
        plt.scatter(_points[:, 0], _points[:, 1], s=12, alpha=0.7, c=[_colors[_i]], label=label_dict[_i])
    plt.title(f't-SNE of the window features (perplexity = {perplexity})')
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
        raise ImportError('Install umap-learn with %pip install umap-learn, then rerun this cell.') from exc
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
    results_umap = reducer.fit_transform(data_input_norm)
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(label_dict)))
    plt.figure(figsize=(10, 6))
    for _i in range(len(label_dict)):
        _points = results_umap[data_output == _i, :]
    # Plot the embedding with one colour for each activity.
        plt.scatter(_points[:, 0], _points[:, 1], s=12, alpha=0.7, c=[_colors[_i]], label=label_dict[_i])
    plt.title('UMAP of the window features')
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
    ### 6.1 K-Nearest Neighbours (KNN)

    KNN predicts the majority class among the K nearest training points. It is simple and
    interpretable, and the choice of K controls the bias/variance trade-off. Test different values of K and select one giving good performance.
    """)
    return


@app.cell
def _():
    # --- TODO (you write this) ---
    # 1. Choose a value of K and justify it (try several values).
    # 2. Evaluate your model.
    # 3. Report the accuracy.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** What is the role of K? What happens with K = 1? With K = (number of
    data points − 1)? What is the potential **methodological problem** of choosing the K (or any parameter)
      that simply gives the best score on the whole dataset? How do you fix it?

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Random Forest
    """)
    return


@app.cell
def _():
    # --- TODO (you write this) ---
    # Train a RandomForestClassifier with a clear train / validation split.
    # Tune n_estimators and max_depth. Inspect `.feature_importances_`.
    # model_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which features are the most important according to the Random Forest?
    Do they match your intuition from section 3?

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Artificial Neural Network (MLP)
    """)
    return


@app.cell
def _():
    # --- TODO (you write this) ---
    # Train a small MLPClassifier on the standardised features.
    # Tune hidden_layer_sizes and max_iter. Report cross-validated accuracy.
    # model_ann = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** How does the ANN compare with KNN and Random Forest? Is a more complex
    model always better on this dataset? Why or why not?

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Evaluation and comparison
    """)
    return


@app.cell
def _():
    # --- TODO (you write this) ---
    # Build a table comparing KNN, Random Forest and ANN:
    #     columns = ['model', 'validation accuracy', 'macro F1']
    # For your best model, plot the confusion matrix on the validation set.
    # cm = confusion_matrix(y_val, y_pred)
    # ConfusionMatrixDisplay(cm, display_labels=list(label_dict.values())).plot()
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Your answer.** Which activities are frequently confused? Does this match what you saw
    on the PCA / t-SNE / UMAP plots? Would the pipeline be usable on a real phone (latency,
    memory, robustness)?

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8) General questions (answer in your own words)

    1. Compare the three classifiers (accuracy, robustness, interpretability, cost).
    2. What are the limitations of this data collection protocol (phone placement, subject,
       environment)? How would you improve it?
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
