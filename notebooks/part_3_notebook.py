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
    # PW - Sleep stage classification with the foundation model SleepFM

    **Goal**: use a *foundation model* (SleepFM) to classify sleep stages, and measure what the pretraining really brings.

    *Context*: during the night, sleep alternates between stages (Wake, light sleep N1-N2, deep sleep N3, REM). Experts label each 30 s epoch of a recording by hand (a long and tedious task), and automating it can be useful, for example to diagnose sleep disorders.

    You will:
    1. compute **embeddings** of 5 nights of sleep with the pretrained **SleepFM** base model,
    2. predict sleep stages **zero-shot** with the authors' pretrained staging head,
    3. evaluate a head **fine-tuned** by the instructor on other HMC nights (starting from the authors' weights),
    4. compare with the **same head trained from scratch** (random initialisation, same nights, same recipe).

    The two heads were trained beforehand (training takes some time and is not repeated here: you only evaluate them). **Your 5 nights were never used to train or to select either of them**,
    so the scores you compute are unbiased.

    | Part | Content |
    |---|---|
    | 0 | Look at the data |
    | 1 | Embeddings with SleepFM |
    | 2 | Zero-shot predictions |
    | 3 | Pretrained head, fine-tuned on HMC |
    | 4 | Head trained from scratch |
    | 5 | Wrap-up |

    Cells marked **TODO** must be completed by you. Questions are in
    **Question** cells: write your answers in the cell below each question.

    *Data: Haaglanden Medisch Centrum (HMC) sleep staging database, CC BY 4.0.*

    *Model: SleepFM (Thapa et al.), CC BY-NC 4.0.*
    **GitHub repository:** https://github.com/zou-group/sleepfm-clinical
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How SleepFM works (read this first)

    A night of sleep is recorded as a **polysomnography (PSG)**: EEG (brain), EOG (eyes), EMG (muscles), ECG (heart), breathing...
    Experts label every 30 s as **Wake, N1, N2, N3 or REM**. This is slow, so we would like a model to help.

    SleepFM has two parts:

    - **Base model** (`checkpoints/model_base`): the *foundation model* (self-supervised, no labels). It reads 5-minute chunks of signal, separately for 4
      modalities (`BAS` = brain+eye, `RESP` = breathing, `EKG` = heart, `EMG` = muscle) and outputs one **embedding**
      (a vector of 128 numbers) per 5 s and per modality. It was pre-trained by the authors in a **self-supervised** way (no sleep-stage labels: the model learns to match the signals of the different modalities recorded at the same time). **We never train it.**
    - **Staging head** (`checkpoints/model_sleep_staging`): a classifier that takes the **embeddings from the base model** as its only input (supervised, with labels).
      It merges the 4 modalities of each 5 s, then looks at the **whole night in order** (so that the context before and after a 5 s epoch helps),
      and outputs a score for each of the 5 stages (Wake, N1, N2, N3, REM) per 5 s.
      It is trained in a **supervised** way, with sleep-stage labels given by experts.

    **Important: SleepFM was not trained on our data.** The base model and the staging head were trained
    by the authors on other, large cohorts (SHHS, MESA, MrOS, SSC). Here we apply it to a **different dataset, HMC**, recorded in another hospital
    with other patients and equipment. So the question of this PW is: does what SleepFM learned **transfer** to new data?

    ```
    raw signals --> [ BASE MODEL ] --> embeddings --> [ STAGING HEAD ] --> sleep stage per 5 s
     (128 Hz)        pretrained on        128 numbers      small network
                     many nights,         per 5 s and      (1.3 M parameters)
                     NO labels            per modality
                     (frozen here)
    ```

    **Metrics.** *Accuracy* is misleading here because N2 is ~40% of the night (unbalanced classes). We use
    - **macro-F1**: F1 score of each of the 5 stages, averaged (rare stages count as much as N2);
    - **Cohen's kappa**: agreement with the expert beyond chance (0 = chance, 1 = perfect; 0.6-0.8 = substantial).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import os
    import sys
    import json
    import subprocess
    import time
    import warnings
    from pathlib import Path

    try:
        import einops  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "einops"])

    # On Google Colab: put the unzipped PW_SleepFM folder on your Drive and adapt the path below.
    COLAB_DIR = "/content/drive/MyDrive/PW_SleepFM"
    if "google.colab" in sys.modules:
        from google.colab import drive
        drive.mount("/content/drive")
        os.chdir(COLAB_DIR)

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score, f1_score

    from sleepfm_models import SetTransformer, SleepEventLSTMClassifier

    warnings.filterwarnings("ignore")
    assert Path("../data/part-3/hdf5").exists(),"Run this notebook from the PW_SleepFM folder (the one containing data/ and checkpoints/)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    DATA = Path("../data/part-3")
    NIGHTS = sorted(p.stem for p in (DATA / "hdf5").glob("*.hdf5"))
    STAGES = ["Wake", "N1", "N2", "N3", "REM"]
    print(len(NIGHTS), "nights:", NIGHTS)
    return (
        ConfusionMatrixDisplay,
        DATA,
        NIGHTS,
        Path,
        STAGES,
        SetTransformer,
        SleepEventLSTMClassifier,
        accuracy_score,
        cohen_kappa_score,
        device,
        f1_score,
        h5py,
        json,
        np,
        os,
        pd,
        plt,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 0 - Look at the data
    `data/hdf5/` has one file per night. The signals were already **preprocessed** (resampled to 128 Hz and standardised, which is what SleepFM expects).
    `data/labels/` has one label per **5 s** window (each 30 s expert label is repeated 6 times).
    """)
    return


@app.cell
def _(DATA, NIGHTS, h5py, np, plt):
    with h5py.File(DATA / 'hdf5' / f'{NIGHTS[0]}.hdf5') as f:
        channels = sorted(f.keys())
        print({c: f[c].shape for c in channels})
        start = 3 * 3600 * 128  # 3 hours after the start of the recording
        _fig, _axes = plt.subplots(len(channels), 1, figsize=(12, 9), sharex=True)
        for _ax, c in zip(_axes, channels):
            _ax.plot(np.arange(30 * 128) / 128, f[c][start:start + 30 * 128].astype(float), lw=0.5)
            _ax.set_ylabel(c, rotation=0, labelpad=45, fontsize=8)
    _axes[-1].set_xlabel('seconds (30 s shown)')
    plt.show()
    return


@app.cell
def _(DATA, NIGHTS, pd):
    labels_df = pd.read_csv(DATA / "labels" / f"{NIGHTS[0]}.csv")
    print(labels_df.head(8))
    dist = pd.DataFrame({n: pd.read_csv(DATA / "labels" / f"{n}.csv")["StageName"].value_counts(normalize=True) for n in NIGHTS})
    print((dist * 100).round(0).reindex(["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 0.** The `RESP_DUMMY` channel is flat. Why? (Hint: which signals did the HMC recording contain?) What does it imply for the model?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1 - Embeddings with the SleepFM base model

    We group the channels into the 4 modalities that SleepFM knows, then give 5-minute chunks (5 min x 60 x 128 Hz = 38 400 samples) to the base model.
    """)
    return


@app.cell
def _(SetTransformer, device, json, torch):
    MODALITIES = {
        "BAS": ["EEG C3-M2", "EEG C4-M1", "EEG F4-M1", "EEG O2-M1", "EOG E1-M2", "EOG E2-M2"],  # brain + eyes
        "RESP": ["RESP_DUMMY"],                                                                  # breathing (not recorded in HMC)
        "EKG": ["ECG"],                                                                          # heart
        "EMG": ["EMG chin"],                                                                     # muscle
    }
    CHUNK = 5 * 60 * 128

    cfg = json.load(open("../checkpoints/model_base/config.json"))
    base_model = SetTransformer(cfg["in_channels"], cfg["patch_size"], cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"],
                                pooling_head=cfg["pooling_head"], dropout=0.0)
    state = torch.load("../checkpoints/model_base/best.pt", map_location="cpu")["state_dict"]
    base_model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    base_model.to(device).eval()
    print(f"SleepFM base model: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f} M parameters")
    return CHUNK, MODALITIES, base_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **TODO 1.** The base model returns **two** things: `(pooled_5min, embeddings_5s)`.
    - `pooled_5min`: one vector for the whole 5-minute chunk, shape `(batch, 128)`;
    - `embeddings_5s`: one vector per 5 s, shape `(batch, 60, 128)`.

    Look at the names and shapes of both outputs, then decide which one to assign to `emb`.
    """)
    return


@app.cell
def _(CHUNK, DATA, MODALITIES, base_model, device, h5py, np, torch):
    def compute_embeddings(night):
        """Return the SleepFM embeddings of a night, shape (4 modalities, n_5s_windows, 128)."""
        with h5py.File(DATA / "hdf5" / f"{night}.hdf5") as f:
            signals = {m: np.stack([f[c][:] for c in chans]).astype(np.float32) for m, chans in MODALITIES.items()}
        n_chunks = signals["BAS"].shape[1] // CHUNK  # the last incomplete chunk is dropped

        all_modalities = []
        for m, sig in signals.items():
            n_ch = sig.shape[0]
            chunks = sig[:, : n_chunks * CHUNK].reshape(n_ch, n_chunks, CHUNK).transpose(1, 0, 2)  # (n_chunks, n_ch, CHUNK)
            out = []
            for batch in torch.from_numpy(chunks).split(16):
                mask = torch.zeros(batch.shape[:2], dtype=torch.bool)  # False = real channel (no padding)
                with torch.no_grad():
                    pooled_5min, embeddings_5s = base_model(batch.to(device), mask.to(device))
                    # TODO 1: select the relevant embeddings
                    emb = embeddings_5s
                out.append(emb.cpu())
            all_modalities.append(torch.cat(out).reshape(-1, 128))  # (n_chunks * 60, 128)
        return torch.stack(all_modalities)

    return (compute_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now compute the embeddings of the 5 nights. They are saved in `embeddings/`, so you can reload them later.
    """)
    return


@app.cell
def _(DATA, NIGHTS, Path, compute_embeddings, np, os, pd, time, torch):
    os.makedirs("embeddings", exist_ok=True)
    EMB, LAB = {}, {}
    t0 = time.time()
    for night in NIGHTS:
        path = Path("embeddings") / f"{night}.npy"
        if path.exists():
            EMB[night] = torch.from_numpy(np.load(path))
        else:
            EMB[night] = compute_embeddings(night)
            np.save(path, EMB[night].numpy())
        # one label per embedding (labels are truncated to the part of the night that was embedded)
        LAB[night] = torch.from_numpy(pd.read_csv(DATA / "labels" / f"{night}.csv")["StageNumber"].to_numpy()[: EMB[night].shape[1]])
        print(night, tuple(EMB[night].shape), f"({time.time() - t0:.0f}s)")
    return EMB, LAB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 1.** For one night the embeddings have shape `(4, n, 128)`. What do the 4, the `n` and the 128 represent? Why choosing embeddings of 5s and not 5 minutes ?
    How many 5 s embeddings would you expect for an 8-hour night? Why is `n` a multiple of 60?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 2 - Zero-shot predictions with the authors' staging head

    **Zero-shot** = use the pretrained staging head **as is**, without training it on other data.

    **Respiration mask.** The head takes a mask of shape `(1, 4, n)`: `1` means "ignore this modality". The RESP modality of HMC is a dummy, so we ignore it.

    **Inside the staging head** (1.3 M parameters, smaller than the 4.8 M of the base model). Input: the embeddings of a whole night, shape `(4 modalities, n steps of 5 s, 128)`. Output: 5 scores per step; the predicted stage is the highest one.

    ```
    4 embeddings of one 5 s step                      (BAS, RESP, EKG, EMG; RESP is ignored here)
            |  1. attention pooling : the modalities "look at each other", then are merged into ONE vector per step
            v
    one vector per 5 s step   --> + position in the night (so the model knows when a step happens)
            |  2. transformer layer : every step looks at ALL other steps of the night (e.g. REM often follows N2/N3)
            v
            |  3. BiLSTM : reads the night forwards and backwards to smooth the context between neighbouring steps
            v
            |  4. linear layer : one vector --> 5 scores (Wake, N1, N2, N3, REM)
            v
    sleep stage per 5 s
    ```

    In short: **combine the modalities at each moment, then use the context of the whole night, then classify.**
    """)
    return


@app.cell
def _(SleepEventLSTMClassifier, device, json, torch):
    def make_mask(n_steps):
        mask = torch.zeros(1, 4, n_steps)
        mask[:, 1] = 1  # modality index 1 = RESP
        return mask.to(device)


    HEAD_PARAMS = json.load(open("checkpoints/model_sleep_staging/config.json"))["model_params"]


    def load_head(path):
        """Build the staging head and load the weights stored in `path`."""
        model = SleepEventLSTMClassifier(**HEAD_PARAMS)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        return model.to(device)


    print(HEAD_PARAMS)
    return load_head, make_mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `predict` runs the head on the embeddings of a night and returns **one predicted stage per 5 s** (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, see `STAGES`).
    - The head returns `(logits, mask)`; `logits` has shape `(1, n, 5)` (5 scores per window).
    - `logits[0]` removes the batch dimension (we feed one night), leaving shape `(n, 5)`.

    **TODO 2.** The predicted stage is the class with the **highest score**. Replace `____` by the code that gives, for each of the `n` windows, the index of its highest score among the 5 (a tensor of shape `(n,)`).
    """)
    return


@app.cell
def _(EMB, device, make_mask, torch):
    def predict(model, night):
        model.eval()
        x = EMB[night][None].to(device)  # (1, 4, n, 128)
        with torch.no_grad():
            logits, _ = model(x, make_mask(x.shape[2]))
            # TODO 2: predicted stage = index of the highest score, for each 5 s window
            pred = logits[0].____(-1)
        return pred.cpu().numpy()

    return (predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helpers to score a model on a list of nights.

    **TODO 3.** In `evaluate`, the **macro-F1** is computed with scikit-learn's `f1_score`. Its `average` argument says how the F1 scores of the 5 stages are combined into one number:
    `"macro"`, `"micro"` or `"weighted"`. Which one matches the F1 given at the top of this notebook (and in the guidelines)?
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    LAB,
    NIGHTS,
    STAGES,
    ____,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    np,
    pd,
    plt,
    predict,
):
    def evaluate(model, nights):
        y = np.concatenate([LAB[n].numpy() for n in nights])
        p = np.concatenate([predict(model, n) for n in nights])
        # TODO 3: choose how the 5 per-stage F1 scores are averaged
        macro_f1 = f1_score(y, p, average=____)
        return {"accuracy": accuracy_score(y, p), "macro_F1": macro_f1, "kappa": cohen_kappa_score(y, p)}, y, p


    def per_night_kappa(model):
        return pd.Series({n: cohen_kappa_score(LAB[n].numpy(), predict(model, n)) for n in NIGHTS})


    def plot_confusion(y, p, title):
        ConfusionMatrixDisplay.from_predictions(y, p, labels=range(5), display_labels=STAGES, normalize="true",
                                                values_format=".2f", cmap="Blues", colorbar=False)
        plt.title(title)
        plt.show()

    return evaluate, per_night_kappa, plot_confusion


@app.cell
def _(NIGHTS, evaluate, load_head, np, plot_confusion):
    zero_shot = load_head("checkpoints/model_sleep_staging/best.pth")
    scores_zs, y_zs, p_zs = evaluate(zero_shot, NIGHTS)
    print("Zero-shot on the 5 nights:", {k: round(v, 3) for k, v in scores_zs.items()})
    print("predicted stages (Wake, N1, N2, N3, REM):", np.bincount(p_zs, minlength=5))
    print("true stages      (Wake, N1, N2, N3, REM):", np.bincount(y_zs, minlength=5))
    plot_confusion(y_zs, p_zs, "Zero-shot (rows = true stage)")
    return scores_zs, zero_shot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 2.** The head was trained on thousands of nights, but works poorly on our nights. Look at the predicted stage counts and at the confusion matrix:
    what is the model doing? Give two possible reasons why a model trained on other cohorts fails on HMC.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 3.** Why is macro-F1 an appropriate metric here, given that sleep-stage classes are imbalanced?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3 - The pretrained head, fine-tuned on HMC

    **Fine-tuning** = continue training the pretrained head on labelled nights of **our** dataset (HMC sleep staging database), starting from the authors' weights.
    The base model stays frozen: only the head (1.3 M parameters) is trained, on precomputed embeddings.

    This was done for you, on **11 other HMC nights**. The **5 validation nights** were used to adjust the learning rate during development and to select the best checkpoint: training ran for 40 epochs, and the checkpoint with the highest validation kappa was saved. The weights are in `checkpoints/finetuned_head.pt`.

    Here you only **evaluate** this head on your 5 nights (test set).
    """)
    return


@app.cell
def _(NIGHTS, evaluate, load_head, per_night_kappa, plot_confusion):
    finetuned = load_head("checkpoints/finetuned_head.pt")
    scores_ft, y_ft, p_ft = evaluate(finetuned, NIGHTS)
    print("Fine-tuned on the 5 nights:", {k: round(v, 3) for k, v in scores_ft.items()})
    print("kappa per night:", per_night_kappa(finetuned).round(2).to_dict())
    plot_confusion(y_ft, p_ft, "Fine-tuned (rows = true stage)")
    return finetuned, scores_ft


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 4.**
    (a) Compare the fine-tuned head with the zero-shot head (kappa and confusion matrix). What changed?
    (b) The weights are the same architecture as the zero-shot head: what does fine-tuning change, and what did it need (data, labels, compute)?
    (c) Which stage is the hardest to recognise? Why do you think so?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 4 - The same head trained from scratch

    Is the **pretraining of the head** useful? The instructor trained exactly the **same architecture, on the same 11 nights, with the same recipe and the same number of epochs**,
    but starting from **random weights** instead of the authors' weights. Its weights are in `checkpoints/random_init_head.pt`.

    **TODO 4.** Load this head (replace `____` by the file name) and evaluate it.
    """)
    return


@app.cell
def _(NIGHTS, evaluate, load_head, per_night_kappa, plot_confusion):
    # TODO 4: load the head trained from scratch
    random_init = load_head("checkpoints/____")
    scores_rand, y_rand, p_rand = evaluate(random_init, NIGHTS)
    print("Random init on the 5 nights:", {k: round(v, 3) for k, v in scores_rand.items()})
    print("kappa per night:", per_night_kappa(random_init).round(2).to_dict())
    plot_confusion(y_rand, p_rand, "Random init (rows = true stage)")
    return random_init, scores_rand


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation of training history of the staging head
    The training curves of both heads were saved during training. Left: the training loss (cross-entropy, lower is better) on the 11 training nights, one value per epoch. Right: Cohen's kappa per epoch (higher is better) for the 11 training nights (dashed, the data the head learns from) and for the 5 validation nights (solid, nights never used to update the weights). Neither curve uses your 5 nights. The validation kappa was used to choose the best epoch (after each epoch, the weights are kept if the validation kappa is the best so far; training runs all 40 epochs, and at the end the best weights are restored), and the saved head is the one from that epoch, not the last one. A large gap between the dashed and solid curves means the head fits the training nights better than new nights (overfitting).
    """)
    return


@app.cell
def _(json, plt):
    history = json.load(open('checkpoints/training_history.json'))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, h in history.items():
        _axes[0].plot(range(1, len(h['train_loss']) + 1), h['train_loss'], label=name)
        line, = _axes[1].plot(range(1, len(h['val_kappa']) + 1), h['val_kappa'], label=f'{name} (validation)')
        _axes[1].plot(range(1, len(h['train_kappa']) + 1), h['train_kappa'], '--', color=line.get_color(), label=f'{name} (train)')
    _axes[0].set_title('training loss')
    _axes[1].set_title('kappa: training nights (dashed) vs 5 validation nights')
    for _ax in _axes:
        _ax.set_xlabel('epoch')
        _ax.grid(alpha=0.3)
    _axes[0].legend()
    _axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 5.** The architecture, the training nights, the embeddings and the recipe are identical. What explains the difference between the two models?
    What does the pretrained head already "know" before it sees the first HMC night? Look at the curves: how do the two models differ during training?

    **Question 6.** In both cases the embeddings come from the SleepFM base model. Does this experiment tell you if the **base model's embeddings** are better than, for instance, a model trained directly on the raw signals? Why (not)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 5 - Wrap-up
    """)
    return


@app.cell
def _(
    display,
    finetuned,
    pd,
    per_night_kappa,
    plt,
    random_init,
    scores_ft,
    scores_rand,
    scores_zs,
    zero_shot,
):
    summary = pd.DataFrame({
        "zero-shot (authors' head)": scores_zs,
        "fine-tuned (pretrained init)": scores_ft,
        "trained from scratch (random init)": scores_rand,
    }).T
    display(summary.round(3))
    summary[["macro_F1", "kappa"]].plot.bar(figsize=(7, 3.5), rot=15, grid=True, title="Heads evaluated on the 5 unseen nights")
    plt.tight_layout()
    plt.show()

    per_night = pd.DataFrame({
        "zero-shot": per_night_kappa(zero_shot),
        "fine-tuned": per_night_kappa(finetuned),
        "random init": per_night_kappa(random_init),
    })
    display(per_night.round(2))
    per_night.plot.bar(figsize=(8, 3.5), grid=True, rot=0, title="Kappa per night")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 7.** Summarise your results in 3-4 sentences: what does pretraining bring here, and what does it cost (compute, data, labels)?

    **Question 8.** Look at the kappa **per night**. How much does it vary from one night to another, and what does this tell you about the reliability of a single number?
    Which weakness of this evaluation remains (think about the number and the origin of the nights: same hospital, same scorers...)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Your answer:*
    """)
    return


if __name__ == "__main__":
    app.run()
