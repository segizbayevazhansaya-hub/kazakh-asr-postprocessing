# Kazakh ASR post-processing: MAPP and CG-ROVER-CRF

Code, data lists and saved outputs for the paper

> Tukeyev, U.; Tukeyev, U.Segizbayeva, Z. *Post-Processing for Kazakh Speech Recognition: Morphology-Aware Correction (MAPP) and Confidence-Guarded ROVER-CRF Multi-System Selection.* Information 2026 (submitted).

Two post-processing methods are compared on the same 100 Kazakh recordings (4 h 19 min):

* **MAPP** corrects the text of one ASR output (CSE + morphology; BERT + morphology + MaxEnt; BERT + morphology + CRF).
* **CG-ROVER-CRF** aligns five ASR outputs into a ROVER confusion network and selects one candidate per slot with a CRF and a confidence guard τ.

## Layout

```
code/
  asr/            MAPP_ASR_KSC2_chunk_VAD.ipynb       -> ksc2-VAD, ksc2-chunk
                  MAPP_ASR_combined_KKTURBO_VAD.ipynb -> kk-turbo-VAD, kk-turbo-chunk
                  MAPP_ASR_MMS_VAD.ipynb              -> MMS-VAD
  mapp/           MAPP notebooks (variants 1-3)
  cg_rover_crf/   ROVER network, CRF selector, guard, nested cross-validation
  ablations/      vote weights, selector model, label noise, anchor
  rtf/            real-time factor measurement
asr_outputs/      raw text output of each of the five ASR systems (csv: duration_class, file, raw)
original_csv/     original per-file outputs read by the CG-ROVER-CRF notebooks (duration, stem, ref, raw, ...)
results/          ablation tables
recordings.csv    the 100 recordings (file, duration class)
references_all.csv  reference transcripts of the 100 recordings (file, duration_class, text)
```

## Data

The recordings come from openly available online resources. **The audio is not redistributed.** The reference transcripts were prepared by one author and checked by three annotators (each checked all 100 files). The ASR outputs are the raw system texts; all scoring applies the same normalisation to references and outputs (see the paper, Section 3.3).

## Evaluation protocol

`norm_eval` followed by `dedup_full` on reference and hypothesis; pooled WER over 30,363 reference words; paired bootstrap over the 100 files (10,000 resamples, seed 0); file-level 5-fold cross-validation (seed 0) for all learned components; the guard τ is chosen by an inner cross-validation.

## Hardware

Tesla T4 GPU (Google Colab and Kaggle). Models (used as published): `abilmansplus/whisper-turbo-ksc2`, `shyngys879/kazakh-whisper-large-v3-turbo`, `facebook/mms-1b-all` (Kazakh adapter), Whisper large-v3 (general-purpose baseline for MAPP).

## Requirements

Python 3, `jiwer`, `sklearn-crfsuite`, `scikit-learn`, `transformers`, `pandas`, `numpy`. Notebooks were run in Colab/Kaggle; data paths must be adapted.

## Licence

Code: MIT.
