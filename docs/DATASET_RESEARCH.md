# Karate Dataset Research Report

**Scope:** locating labelled karate data usable to fine-tune an existing
OpenPose → BODY_25 → COCO-17 → MMAction2 (ST-GCN / PoseC3D) pipeline.
**Method:** every claim below was checked against the live source (DataCite DOI
API, GitHub API, HuggingFace API, Zenodo API, repository pages). Paper claims of
availability are reported separately from verified availability.
**Date of verification:** September 2026.

---

## 1. Executive Summary

**Direct answer to the central question: No.** There is currently **no publicly
downloadable, labelled karate *video* dataset** of sufficient size to train a
performer-independent karate action recogniser for this pipeline.

What actually exists splits into three groups:

| Group | Downloadable now | Usable for MMAction2 fine-tuning |
|---|---|---|
| **MoCap / marker-based** (Kyokushin, Karate Stances) | ✅ yes, permissive licences | ⚠️ only after 3D→2D projection; no RGB |
| **RGB video with technique labels** (MS-KARD, Karate Technical Action, Kumite 35K) | ❌ none — all paper-only | ✅ ideal, if obtained |
| **Small / adjacent** (Punch_DL) | ✅ yes | ⚠️ boxing punches, 240 samples |

Three findings worth stating plainly:

1. **The best-matched datasets are all gated.** MS-KARD (1,518 clips, 23 karate
   classes), Karate Technical Action (1,847 clips), and Karate Kumite 35K+ are
   exactly what you need and **none has a working download**. MS-KARD's own
   repository page says "The dataset and code will be made publicly available" —
   it has not been.
2. **The best-licensed dataset is the wrong modality.** The Kyokushin dataset is
   **CC0 (public domain), 9.87 GB, 37 athletes, 1,411 recordings, 3,229
   strikes** — genuinely excellent, but it is Vicon marker trajectories, not video.
3. **HuggingFace, Kaggle and Zenodo have essentially nothing.** Searched directly
   via their APIs. The only karate "video" dataset on HuggingFace
   (`thordata/martial-arts-video-v1`) is **five staged period-drama clips** whose
   own README says it is "not intended as a complete martial-arts benchmark".
   Every Zenodo hit for "karate" is Zachary's Karate Club *graph* dataset.

**Realistic path:** author contact for the gated sets, in parallel with
annotating your own tournament footage, using the pipeline you already have to
make annotation cheap (Strategy C + F).

---

## 2. Dataset Table

Availability codes: `DIRECT` = verified working download · `AUTHOR` = contact
required · `PAPER_ONLY` = described in a paper, no repository found ·
`BROKEN` = link exists but unreachable · `UNSUITABLE` = available but wrong content.

| # | Dataset | Availability | Modality | Kumite/Kata | Clips | Subjects | Classes | Temporal ann. | Licence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Kyokushin MoCap** | ✅ `DIRECT` | MoCap C3D (39 markers) | Techniques, air/shield/opponent | 1,411 rec / 3,229 strikes | **37** | 4 techniques | per-repetition | **CC0** |
| 2 | **Karate Stances 2026** | ✅ `DIRECT` | Markerless kinematics + force plates | Stances + gyaku-zuki | 180 trials | 12 | 3 stances | per-trial | **CC-BY 4.0** |
| 3 | **Punch_DL** | ✅ `DIRECT` | Video (Google Drive) + MoveNet COCO-17 | Punches (boxing style) | 240 punches | 24 | 7–8 | per-sample | ❌ **none stated** |
| 4 | **MS-KARD** | ❌ `PAPER_ONLY` | RGB video ×2 cams + 3 IMUs | Techniques | 1,518 | 13 | **23** | ~1 min clips | paper CC-BY-NC; data n/a |
| 5 | **Karate Technical Action** | ❌ `PAPER_ONLY` | RGB video, 8 directions | Techniques | 1,847 | 10 | 10 | ~10 s clips | unknown |
| 6 | **Karate Kumite 35K+** (Bern) | ❌ `PAPER_ONLY` | RGB video, 4 views | **Kumite, scoring** | 35,000+ | 8 | 6 (scoring) | scoring events | unknown |
| 7 | **KUMITRON** | ❌ `PAPER_ONLY` | Video + OpenPose | Ippon Kihon Kumite | n/a | n/a | 5 | frame-level 25 fps | unknown |
| 8 | **MADS** | ⚠️ `BROKEN` + `UNSUITABLE` | Multi-view RGB + depth + MoCap | Kata (6 seq.) | 6 karate seq. | 2 masters | **0 action classes** | ❌ none | unknown |
| 9 | **Infomus / Kolykhalova** | ❌ `AUTHOR` | MoCap + video + audio | Kata (Heian Yondan, Bassai Dai) | 42 rec. | 7 | quality, not classes | partial | unknown |
| 10 | `thordata/martial-arts-video-v1` | ✅ `DIRECT` but `UNSUITABLE` | RGB video | Neither — staged drama | **5** | n/a | coarse tags | "not frame-accurate" | none stated |
| 11 | `Reuphrosyne/Karate-Kata-Assessment-Dataset` | ⚠️ **EMPTY REPO** | — | — | 0 | — | — | — | none |
| 12 | Kaggle *Martial Arts Training Sensor* | ✅ `DIRECT` but `UNSUITABLE` | IMU sensors only | — | — | — | — | — | Kaggle terms |

---

## 3. Kumite Datasets — Detailed

### 3.1 Karate Kumite 35K+ (University of Bern) — the best match, unobtainable

Described in a BSc thesis hosted at
`prg.inf.unibe.ch/wp-content/uploads/2025/03/BA_AlessioPetrone.pdf`.

| | |
|---|---|
| Size | 35,000+ videos |
| Views | 4 synchronised perspectives |
| Athletes | 8 |
| Classes | `1 point`, `2 points`, `3 points`, `no point`, `nothing happening`, `Jogai` |
| Ground truth | derived from all 4 views to emulate the refereeing process |

**Task type: scoring-event detection, NOT technique classification.** The labels
are referee decisions (WKF *yuko* / *waza-ari* / *ippon*), not *gyaku-zuki* vs
*mawashi-geri*. If you obtained it you could train a "was that a point?" model —
a different and arguably harder problem than naming the technique.

**Verification:** no GitHub repository, no DOI, no data page found. GitHub search
for `kumite karate` returns only scoreboard apps. `PAPER_ONLY`. Only 8 athletes
also makes performer-independent evaluation weak (leave-one-out over 8).

### 3.2 MS-KARD — 23 classes, closest to fine-grained technique recognition

| | |
|---|---|
| Clips | 1,518 |
| Classes | 23 (e.g. *Jodan Zuki*, *Oi Zuki*) |
| Karateka | 13 |
| Capture | 2 orthogonal cameras + 3 wearable IMUs |
| Rate / length | 30 fps, ~1 min per video |
| Volume | 2,814,930 frames; 5,623,734 sensor samples |

**Verified:** the Bournemouth repository page (`eprints.bournemouth.ac.uk/36996/`)
carries **only the PDF**. No DOI, no GitHub, no data link. The abstract's "The
dataset and code will be made publicly available" is unfulfilled. The page states
access is "only for research, scholarly or other non-commercial purposes".

**Task type: video-level classification** (one technique per ~1 min clip). Would
need re-segmentation for temporal localisation.

### 3.3 Karate Technical Action Dataset — best class balance

| | |
|---|---|
| Clips | 1,847 (696 kick / 625 punch / 526 moving) |
| Athletes | 10 (Hunan Karate Team) |
| Capture | 8 directions, ~10 s clips |
| Split used in paper | train 1,786 / test 61 |

Techniques listed: Middle Punch, Upper Punch, Forward Upper Punch, Back Upper
Punch, Back Pounce, Side Rising Kick, Back Kick, Inside Crescent Kick, Side Kick,
Movement.

**Verified:** MDPI page returns HTTP 403 to automated fetch; no data repository
or GitHub found by search. `PAPER_ONLY`. Note the paper's own test split is
**61 clips** — far too small for reliable per-class metrics, and probably not
performer-independent.

### 3.4 KUMITRON

Ippon Kihon Kumite, classes `Kamae`, `Gedan Barai`, `Oi Tsuki`, `Soto Uke`,
`Gyaku Tsuki`, frame-level at 25 fps, built on OpenPose — architecturally the
closest to your pipeline. The paper references `github.com/Kumitron`.

**Verified:** no such GitHub organisation or repository surfaced in search.
`PAPER_ONLY` / possibly `BROKEN`.

---

## 4. Kata Datasets — Detailed

Kata datasets are fewer but structurally easier (single performer, no occlusion,
no tracker problems).

### 4.1 MADS — must be corrected

Commonly cited as a karate dataset. It is **not an action-recognition dataset**:

- 5 categories (Tai-chi, Karate, Jazz, Hip-hop, Sports), **6 sequences each**
- ~53,000 frames **across all five categories**
- Ground truth = **3D joint coordinates** from optical MoCap at 60 Hz
- Purpose: benchmark for **3D human pose estimation**, i.e. "how accurate is your
  pose estimator on hard motions"

**It contains zero action class labels.** With 6 karate sequences it could not
train a classifier even if it did.

**Verified:** `visal.cs.cityu.edu.hk` currently returns `ECONNREFUSED`. `BROKEN`.

*Possible* use: evaluating OpenPose's own accuracy on martial-arts motion, or
2D→3D lifting research. Not for your classifier.

### 4.2 Infomus / Kolykhalova multimodal kata dataset

Heian Yondan and Bassai Dai, 7 participants at mixed skill levels, 42 recordings,
synchronised MoCap (25 markers, 250 Hz) + video (1024×576, 50 fps) + audio.

Annotations concern **movement quality**, not action class. Suited to kata
*scoring/quality assessment*, which is a regression/ranking problem, not
classification. `AUTHOR` contact required.

---

## 5. Pose / Skeleton / MoCap Datasets (separate from RGB)

### 5.1 Kyokushin MoCap — the best-licensed karate data that exists

| | Verified value |
|---|---|
| DOI | `10.6084/m9.figshare.12315629` |
| Size | **9,865,898,389 bytes (9.87 GB)** |
| Licence | **CC0 1.0 Universal (public domain — commercial use permitted)** |
| Athletes | 37 (beginner → advanced) |
| Recordings | 1,411 |
| Individual strikes | 3,229 |
| Capture | Vicon optical MoCap, 39 reflective markers, Plug-In Gait |
| Techniques | *Gyaku-Zuki*, *Mae-Geri*, *Mawashi-Geri*, *Ushiro-Mawashi-Geri* |
| Conditions | in air / against shield / against opponent |
| Paper | `10.1038/s41597-021-00801-5` (Scientific Data, 2021) |

**This is the only karate dataset with both a permissive licence and enough
athletes (37) for genuine leave-one-athlete-out evaluation.**

**But it is not RGB video.** To use it with your 2D pipeline you must project the
3D marker trajectories to 2D under synthetic camera poses, then map the 39-marker
Plug-In Gait set onto COCO-17. That is a real but tractable engineering task, and
it lets you *augment* by rendering many virtual viewpoints from one recording.

A third-party pipeline already exists for this data:
`github.com/NoeMartinezSanchez/kyokushin-c3d-analysis` — C3D segmentation,
repetition detection, per-execution biomechanical features, and explicitly
**GroupKFold/LOSO validation by participant**. Worth reading before starting.

### 5.2 Karate Stances 2026

| | Verified value |
|---|---|
| DOI | `10.6084/m9.figshare.32288943` |
| Size | **1,204,800,735 bytes (1.20 GB)** |
| Licence | **CC-BY 4.0** |
| Year | 2026 |
| Subjects | 12 karateka, 180 trials |
| Content | 3 fundamental stances + *gyaku-tsuki*; markerless kinematics + ground reaction forces |
| Code | `github.com/luiz-pavani/karate-stances-dataset-scripts` (MIT, updated 2026-09-07) |

Markerless capture means it is closer to your modality than Vicon data. Only 3
stances plus one punch, so it cannot carry a technique taxonomy on its own — but
it is current, maintained and cleanly licensed.

### 5.3 Punch_DL

`github.com/balezz/Punch_DL` — **verified live**, 82 MB repo, 20 stars, last push
2022-05-17, **no licence file** (default: all rights reserved — do not assume
reuse rights).

240 punches from 24 individuals (10 weak + 10 strong each). Classes are **boxing**
terminology: `no punch`, `jab`, `cross`, `left/right hook`, `left/right uppercut`,
plus strong variants. Videos live on Google Drive, not in the repo. Keypoints
extracted with **MoveNet**, which outputs **17 COCO keypoints — the same layout
your pipeline already targets**, so no conversion is needed.

Too small and too boxing-specific for a karate model, but ideal as an end-to-end
*dry run* of your training and evaluation code.

---

## 6. Verified Direct Download Links

Only these were confirmed reachable and correctly described:

1. **Kyokushin MoCap** — https://doi.org/10.6084/m9.figshare.12315629 (CC0, 9.87 GB)
2. **Karate Stances 2026** — https://doi.org/10.6084/m9.figshare.32288943 (CC-BY 4.0, 1.20 GB)
3. **Karate Stances code** — https://github.com/luiz-pavani/karate-stances-dataset-scripts (MIT)
4. **Punch_DL** — https://github.com/balezz/Punch_DL (no licence stated)
5. **Kyokushin C3D analysis pipeline** — https://github.com/NoeMartinezSanchez/kyokushin-c3d-analysis

No verified download link exists for MS-KARD, Karate Technical Action,
Karate Kumite 35K+, KUMITRON, or MADS.

---

## 7. Paper-Only / Gated — How to Request

Send a short, specific request. Researchers respond far better to a concrete
technical ask than to "can I have your dataset".

**MS-KARD** — Yadav, Deshmukh, Gonela, Kera, Tiwari, Pandey, Akbar.
Repository contact `BURO@bournemouth.ac.uk`; better to email the corresponding
author directly from the IEEE/PDF version.
Ask for: the 1,518 clips, the 23-class label file, the subject IDs (needed for
performer-independent splits), and the camera calibration if available. State
explicitly that use is **non-commercial academic research** — their page already
restricts to that.

**Karate Technical Action** — corresponding author of
*Applied Sciences* 11(18):8641.
Ask for: the 1,847 clips with per-clip technique labels and the athlete IDs for
the 10 Hunan Karate Team members.

**Karate Kumite 35K+** — Alessio Petrone, supervised at the Software Composition
Group / PRG, University of Bern.
Ask for: the scoring annotations and whether the 4-view raw footage can be shared
or whether only extracted features/poses are releasable (broadcast footage often
cannot be redistributed — expect this to be the blocker).

**KUMITRON** — UNED (PHYUM project). Ask whether the `github.com/Kumitron`
repository was renamed or archived.

**Infomus kata dataset** — Kolykhalova et al., InfoMus / Casa Paganini, Genoa.

In every case ask for **athlete/performer identifiers**. Without them you cannot
build a leakage-free split, and a dataset without them is substantially less
useful.

---

## 8. Licensing

| Dataset | Licence (verified) | Commercial use |
|---|---|---|
| Kyokushin MoCap | **CC0 1.0** | ✅ permitted |
| Karate Stances 2026 | **CC-BY 4.0** | ✅ with attribution |
| Karate Stances scripts | **MIT** | ✅ |
| Punch_DL | **none stated** | ❌ assume all rights reserved |
| MS-KARD | paper CC-BY-NC; page says research/non-commercial only | ❌ |
| Karate Technical Action | unknown | ❌ assume not |
| Karate Kumite 35K+ | unknown; likely broadcast-derived | ❌ assume not |
| MADS | unknown | ❌ assume not |
| `thordata/martial-arts-video-v1` | none stated | ❌ |

**Separately: your own tournament footage.** Competition video is frequently
owned by the federation or broadcaster. Recording at a tournament, or using
footage you did not shoot, may not grant redistribution rights even if you may
analyse it privately. If a commercial product is the goal, get written permission
from the federation — this is a real risk, not a formality.

---

## 9. MMAction2 Compatibility and Required Conversion

Your annotation target:

```python
{'frame_dir': str, 'label': int,
 'keypoint': ndarray(M, T, 17, 2), 'keypoint_score': ndarray(M, T, 17),
 'img_shape': (h, w), 'total_frames': T}
```

| Dataset | Conversion required | ST-GCN | PoseC3D | RGB models | TAL |
|---|---|:--:|:--:|:--:|:--:|
| **Kyokushin MoCap** | 3D C3D → project to 2D under synthetic cameras → map 39 Plug-In Gait markers → COCO-17; `keypoint_score` = 1.0 (or simulate dropout); segment per repetition | ✅ | ⚠️ synthetic heatmaps lack real jitter | ❌ no RGB | ⚠️ if repetition boundaries kept |
| **Karate Stances 2026** | same projection route; markerless so joints are closer to COCO already | ✅ | ⚠️ | ❌ | ❌ |
| **Punch_DL** | already MoveNet **COCO-17** → reshape to (M=1, T, 17, 2); scores available | ✅ | ✅ | ✅ | ❌ |
| **MS-KARD** *(if obtained)* | run your OpenPose pipeline over the clips → your existing converter handles the rest | ✅ | ✅ | ✅ | ⚠️ needs re-segmentation |
| **Karate Technical Action** *(if obtained)* | same — straight through your pipeline | ✅ | ✅ | ✅ | ❌ clip-level only |
| **Kumite 35K+** *(if obtained)* | same, plus scoring events → AVA-style or TAL annotation | ✅ | ✅ | ✅ | ✅ best candidate |

**Important caveat on synthetic 2D from MoCap:** projected marker data is
*too clean*. It has no detector noise, no missing joints, no motion blur, none of
the `(0,0,0)` dropouts OpenPose produces constantly. A model trained purely on it
will degrade on your real footage. Mitigate by injecting realistic corruption:
random joint dropout, confidence jitter, coordinate noise, and occasional whole-
limb loss, calibrated against the confidence distribution your own pipeline emits.

**On MMAction2 task support:**
- `PoseDataset` + ST-GCN/PoseC3D → **trimmed clip classification**. This is what
  your current pipeline does per window.
- **Temporal Action Detection** (BMN/BSN etc. in MMAction2) operates on RGB/flow
  features, not skeletons. For skeleton-based *localisation*, the practical route
  is what you already built: dense sliding-window classification plus
  post-processing (smoothing, non-maximum suppression over time). That is a
  legitimate and common approach.
- **AVA-style spatio-temporal detection** fits scoring events (which athlete, when)
  but requires per-frame boxes plus action labels — considerably more annotation.

---

## 10. Strategy Comparison

| | **A** Single public | **B** Combine small sets | **C** Public pretrain + own annotation | **D** Fully custom | **E** Split: technique vs scoring | **F** Pseudo-label / active learning |
|---|---|---|---|---|---|---|
| Data quantity | ~0 usable video | 1.4 GB MoCap + 240 punches | your footage, unbounded | unbounded | unbounded | same as C/D but labelled faster |
| Label quality | n/a | high (lab) | high | high if careful | high | **variable — needs audit** |
| Domain similarity | n/a | ❌ lab ≠ tournament | ✅ identical | ✅ identical | ✅ | ✅ |
| Annotation effort | none | low | **high** | **very high** | very high | **medium** |
| Technical difficulty | — | high (3D→2D, taxonomy merge) | medium | medium | high | high |
| Overfitting risk | — | high (few subjects) | medium | medium | medium | **high if labels unaudited** |
| Athlete-leakage risk | — | low (IDs present) | **depends on your discipline** | depends | depends | depends |
| Licensing risk | — | low (CC0/CC-BY) | **footage rights unclear** | same | same | same |
| Fits RTX 3050 4 GB | — | ✅ ST-GCN tiny | ✅ | ✅ | ⚠️ AVA-style is heavy | ✅ |
| Role | not viable alone | **pretraining only** | **primary** | primary, slowest | **correct long-term split** | **accelerator for C** |

**Factual notes rather than a ranking:**

- **A is not viable.** No single public dataset can do the job.
- **B cannot stand alone** — combining Kyokushin (4 techniques, MoCap) with
  Punch_DL (boxing, 240 samples) yields a classifier whose classes do not match
  karate competition and whose domain is a laboratory.
- **C and F compose well.** Pretrain on projected MoCap for motion priors,
  fine-tune on your annotated tournament footage, use the existing pipeline's
  window predictions as *candidate proposals* so an annotator confirms/corrects
  rather than scrubbing blind.
- **E reflects a real distinction.** "Which technique" and "did it score" are
  different problems with different labels and probably different models. WKF
  scoring depends on control, timing, target area and referee judgement — not on
  the technique alone. Merging them into one label set will hurt both.
- **F needs an audit protocol.** Pseudo-labels from an NTU60 model are weak; use
  them only to *propose* segments for human review, never as training labels
  directly. Measure annotator agreement on a held-out subset.

---

## 11. Recommended Label Taxonomy

Use **WKF competition terminology**, not generic labels — it is what your domain
experts already use and what any future dataset will align to.

### Stage 1 — coarse (start here)

```
NO_ACTION | MOVEMENT | PUNCH (tsuki) | KICK (geri) | BLOCK (uke) | SWEEP (ashi-barai) | CLINCH
```

7 classes. Reliable to annotate, tolerant of the tracker's weaknesses, and
`PUNCH`/`KICK` have *partial* support from existing models for proposal generation.

### Stage 2 — fine-grained technique

```
Tsuki:  KIZAMI_ZUKI · GYAKU_ZUKI · OI_ZUKI · URA_ZUKI
Uchi:   URAKEN · SHUTO_UCHI
Geri:   MAE_GERI · MAWASHI_GERI · URA_MAWASHI_GERI · YOKO_GERI · USHIRO_GERI
Uke:    GEDAN_BARAI · AGE_UKE · SOTO_UKE · UCHI_UKE · SHUTO_UKE
```

### Stage 3 — scoring events (separate model)

```
YUKO (1 pt) · WAZA_ARI (2 pts) · IPPON (3 pts) · NO_SCORE · JOGAI · penalties
```

### Which labels have existing data

| Label group | Existing support |
|---|---|
| `GYAKU_ZUKI`, `MAE_GERI`, `MAWASHI_GERI`, `USHIRO_MAWASHI_GERI` | ✅ **Kyokushin MoCap, 37 athletes** — the only fine-grained classes with real public data |
| Stances (zenkutsu/kokutsu/kiba-dachi) | ✅ Karate Stances 2026 |
| `GEDAN_BARAI`, `SOTO_UKE`, `UCHI_UKE`, `OI_TSUKI` | ⚠️ KUMITRON / Oyama — gated |
| Everything else, and all scoring events | ❌ **custom annotation required** |

Note the overlap: Kyokushin's four techniques map cleanly onto Stage 2 and are
CC0. That is your only free fine-grained foothold.

---

## 12. Custom Annotation Plan (MVP sizing)

For a **Stage 1, 7-class** model with credible performer-independent evaluation:

| Quantity | Target | Reasoning |
|---|---|---|
| Athletes | **≥ 20**, ideally 30+ | need ≥ 5 held out; 8 athletes (as in Kumite 35K) is too few |
| Matches | 30–50 | ~2 min of action each |
| Clips per class | **200–500** | below ~150 per class, per-class recall is unstable |
| Total clips | ~2,000–3,500 | 7 classes × ~300, plus `NO_ACTION`/`MOVEMENT` which accumulate fast |
| Clip length | 1–2 s | matches the measured 1-second optimum |
| Tournaments | **≥ 3 different events** | otherwise the model learns the venue, lighting and camera |

**Annotation throughput estimate:** with the pipeline pre-proposing candidate
windows (Strategy F), expect roughly **3–6× realtime** for a trained annotator —
so a 2-minute match takes 6–12 minutes. 40 matches ≈ **5–8 hours** of focused
work for Stage 1. Stage 2 (fine-grained) is considerably slower and needs a
karate expert, not a general annotator.

**Do not skip:** double-annotate ~10% and compute Cohen's κ. If κ < 0.7 on
Stage 1 labels, the taxonomy is ambiguous and must be fixed before scaling.

---

## 13. Evaluation Protocol — Avoiding Leakage

Five leakage routes, all of which inflate accuracy:

| Leakage | Cause | Prevention |
|---|---|---|
| **Athlete** | same fighter in train and test | **group split by athlete ID**; leave-one-athlete-out if ≤ 30 athletes |
| **Video** | clips from one match in both splits | split at **match** level, never clip level |
| **Near-duplicate** | overlapping sliding windows land in both splits | split **before** windowing, not after |
| **Camera** | one venue's angle/lighting memorised | hold out **whole tournaments** |
| **Tournament** | same event in both splits | at least one event exclusively in test |

**The strict rule:** an athlete, a match, a camera and an event must each appear
in exactly one split.

**Metrics:**
- Per-class **precision / recall / F1** — accuracy alone hides the fact that
  `NO_ACTION` and `MOVEMENT` dominate the class distribution
- **Balanced accuracy** and a **confusion matrix** (expect *kizami-zuki* ↔
  *gyaku-zuki* confusion; that is the interesting number)
- Top-1; top-5 only once Stage 2 has ≥ 10 classes
- For localisation: **temporal IoU, mAP@tIoU={0.3,0.5,0.7}**
- For scoring: **event-level precision/recall** with a tolerance window
  (e.g. ±0.5 s), plus per-referee-decision agreement

Report the **number of held-out athletes** alongside every metric. An F1 from
2 test athletes is an anecdote.

---

## 14. Exact Next Steps (2–4 weeks)

**Week 1 — secure what is free, and ask for what is not**
1. Download Kyokushin MoCap (CC0, 9.87 GB → put it on D:) and Karate Stances (1.2 GB).
2. Send the four author emails (§7). These have multi-week latency — send them first.
3. Clone `kyokushin-c3d-analysis` and read its LOSO validation approach.

**Week 2 — prove the training loop on data you already have**
4. Build the C3D → 2D projection + 39-marker → COCO-17 mapping.
5. Build the `PoseDataset` pickle writer.
6. Fine-tune ST-GCN on Kyokushin's **4 techniques** with leave-one-athlete-out
   over 37 athletes. *This is the key milestone:* it validates the entire training
   and evaluation chain on real karate labels, before any annotation effort.
7. Run the same loop on Punch_DL as a second sanity check (already COCO-17).

**Week 3 — annotation infrastructure**
8. Build the annotation UI: scrub video, mark `[start_s, end_s] → label`, persist.
9. Wire the existing sliding-window output in as **candidate proposals**.
10. Annotate one full match. Measure actual throughput. Recalibrate §12 from the
    real number, not the estimate.

**Week 4 — scale and measure honestly**
11. Annotate to ~300 clips/class for Stage 1 across ≥ 3 tournaments.
12. Fine-tune from the NTU60 checkpoint (`load_from`), grouped split by athlete.
13. Report per-class F1 + confusion matrix + number of held-out athletes.
14. Only after step 13 may the system be described as recognising karate.

**Parallel track:** upgrade the tracker (ByteTrack/OC-SORT) before serious kumite
annotation. Identity swaps during clinch corrupt the skeleton sequences you are
about to spend weeks labelling.

---

## 15. Sources

- MS-KARD paper — https://eprints.bournemouth.ac.uk/36996/1/KarateNet_WCCI.pdf
- MS-KARD repository page (verified: no data link) — https://eprints.bournemouth.ac.uk/36996/
- MS-KARD IEEE — https://ieeexplore.ieee.org/document/9892646/
- Karate Kumite 35K+ thesis — https://prg.inf.unibe.ch/wp-content/uploads/2025/03/BA_AlessioPetrone.pdf
- Karate Technical Action — https://www.mdpi.com/2076-3417/11/18/8641
- KUMITRON — https://www.mdpi.com/1424-8220/21/24/8378
- Kumite preliminary-action detection — https://www.mdpi.com/1424-8220/25/13/4134
- Kyokushin dataset paper — https://www.nature.com/articles/s41597-021-00801-5
- **Kyokushin data (CC0, 9.87 GB)** — https://doi.org/10.6084/m9.figshare.12315629
- **Karate Stances 2026 (CC-BY 4.0, 1.20 GB)** — https://doi.org/10.6084/m9.figshare.32288943
- Karate Stances scripts (MIT) — https://github.com/luiz-pavani/karate-stances-dataset-scripts
- Kyokushin C3D analysis pipeline — https://github.com/NoeMartinezSanchez/kyokushin-c3d-analysis
- Punch_DL — https://github.com/balezz/Punch_DL
- MADS project page (verified unreachable) — http://visal.cs.cityu.edu.hk/research/mads/
- MADS paper — https://dl.acm.org/doi/abs/10.1016/j.imavis.2017.02.002
- Oyama karate recognition — https://doi.org/10.3390/sym7041670
- Infomus multimodal kata dataset — https://eudl.eu/doi/10.4108/icst.intetain.2015.260039
- AI in martial arts survey — https://journals.sagepub.com/doi/abs/10.1177/17543371241273827
- `thordata/martial-arts-video-v1` — https://huggingface.co/datasets/thordata/martial-arts-video-v1
- Kaggle Martial Arts Training Sensor — https://www.kaggle.com/datasets/ziya07/martial-arts-training-sensor-dataset
