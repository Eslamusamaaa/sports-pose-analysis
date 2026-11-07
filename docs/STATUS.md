# Project status — what is finished, exactly

## Overview
One project that analyses karate kumite and swimming from video with a shared pipeline:
`video → pose estimation → player tracking → BODY_25 → COCO-17 → classifier (ST-GCN / PoseC3D) or VLM → report`.

## ✅ Finished and verified
1. **Full web app** (React + FastAPI): upload a video → OpenPose → tracking → classification → results in the browser, plus a **live webcam mode** (~10 fps) and a browser for previous jobs.
2. **GPU OpenPose** without the CUDA toolkit: 0.071 s/frame (30× faster than CPU). A 4,865-frame 1080p video takes about 18.5 min.
3. **BODY_25 → COCO-17 conversion** (subset + reorder, no interpolation, 8 joints dropped).
4. **Kumite player tracking**: ByteTrack adapted to skeletons; the two players chosen by hand/foot activity (coach and spectators excluded); Player 1/2 fixed to the same person for the whole video; clinch swaps repaired with a Viterbi pass; OSNet re-ID.
5. **Alternative pose engines** (RTMPose, YOLO11-pose) compared on the same clips: kumite 100 % detection for both; swimming 69 % (RTMPose) / 34 % (YOLO11) / 25 % (OpenPose).
6. **Sliding-window classification** (1 s is the optimum: 6 strikes found vs 1 with a 3.3 s window and 0 with 0.4 s) and a multi-model ensemble.
7. **Swimming model trained** (ST-GCN, 4 strokes: freestyle / backstroke / breaststroke / butterfly):
   - data: synthetic **SwimXYZ** (16,647 train / 4,197 validation windows);
   - leakage-aware split by underlying motion, not by swimmer;
   - training-time corruption (`SwimSkeletonAug`) so clean joints resemble OpenPose output;
   - result: **99.93 % top-1** on held-out *synthetic* motions (epoch 9). File: `app/models/swim/stgcn_swim4.pth`.
8. **Dataset research** (`DATASET_RESEARCH.md`): there is no downloadable labelled karate video dataset; lists what is usable (Kyokushin MoCap CC0, Karate Stances, Punch_DL), what is gated (MS-KARD, Karate Technical Action, Kumite 35K+), whom to contact, a label taxonomy, annotation plan and a leakage-prevention protocol.
9. **Karate VLM prototype** (`karate_vlm/`): three models (Qwen3-VL-4B, Qwen3.5-4B, Gemma 3 4B) run on 138 windows of one match with 12 WKF labels, plus HTML reports and a comparison page.

## ⚠️ Not finished / not proven
- **There is no trained karate model.** The wired-in models (NTU RGB+D 60) do not know karate and only output "punching other person"-style labels. HMDB/UCF/K400 were tried and removed because they returned random labels.
- **Swimming accuracy on real video is unknown** — the 99.93 % is on synthetic data only. A few short phone clips were run without labels.
- **The VLM has no measured accuracy** (no ground truth). The three models disagree strongly (e.g. Gemma 3 labels 41 windows `MAWASHI_GERI_CHUDAN`, Qwen3.5 labels 13), so it is only useful as a *proposal generator* for human review.
- Not started: annotation tool, karate `PoseDataset` builder, Kyokushin MoCap → 2D conversion.
- `evaluate.py` (per-view accuracy + confusion matrix for swimming) is ready, but no saved report exists, so it was most likely never run. Run it and commit `swimming/reports/eval_report.json`.

## Videos that were analysed
- **Karate (7 videos):** `Match 3.mov` (162 s), `video1` (145 s), `video2` (111 s), `video3` (107 s), `video4` (162 s), `video5` (121 s), `kumite_20s` — all 1920×1080 (≈ 47 jobs, mostly repeats of the same videos).
- **Swimming:** 5 short phone clips (9–16 s) plus `swim_front.mp4`.
- **Smoke test:** `demo.mp4` (10 s).
- The source videos are not in the repository (usage rights unclear).

## Approaches explored
| Approach | State |
|---|---|
| Pretrained skeleton model (NTU60) | works technically, out of domain |
| Zero-shot VLM | prototype, needs human review |
| Simulated data (SwimXYZ) | done for swimming; proposed for karate via MoCap |
| Data acquisition | full research and contact plan; no karate video data obtained |
| Analysis pipeline | tracking, player identity, timeline, ensemble |

Details: [`KARATE_APPROACHES.md`](KARATE_APPROACHES.md).

## Next steps (in order)
1. Annotation tool + VLM proposals confirmed by a human.
2. Download Kyokushin MoCap (CC0), project to 2D with realistic noise, fine-tune, leave-one-athlete-out.
3. Label some real swimming video to measure the true gap of the synthetic-trained model.
4. Evaluate on held-out performers before claiming the system recognises karate.
