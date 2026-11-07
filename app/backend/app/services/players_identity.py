"""Who are the two fighters, and which one is Player 1 - for the whole video.

Lessons from real club footage that shaped this:
  * picking players by whole-body motion labelled the COACH (he walks the mat);
  * tagging every frame by appearance put "Player 2" on a TEAMMATE ON THE
    BENCH - club members wear the same team shirt, so "looks most like the
    player" is not "is the player";
  * labels swapped when fighters clinched (the tracker exchanges ids).

So identity comes from TRACKING, and appearance is only evidence:
  1. players = the two tracks that move their HANDS AND FEET most relative to
     their own body (x time on screen) AND are on screen at the same time -
     two people fighting each other (a coach walking moves the body, not the
     limbs);
  2. each player's track is extended over tracker breaks by joining pieces
     that follow it in time and space - and, when signatures exist, only if
     the piece looks like the same person (OSNet re-id / colour histogram);
  3. 1<->2 swaps are checked ONLY between the two players: a two-state Viterbi
     over the whole clip ("labels as tracked" vs "labels swapped", with a cost
     per switch), so a label flips only on strong, sustained evidence, and a
     bystander can never receive a label however similar they look.
All thresholds are in seconds, shares of the frame, or signature units - no
pixel or frame-count constants - so the same rules apply to any video.
"""
from __future__ import annotations

import math

import numpy as np

from ..utils.logging_utils import JobLogger
from .skeleton_service import _body_center_and_scale, limb_step

PLAYER_TRACK_BASE = 100000   # synthetic track ids for the players (never clash)
# Two kinds of signature, each with its own scale:
#   "reid"       OSNet-AIN 512-d embedding, cosine distance (0 same .. 2 opposite).
#                Preferred: trained to recognise the same PERSON, not colours.
#   "appearance" HSV clothing histogram, intersection distance (0..1). Fallback
#                when the re-id model is not installed.
MERGE_DIST = 0.30            # appearance: pieces closer than this may be the same person
REID_MERGE_DIST = 0.30       # re-id cosine distance: same person
MIN_TRACKLET_S = 0.4         # ignore tracker pieces shorter than this
MAX_OVERLAP_S = 0.3          # a tracker handing one person to a new id can overlap this
                             # long (seen: 4 frames); longer together = two people
LONG_GAP_S = 20.0            # second pass may bridge breaks this long (person off-screen /
                             # hidden), but only on strict, unambiguous visual evidence


def app_dist(a, b) -> float:
    """0 = identical clothing colours, 1 = nothing in common (per-half
    histogram intersection, averaged over shirt and trousers)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    h = len(a) // 2
    return float((2.0 - np.minimum(a[:h], b[:h]).sum() - np.minimum(a[h:], b[h:]).sum()) / 2.0)


def reid_dist(a, b) -> float:
    """Cosine distance between L2-normalised re-id embeddings."""
    return max(0.0, 1.0 - float(np.dot(a, b)))


METRICS = {
    "reid": (reid_dist, REID_MERGE_DIST),
    "appearance": (app_dist, MERGE_DIST),
}


def _proto(vectors: list, key: str) -> np.ndarray:
    m = np.mean(np.asarray(vectors, np.float32), axis=0)
    return m / (np.linalg.norm(m) + 1e-8) if key == "reid" else m


def _update_proto(proto: np.ndarray, v, key: str, rate: float) -> np.ndarray:
    m = (1.0 - rate) * proto + rate * np.asarray(v, np.float32)
    return m / (np.linalg.norm(m) + 1e-8) if key == "reid" else m


def sig_key(dets: list[dict]) -> str | None:
    """Which signature is available for most detections (re-id preferred)."""
    for k in ("reid", "appearance"):
        if dets and sum(p.get(k) is not None for p in dets) >= 0.5 * len(dets):
            return k
    return None


def swapped_is_better(sig0, sig1, protos, key: str) -> float:
    """Evidence that two player detections carry each other's labels:
    > 0 means "swapped" fits the reference looks better than "as labelled"."""
    dist, _ = METRICS[key]
    keep = dist(sig0, protos[0]) + dist(sig1, protos[1])
    swap = dist(sig0, protos[1]) + dist(sig1, protos[0])
    return keep - swap


def _is_standing(kp) -> bool | None:
    """True when the ankles are a leg's length below the hips and the knees
    clearly below them (standing or in a fighting stance), False when sitting
    (on the floor or on a chair) or kneeling. None when hips, neck or ankles
    are not visible."""
    bs = _body_center_and_scale(kp)
    if bs is None:
        return None
    (hx, hy), torso = bs
    ankles = [kp[j][1] for j in (11, 14) if kp[j][2] > 0.1]
    if not ankles:
        return None
    if (max(ankles) - hy) < 0.7 * torso:
        return False
    # Sitting on a chair or bench keeps the feet down but brings the thighs
    # level with the hips: a standing / fighting stance has the knees well
    # below the hips.
    knees = [kp[j][1] for j in (10, 13) if kp[j][2] > 0.1]
    return not knees or (max(knees) - hy) >= 0.3 * torso


def _behaviour(dets: list, fps: float) -> tuple[float, float | None]:
    """(limb activity in torso lengths / s, share of frames standing) of one
    tracker piece. Used to refuse joining pieces that behave like different
    people - a team-mate sitting on the bench can wear the SAME shirt as a
    fighter, so appearance alone cannot keep them apart, but sitting still vs
    fighting can."""
    dets = sorted(dets, key=lambda d: d[0])
    steps = [limb_step(a[1]["keypoints"], b[1]["keypoints"])
             for a, b in zip(dets, dets[1:]) if b[0] - a[0] == 1]
    steps = [x for x in steps if x is not None]
    flags = [x for x in (_is_standing(d[1]["keypoints"]) for d in dets) if x is not None]
    limb = (sum(steps) / len(steps) * fps) if steps else 0.0
    return limb, (sum(flags) / len(flags) if flags else None)


def _same_behaviour(limb_a: float, st_a: float | None, limb_b: float, st_b: float | None) -> bool:
    """Could two pieces be the same person, judging by posture and activity?
    Standing share must not jump by more than half, and limb activity must be
    within 3x (floored at 0.3 torso/s so two calm pieces always agree)."""
    if st_a is not None and st_b is not None and abs(st_a - st_b) > 0.5:
        return False
    hi, lo = max(limb_a, limb_b), max(min(limb_a, limb_b), 0.3)
    return hi <= 3.0 * lo


def _calibrate_merge(span: dict, dist, default_thr: float, min_frames: int,
                     log: JobLogger) -> float:
    """Signature distance below which two pieces may be the same person,
    measured on THIS video instead of a fixed number.

    Absolute thresholds do not transfer: on a real club video OSNet put the
    same fighter's pieces at 0.01-0.06 and DIFFERENT people (fighter vs coach)
    at only 0.10-0.20 - a fixed 0.30 let the coach join a fighter. Pieces seen
    on screen at the same time are certainly different people, so their
    distances describe "different" for this model and this footage; the
    threshold sits safely below the closest of them.
    """
    import numpy as np

    pcs = [sp for sp in span.values() if sp["proto"] is not None and len(sp["frames"]) >= min_frames]
    diff = []
    for i, a in enumerate(pcs):
        for b in pcs[i + 1:]:
            if len(a["frames"] & b["frames"]) >= min_frames:
                diff.append(dist(a["proto"], b["proto"]))
    if len(diff) < 5:
        log.info(f"players: too few co-visible pairs to calibrate; join threshold {default_thr:.3f}",
                 stage="skeleton")
        return default_thr
    thr = float(np.clip(0.8 * np.percentile(diff, 5), 0.01, default_thr))
    log.info(f"players: join threshold {thr:.3f} (from {len(diff)} pairs of people seen together; "
             f"closest different people at {np.percentile(diff, 5):.3f})", stage="skeleton")
    return thr


def _merge_chains(chains: list, dist, thr, max_gap: int, max_overlap: int, diag: float,
                  key, log: JobLogger, long_gap: int | None = None) -> list:
    """Second pass: join whole people that the first, time-ordered pass kept
    apart (a fighter's first minute vs the rest; a fighter hidden behind the
    coach for 16 s). Never on screen together, same look, same behaviour.

    Short breaks (<= max_gap) use the normal threshold. Longer breaks (up to
    long_gap) need a stricter look match AND must be unambiguous: the best
    candidate has to be clearly closer than the next one - with team-mates in
    the same shirt, two similar candidates means "don't guess", and an
    unlabelled stretch is better than a label on the wrong person."""
    long_gap = long_gap or max_gap
    merged = 0

    def sig(a, b):
        if not (key and a["proto"] is not None and b["proto"] is not None
                and a.get("last_proto") is not None and b.get("first_proto") is not None):
            return None
        return max(dist(a["proto"], b["proto"]), dist(a["last_proto"], b["first_proto"]))

    while True:
        best = None
        for i, a in enumerate(chains):
            cands = []
            for j, b in enumerate(chains):
                if i == j or a["last"] > b["first"] + max_overlap:
                    continue  # b must follow a
                if len(a["frames"] & b["frames"]) > max_overlap:
                    continue
                gap = max(b["first"] - a["last"], 0)
                if gap > long_gap:
                    continue
                sd = sig(a, b)
                if sd is None:
                    continue  # whole people are only merged on positive visual evidence
                if not _same_behaviour(a["b_limb"], a["b_standing"], b["b_limb"], b["b_standing"]):
                    continue
                cands.append((sd, gap, j, b))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0])
            for rank, (sd, gap, j, b) in enumerate(cands):
                d = math.dist(a["c1"], b["c0"])
                if gap <= max_gap:
                    d_lim = diag * (0.25 + 0.25 * gap / max_gap)
                    if d > d_lim or sd > thr:
                        continue
                    cost = gap / max_gap + d / d_lim + sd / thr
                else:
                    # long break: anywhere on screen, but strict and unambiguous
                    nxt = cands[rank + 1][0] if rank + 1 < len(cands) else None
                    if sd > 0.8 * thr or (nxt is not None and sd > 0.8 * nxt):
                        continue
                    cost = 2.0 + gap / long_gap + sd / thr
                if best is None or cost < best[0]:
                    best = (cost, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = chains[i], chains[j]
        wa, wb = len(a["frames"]), len(b["frames"])
        a["tids"] += b["tids"]
        a["frames"] |= b["frames"]
        a["last"], a["c1"] = b["last"], b["c1"]
        a["last_proto"] = b["last_proto"]
        a["b_limb"] = (a["b_limb"] * wa + b["b_limb"] * wb) / (wa + wb)
        if b["b_standing"] is not None:
            a["b_standing"] = (b["b_standing"] if a["b_standing"] is None
                               else (a["b_standing"] * wa + b["b_standing"] * wb) / (wa + wb))
        a["proto"] = _update_proto(a["proto"], b["proto"], key, wb / (wa + wb))
        chains.pop(j)
        merged += 1
    if merged:
        log.info(f"players: merged {merged} split people (second pass)", stage="skeleton")
    return chains


def identify_players(parsed: dict, n_players: int, log: JobLogger, fps: float = 30.0,
                     max_gap_s: float = 3.0, switch_cost_s: float = 1.0,
                     choose_only: bool = False) -> dict | None:
    """Label Player 1..n on every frame (see module doc). Expects parsed tracks
    ranked by limb activity (track_people(rank_by="limbs")). Returns None when
    there is nothing to label."""
    frames = parsed["frames"]
    tracks = parsed.get("tracks") or []
    if not frames or not tracks:
        return None
    fps = fps or 30.0
    min_frames = max(3, round(MIN_TRACKLET_S * fps))
    max_overlap = max(1, round(MAX_OVERLAP_S * fps))
    max_gap = max(1, round(max_gap_s * fps))
    dets = [p for f in frames for p in f["people"]]
    key = sig_key(dets)
    dist, merge_thr = METRICS[key] if key else (None, None)
    cents = [p["centroid"] for p in dets if p.get("centroid")]
    if not cents:
        return None
    diag = math.hypot(max(c[0] for c in cents) - min(c[0] for c in cents),
                      max(c[1] for c in cents) - min(c[1] for c in cents)) or 1.0

    # --- per-track spans and signatures -------------------------------------
    span: dict = {}
    for f in frames:
        for p in f["people"]:
            tid = p.get("track_id")
            if tid is None or not p.get("centroid"):
                continue
            sp = span.setdefault(tid, {"frames": set(), "first": f["frame"], "last": f["frame"],
                                       "c0": p["centroid"], "c1": p["centroid"], "sig": [], "dets": []})
            sp["frames"].add(f["frame"])
            sp["dets"].append((f["frame"], p))
            sp["last"], sp["c1"] = f["frame"], p["centroid"]
            if key and p.get(key) is not None:
                sp["sig"].append(p[key])
    for sp in span.values():
        sp["proto"] = _proto(sp["sig"], key) if key and sp["sig"] else None
        sp["limb"], sp["standing"] = _behaviour(sp["dets"], fps)
    if key:
        merge_thr = _calibrate_merge(span, dist, merge_thr, min_frames, log)

    # --- 1. join tracker pieces into PEOPLE (everyone, not only players) -----
    # A still bystander stays one long piece while a fighter, who moves and is
    # occluded, breaks into many short ones - so ranking PIECES favoured people
    # sitting on the bench. Pieces are chained first: a piece continues a
    # person if it starts after that person's last frame (within max_gap),
    # nearby, and - when signatures exist - looks like the same person.
    chains: list[dict] = []
    for tid in sorted((t for t in span if len(span[t]["frames"]) >= min_frames),
                      key=lambda t: span[t]["first"]):
        sp = span[tid]
        best = None
        for ch in chains:
            gap = sp["first"] - ch["last"]
            if gap < 1 - max_overlap or gap > max_gap:
                continue
            d = math.dist(ch["c1"], sp["c0"])
            d_lim = diag * (0.25 + 0.25 * max(gap, 0) / max_gap)  # people move during a break
            if d > d_lim:
                continue
            # Compare with the person's average AND their most recent piece:
            # at a crossing the tracker can hand a fighter's position to the
            # coach, who may resemble the fighter's whole-match average more
            # than the fighter a moment ago (real case: 0.143 vs the last piece).
            sd = (max(dist(sp["proto"], ch["proto"]), dist(sp["proto"], ch["last_proto"]))
                  if key and sp["proto"] is not None and ch["proto"] is not None
                  and ch.get("last_proto") is not None else None)
            if sd is not None and sd > merge_thr:
                continue  # does not look like the same person
            if not _same_behaviour(ch["b_limb"], ch["b_standing"], sp["limb"], sp["standing"]):
                continue  # e.g. a fighter's piece followed by someone SITTING in the same shirt
            cost = max(gap, 0) / max_gap + d / d_lim + (sd / merge_thr if sd is not None else 0.5)
            if best is None or cost < best[0]:
                best = (cost, ch)
        if best is None:
            chains.append({"tids": [tid], "frames": set(sp["frames"]), "first": sp["first"],
                           "last": sp["last"], "c0": sp["c0"], "c1": sp["c1"], "proto": sp["proto"],
                           "first_proto": sp["proto"], "last_proto": sp["proto"],
                           "b_limb": sp["limb"], "b_standing": sp["standing"]})
        else:
            ch = best[1]
            w = len(sp["frames"]) / (len(ch["frames"]) + len(sp["frames"]))
            ch["b_limb"] = (1 - w) * ch["b_limb"] + w * sp["limb"]
            if sp["standing"] is not None:
                ch["b_standing"] = (sp["standing"] if ch["b_standing"] is None
                                    else (1 - w) * ch["b_standing"] + w * sp["standing"])
            ch["tids"].append(tid)
            ch["frames"] |= sp["frames"]
            ch["last"], ch["c1"] = sp["last"], sp["c1"]
            if sp["proto"] is not None:
                ch["last_proto"] = sp["proto"]
            if ch["proto"] is not None and sp["proto"] is not None:
                ch["proto"] = _update_proto(ch["proto"], sp["proto"], key,
                                            len(sp["frames"]) / len(ch["frames"]))
    if not chains:
        return None
    chains = _merge_chains(chains, dist, merge_thr, max_gap, max_overlap, diag, key, log,
                           long_gap=max(max_gap, round(LONG_GAP_S * fps)))

    # --- 2. score each PERSON: is this someone fighting? -----------------------
    # All measures are in torso lengths and seconds, so they mean the same at
    # any resolution, zoom or frame rate.
    by_tid = {}
    for f in frames:
        for p in f["people"]:
            if p.get("track_id") is not None:
                by_tid.setdefault(p["track_id"], {})[f["frame"]] = p
    for ch in chains:
        # One detection per frame (two pieces may overlap by a frame or two);
        # sort by frame number only - the detections themselves are dicts.
        per_frame: dict = {}
        for tid in ch["tids"]:
            for fr, p in by_tid.get(tid, {}).items():
                per_frame.setdefault(fr, p)
        seq = sorted(per_frame.items(), key=lambda kv: kv[0])
        limb, foot, standing, n_pose = [], [], 0, 0
        for (f0, p0), (f1, p1) in zip(seq, seq[1:]):
            if f1 - f0 != 1:
                continue
            ls = limb_step(p0["keypoints"], p1["keypoints"])
            if ls is not None:
                limb.append(ls)
            bs = _body_center_and_scale(p1["keypoints"])
            if bs and p0.get("centroid") and p1.get("centroid"):
                foot.append(math.dist(p0["centroid"], p1["centroid"]) / bs[1])
        for _, p in seq:
            st = _is_standing(p["keypoints"])
            if st is not None:
                n_pose += 1
                standing += st
        ch["limb"] = (sum(limb) / len(limb) * fps) if limb else 0.0       # torso lengths / s
        ch["footwork"] = (sum(foot) / len(foot) * fps) if foot else 0.0   # torso lengths / s
        ch["standing"] = standing / n_pose if n_pose else 0.5
        ch["seconds"] = len(ch["frames"]) / fps
        # Activity decides; time on screen only helps (square root) so someone
        # sitting there the whole clip cannot outweigh the people fighting.
        ch["score"] = (math.sqrt(ch["seconds"]) * (ch["limb"] + 0.5 * ch["footwork"])
                       * (0.2 + 0.8 * ch["standing"]))
    chains.sort(key=lambda c: c["score"], reverse=True)
    # Kept on the result for inspection (which tracker pieces became which person).
    parsed["people_found"] = [
        {"tids": sorted(c["tids"]), "first_s": round(c["first"] / fps, 1), "last_s": round(c["last"] / fps, 1),
         "seconds": round(c["seconds"], 1), "limb": round(c["limb"], 2), "standing": c["standing"],
         "score": round(c["score"], 2)}
        for c in chains[:10]
    ]

    # --- 3. players: the most active person, then the most active person who
    #        is on screen at the SAME TIME (two people fighting each other)
    players = [chains[0]]
    if n_players >= 2:
        for ch in chains[1:]:
            if len(ch["frames"] & chains[0]["frames"]) >= max(min_frames, max_overlap + 1):
                players.append(ch)
                break
    for k, pl in enumerate(players):
        log.info(f"players: Player {k + 1} candidate: {pl['seconds']:.1f}s on screen, limbs "
                 f"{pl['limb']:.2f}, footwork {pl['footwork']:.2f} torso/s, standing "
                 f"{pl['standing']:.0%}", stage="skeleton")
    for other in chains[len(players):len(players) + 3]:
        log.info(f"  not a player: {other['seconds']:.1f}s on screen, limbs {other['limb']:.2f}, "
                 f"footwork {other['footwork']:.2f} torso/s, standing {other['standing']:.0%}",
                 stage="skeleton")
    owned = {tid for pl in players for tid in pl["tids"]}
    if choose_only:  # caller only wants WHO the players are (player_follow)
        return {"players": players, "key": key, "dist": dist, "thr": merge_thr, "span": span}

    slot_of = {tid: k for k, pl in enumerate(players) for tid in pl["tids"]}
    for f in frames:
        for p in f["people"]:
            k = slot_of.get(p.get("track_id"))
            if k is not None:
                p["player_slot"] = k

    # --- 3. 1<->2 swap check between the two players only (Viterbi) -----------
    swaps = 0
    if key and len(players) == 2:
        both = []
        for f in frames:
            d = {p["player_slot"]: p for p in f["people"] if p.get("player_slot") is not None}
            if (0 in d and 1 in d and d[0].get(key) is not None and d[1].get(key) is not None
                    and d[0].get("centroid") and d[1].get("centroid")):
                both.append((f, d))

        def apart(d):  # fighters clearly separated -> their looks are not mixed up
            b = d[0].get("bbox")
            h = (b[3] - b[1]) if b else diag * 0.2
            return math.dist(d[0]["centroid"], d[1]["centroid"]) > h

        ref = [d for _, d in both if apart(d)] or [d for _, d in both]
        if ref:
            protos = [_proto([d[k][key] for d in ref], key) for k in (0, 1)]
            evidence = {f["frame"]: swapped_is_better(d[0][key], d[1][key], protos, key)
                        for f, d in both}
            # cost per state per frame: state 0 = as tracked, 1 = swapped.
            switch = switch_cost_s * fps * 0.1  # ~1 s of 0.1-unit evidence per switch
            cost = [0.0, switch]
            back = []
            for f in frames:
                e = evidence.get(f["frame"], 0.0)
                emit = (max(e, 0.0), max(-e, 0.0))  # pay for evidence against the state
                new, ptr = [0.0, 0.0], [0, 0]
                for st in (0, 1):
                    stay, flip = cost[st], cost[1 - st] + switch
                    new[st], ptr[st] = (stay, st) if stay <= flip else (flip, 1 - st)
                    new[st] += emit[st]
                back.append(ptr)
                cost = new
            st = 0 if cost[0] <= cost[1] else 1
            states = [0] * len(frames)
            for i in range(len(frames) - 1, -1, -1):
                states[i] = st
                st = back[i][st]
            prev = 0
            for f, st in zip(frames, states):
                if st != prev:
                    swaps += 1
                    prev = st
                if st == 1:
                    for p in f["people"]:
                        if p.get("player_slot") is not None:
                            p["player_slot"] = 1 - p["player_slot"]

    # --- 4. players become the first tracks (sent to the action model) --------
    tagged = [0] * len(players)
    for f in frames:
        for p in f["people"]:
            k = p.get("player_slot")
            if k is not None:
                p["track_id"] = PLAYER_TRACK_BASE + k
                tagged[k] += 1
    head = [{
        "track_id": PLAYER_TRACK_BASE + k,
        "frames_present": tagged[k],
        "mean_confidence": 1.0,
        "score": 1e9 - k,
        "player": f"Player {k + 1}",
        "from_tracker_pieces": sorted(pl["tids"]),
    } for k, pl in enumerate(players)]
    parsed["tracks"] = head + [t for t in tracks if t["track_id"] not in owned]
    parsed["track_count"] = len(parsed["tracks"])
    parsed["tracker"] = (parsed.get("tracker") or "tracker") + (
        f" + {'OSNet re-id' if key == 'reid' else 'colour'}-checked linking" if key else " + linking")
    for k, pl in enumerate(players):
        log.info(f"players: Player {k + 1} = tracker pieces {sorted(pl['tids'])[:15]}"
                 f"{' ...' if len(pl['tids']) > 15 else ''} -> {tagged[k]}/{len(frames)} frames",
                 stage="skeleton")
    log.info(f"players: signature = {key or 'none'}; 1<->2 swap corrections: {swaps}",
             stage="skeleton")
    return {"players": len(players), "frames_tagged": tagged, "swap_corrections": swaps,
            "signature": key}
