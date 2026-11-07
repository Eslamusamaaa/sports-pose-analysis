"""Pick two players at the start, then FOLLOW THOSE SAME TWO PEOPLE to the end.

The requirement is not "decide who the players are at every moment" but:
choose Player 1 and Player 2 once, early in the clip, and keep each label on
that same person for the whole video - never swap them, never move a label to
a third person. When unsure, show no label rather than a wrong one.

That is persistent two-target tracking, which needs far fewer rules than
re-deciding identity from scratch, and so generalises better:

  1. WHO: the most hand/foot-active standing person and the most active one
     on screen with them, judged on the whole clip (a quiet start - bowing,
     waiting at the lines - would otherwise pick people on the bench). The
     labels are fixed at the first moment both are seen together, and each
     player's REFERENCE LOOK (re-id signature) is taken there and never
     changed, so it cannot drift onto someone else.
  2. EVERY FRAME, forward and backward from that moment, both labels are
     assigned jointly to the detections:
       - where the player should be (last position + velocity), with a search
         radius in units of the player's own height that grows while missing;
       - how much the detection looks like THIS player's reference, against a
         per-player threshold measured on this video (just below how close
         anybody else ever comes to that player's look);
       - a bonus for keeping the same tracker id;
       - one detection can never carry both labels.
  3. RE-ACQUIRING after the player was lost needs a clearly matching look
     that is closer to this player than to the OTHER player - so a coach or
     a team-mate in the same shirt does not inherit the label.
  4. A whole-clip check (two-state Viterbi) undoes any 1<->2 exchange that
     the frame-by-frame step made during a clinch.
All units are relative (person heights, seconds, a look-threshold measured on
the same video), so nothing is tuned to a particular clip.
"""
from __future__ import annotations

import math

import numpy as np

from ..utils.logging_utils import JobLogger
from . import players_identity
from .players_identity import PLAYER_TRACK_BASE
from .skeleton_service import _body_center_and_scale, limb_step

GAP_FILL_S = 1.0      # gaps up to this long, with the same player on both sides, are filled
ACTIVITY_LAG_S = 0.2  # hand/foot movement measured over this interval (jitter averages out)
SPLIT_SLACK = 1.5     # a group this close to a player (and never with them) is that player
GROUP_BONUS = 0.3
TOGETHER_MIN = 0.2    # the two players are seen apart, together, for >= this share of the shorter one
TOGETHER_SPLIT = 0.1  # a group seen with the player at most this share of its time may be them ...
TOGETHER_OTHER = 0.3  # ... seen with them at least this share: certainly somebody else
MIN_PIECE_S = 0.4     # tracker pieces shorter than this are too noisy to group by look
GATE_H = 0.8          # search radius (in player heights) right after being seen ...
GATE_GROW_H = 1.0     # ... growing by this many heights per second while missing
GATE_MAX_FRAC = 0.6   # ... capped at this share of the frame diagonal
REACQUIRE_S = 0.5     # missing longer than this -> strict re-acquisition rules
REACQUIRE_LOOK = 0.75  # ... and a look this much closer than the threshold
LOST_S = 3.0          # missing this long -> search the whole frame (by look)
SAME_ID_BONUS = 0.3
THIRD_S = 0.5         # seen this long beside both players -> a third person for good
OTHERS_MARGIN = 0.9   # per-player look threshold = 0.9 x closest other people (2nd percentile)
SWITCH_COST_S = 1.0   # Viterbi: evidence needed (seconds-worth) to swap the two labels


def _box(p):
    b = p.get("bbox")
    if b:
        return b
    xs = [x for x, y, c in p["keypoints"] if c > 0.1]
    ys = [y for x, y, c in p["keypoints"] if c > 0.1]
    if len(xs) < 2:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _height(p) -> float:
    b = _box(p)
    return max((b[3] - b[1]) if b else 0.0, 1.0)


def _overlap(a, b) -> float:
    """Share of a's box covered by b's box (crop contamination)."""
    ba, bb = _box(a), _box(b)
    if not ba or not bb:
        return 0.0
    w = min(ba[2], bb[2]) - max(ba[0], bb[0])
    h = min(ba[3], bb[3]) - max(ba[1], bb[1])
    area = max((ba[2] - ba[0]) * (ba[3] - ba[1]), 1.0)
    return max(w, 0.0) * max(h, 0.0) / area


def _calibrate(frames, key, dist, log) -> float | None:
    """Look-distance below which two detections can be the same person,
    measured on THIS video: people visible together are certainly different,
    so the threshold sits below the closest such pairs (per frame, sampled)."""
    if key is None:
        return None
    diff = []
    step = max(1, len(frames) // 400)  # sample ~400 frames
    for f in frames[::step]:
        sigs = [p[key] for p in f["people"] if p.get(key) is not None]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                diff.append(dist(sigs[i], sigs[j]))
    if len(diff) < 20:
        return None
    thr = float(0.8 * np.percentile(diff, 5))
    log.info(f"players: look threshold {thr:.3f} (from {len(diff)} pairs of people seen together)",
             stage="skeleton")
    return thr


def _split_by_look(frames, fps, log) -> int:
    """Cut tracker pieces where the person inside changes.

    In a clinch the tracker can carry one id from one fighter onto the other;
    such a piece is half one person, half the other, and its average look
    belongs to nobody. Compare the mean look of the half second before and
    after each moment; where they differ by more than the distance between
    pieces of DIFFERENT people seen together in this video, start a new id.
    Rewrites track_id in place; returns the number of cuts."""
    seqs: dict = {}
    for f in frames:
        for p in f["people"]:
            if p.get("track_id") is not None and p.get("reid") is not None:
                seqs.setdefault(p["track_id"], []).append((f["frame"], p))
    w = max(3, round(0.5 * fps))
    long = {t: d for t, d in seqs.items() if len(d) >= 2 * w}
    if len(long) < 2:
        return 0
    protos = {t: np.mean([p["reid"] for _, p in d], axis=0) for t, d in long.items()}
    protos = {t: m / (np.linalg.norm(m) + 1e-12) for t, m in protos.items()}
    spans = {t: (d[0][0], d[-1][0]) for t, d in long.items()}
    ids = list(long)
    co = [1.0 - float(protos[a] @ protos[b]) for i, a in enumerate(ids) for b in ids[i + 1:]
          if min(spans[a][1], spans[b][1]) - max(spans[a][0], spans[b][0]) > fps]
    if len(co) < 10:
        return 0
    cut_thr = float(np.percentile(co, 5))
    next_id = max(seqs) + 1
    cuts = 0
    for t, d in long.items():
        v = np.stack([p["reid"] for _, p in d]).astype(np.float64)
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        score = np.zeros(len(d))
        for i in range(w, len(d) - w + 1):
            a, b = cs[i] - cs[i - w], cs[i + w] - cs[i]
            score[i] = 1.0 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        points, i = [], w
        while i <= len(d) - w:
            if score[i] > cut_thr:
                j = i + int(np.argmax(score[i:i + w]))  # the sharpest change nearby
                points.append(j)
                i = j + w
            else:
                i += 1
        for j in points:
            for _, p in d[j:]:
                p["track_id"] = next_id
            next_id += 1
            cuts += 1
    if cuts:
        log.info(f"players: cut {cuts} tracker pieces where the person inside changed "
                 f"(look change > {cut_thr:.3f}, measured on this video)", stage="skeleton")
    return cuts


def _people_by_look(frames, fps, log):
    """Group tracker pieces into PEOPLE by their re-id look, for the whole clip.

    Average-linkage merging of piece signatures, with one hard rule: two
    pieces seen at the same time are two different people. The merge limit
    comes from this video (just below the closest pair of people ever seen
    together), so it adapts to the camera, lighting and how alike the clothes
    are. Returns [{"tids", "frames", "proto", "pieces"}] or None without looks.
    """
    pieces: dict = {}
    for f in frames:
        for p in f["people"]:
            t = p.get("track_id")
            if t is not None and p.get("reid") is not None:
                pieces.setdefault(t, []).append((f["frame"], p))
    min_len = max(3, round(MIN_PIECE_S * fps))
    tids = [t for t, d in pieces.items() if len(d) >= min_len]
    if len(tids) < 2:
        return None
    sums = np.stack([np.sum([p["reid"] for _, p in pieces[t]], axis=0) for t in tids]).astype(np.float64)
    span = np.array([[pieces[t][0][0], pieces[t][-1][0]] for t in tids])
    n = len(tids)
    ov = np.maximum(0, np.minimum(span[:, None, 1], span[None, :, 1])
                    - np.maximum(span[:, None, 0], span[None, :, 0]) + 1).astype(np.float64)
    np.fill_diagonal(ov, 0)

    def unit(m):
        return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)

    u = unit(sums)
    dist = 1.0 - u @ u.T
    co = dist[ov > fps]  # pieces on screen together for > 1 s: certainly different people
    if co.size < 10:
        return None
    thr = float(0.8 * np.percentile(co, 5))
    max_ov = round(0.3 * fps)  # tracker hand-overs overlap by a few frames
    members = {i: [i] for i in range(n)}
    active = np.ones(n, bool)
    big = 9.0
    d = dist.copy()
    d[ov > max_ov] = big
    np.fill_diagonal(d, big)
    while True:
        k = int(np.argmin(d))
        i, j = divmod(k, n)
        if d[i, j] >= thr:
            break
        members[i] += members.pop(j)
        sums[i] += sums[j]
        ov[i] += ov[j]
        ov[:, i] = ov[i]
        active[j] = False
        d[j, :] = big
        d[:, j] = big
        row = 1.0 - unit(sums) @ unit(sums[i])
        row[(ov[i] > max_ov) | ~active] = big
        row[i] = big
        d[i, :] = row
        d[:, i] = row
    people = []
    for i, m in members.items():
        ts = [tids[x] for x in m]
        people.append({"tids": ts, "proto": unit(sums[i]).astype(np.float32), "merge_thr": thr,
                       "frames": {fr for t in ts for fr, _ in pieces[t]},
                       "pieces": {t: pieces[t] for t in ts}})
    log.info(f"players: {len(tids)} tracker pieces grouped into {len(people)} people by look "
             f"(merge limit {thr:.3f}, from people seen together in this video)", stage="skeleton")
    return people


def _pick_by_look(frames, n, fps, log):
    """WHO the players are, from the people found by look (whole clip).

    A filmed match keeps the fighters in the picture, so they are the people
    on screen the longest; they also move hands and feet the most and stand.
    Score = share of the clip on screen x limb activity (torso lengths/s) x
    standing. The second player is the best-scoring person seen together with
    the first. Everything is relative (shares, torso lengths, seconds).
    """
    people = _people_by_look(frames, fps, log)
    if not people or len(people) < n:
        return None
    n_frames = max(len(frames), 1)
    lag = max(1, round(ACTIVITY_LAG_S * fps))
    for pe in people:
        limb, stand = [], []
        for d in pe["pieces"].values():
            # Over a fifth of a second, not frame to frame: joint jitter (strong
            # with bottom-up engines on people near the camera) averages out,
            # real punches and kicks do not.
            by_fr = dict(d)
            for fr, p0 in d:
                p1 = by_fr.get(fr + lag)
                if p1 is not None:
                    s = limb_step(p0["keypoints"], p1["keypoints"])
                    if s is not None:
                        limb.append(s / lag)
            stand += [s for s in (players_identity._is_standing(p["keypoints"]) for _, p in d) if s is not None]
        pe["share"] = len(pe["frames"]) / n_frames
        pe["limb"] = float(np.mean(limb) * fps) if limb else 0.0
        pe["standing"] = float(np.mean(stand)) if stand else 0.5
        pe["score"] = pe["share"] * pe["limb"] * (0.2 + 0.8 * pe["standing"])
    people.sort(key=lambda pe: pe["score"], reverse=True)
    def apart_frames(a, b):
        """Frames where a and b are two SEPARATE people (boxes not on top of
        each other) - one person split into two groups, or given two boxes in
        a close-up, never is."""
        da = {fr: p for d in a["pieces"].values() for fr, p in d}
        db = {fr: p for d in b["pieces"].values() for fr, p in d}
        return sum(1 for fr in da.keys() & db.keys()
                   if _overlap(da[fr], db[fr]) < 0.3 and _overlap(db[fr], da[fr]) < 0.3)

    chosen = [people[0]]
    for pe in people[1:]:
        if len(chosen) == n:
            break
        together = apart_frames(pe, chosen[0])
        if together >= max(fps, TOGETHER_MIN * min(len(pe["frames"]), len(chosen[0]["frames"]))):
            chosen.append(pe)
    if len(chosen) < n:
        return None
    for k, pe in enumerate(chosen):
        log.info(f"players: Player {k + 1} = on screen {pe['share']:.0%} of the clip, limbs "
                 f"{pe['limb']:.2f} torso/s, standing {pe['standing']:.0%}", stage="skeleton")
    for pe in people[len(chosen):len(chosen) + 3]:
        log.info(f"  not a player: on screen {pe['share']:.0%}, limbs {pe['limb']:.2f} torso/s, "
                 f"standing {pe['standing']:.0%}", stage="skeleton")
    # Start point: a frame showing both, where each one's tracker piece looks
    # most like that person overall (the cleanest moment to fix the labels).
    best = None
    for k_fr in sorted(set.intersection(*(pe["frames"] for pe in chosen))):
        picks = []
        for pe in chosen:
            hit = None
            for t, d in pe["pieces"].items():
                if d[0][0] <= k_fr <= d[-1][0]:
                    det = next((p for fr, p in d if fr == k_fr), None)
                    if det is not None:
                        m = np.mean([p["reid"] for _, p in d], axis=0)
                        m = m / (np.linalg.norm(m) + 1e-12)
                        hit = (1.0 - float(m @ pe["proto"]), t, det, len(d))
                        break
            if hit is None:
                break
            picks.append(hit)
        if len(picks) < n:
            continue
        cost = max(h[0] for h in picks) - 1e-6 * min(h[3] for h in picks)
        if best is None or cost < best[0]:
            best = (cost, k_fr, picks)
    if best is None:
        return None
    _, anchor, picks = best
    out = [{"ref": pe["proto"], "last": h[2], "tid": h[1]} for pe, h in zip(chosen, picks)]
    # Which tracker pieces may carry each label.
    #  - ALLOWED: the player's own group, plus a group that is the same player
    #    split in two: looks within the grouping limit (with slack), closer to
    #    this player than to the other one, and (almost) never on screen
    #    together with them (a close-up can give one person two boxes).
    #  - FORBIDDEN: a group on screen together with the player for a good
    #    part of its time is certainly somebody else (team-mate in the same
    #    shirt, coach...), and so is a group that looks more like the other
    #    player. Everything else is left to the frame-by-frame look check.
    thr = chosen[0]["merge_thr"]
    for k, (pe, o) in enumerate(zip(chosen, out)):
        mine = set(pe["tids"])
        for other in people:
            if other is pe or other in chosen:
                continue
            d_me = 1.0 - float(other["proto"] @ pe["proto"])
            d_rest = [1.0 - float(other["proto"] @ q["proto"]) for q in chosen if q is not pe]
            together = len(other["frames"] & pe["frames"]) / max(len(other["frames"]), 1)
            closer_to_me = all(d_me < x for x in d_rest)
            if together <= TOGETHER_SPLIT and d_me < SPLIT_SLACK * thr and closer_to_me:
                mine |= set(other["tids"])
            elif together >= TOGETHER_OTHER or not closer_to_me:
                o.setdefault("forbidden", set()).update(other["tids"])
        for q in chosen:
            if q is not pe:
                o.setdefault("forbidden", set()).update(q["tids"])
        o["allowed"] = mine
    log.info("players: tracker pieces allowed per player " + ", ".join(
        f"{len(o['allowed'])}" for o in out) + " (pieces of other people are never labelled)",
        stage="skeleton")
    log.info(f"players: Player 1/2 fixed at {anchor / fps:.1f}s, then followed to both ends of the clip",
             stage="skeleton")
    return anchor, out, "reid", players_identity.reid_dist, None


def _pick_players(parsed, n, fps, log):
    """WHO the players are. With re-id looks: people grouped by look over the
    whole clip (_pick_by_look). Without: the most hand/foot-active person and
    the most active person on screen with them (players_identity).
    Following then starts from the chosen moment and runs to both ends."""
    if any(p.get("reid") is not None for f in parsed["frames"][:: max(1, len(parsed["frames"]) // 50)]
           for p in f["people"]):
        _split_by_look(parsed["frames"], fps, log)
        picked = _pick_by_look(parsed["frames"], n, fps, log)
        if picked is not None:
            return picked
    sel = players_identity.identify_players(parsed, n, log, fps=fps, choose_only=True)
    if not sel or len(sel["players"]) < n:
        return None
    span, key = sel["span"], sel["key"]
    chains = sel["players"]
    sets = [set(ch["frames"]) for ch in chains]
    both = sorted(set.intersection(*sets))
    if not both:
        return None
    anchor = both[0]
    out = []
    for ch in chains:
        tid = next(t for t in ch["tids"] if anchor in span[t]["frames"])
        sp = span[tid]
        det = next(p for fr, p in sp["dets"] if fr == anchor)
        out.append({"ref": sp["proto"], "last": det, "tid": tid})
    log.info(f"players: Player 1/2 fixed at {anchor / fps:.1f}s (first moment the two most active "
             f"people are on screen together), then followed to both ends of the clip", stage="skeleton")
    return anchor, out, key, sel["dist"], sel["thr"]


def _fill_gaps(frames, n, fps) -> dict:
    """Fill SHORT gaps in a player's skeleton for the action model.

    A gap is safe to fill when the same player is labelled right before and
    right after it: both ends are fixed, so the frames in between belong to
    that person. The skeleton is interpolated between the two ends, joint by
    joint (only joints seen at both ends). These frames are marked
    "interpolated": they feed the action model (no zero frames in the middle
    of a technique) but are NOT drawn, and a training-set builder can leave
    them out.

    Re-labelling a detection inside the gap was tried and dropped: in a
    clinch the tracker can hand a fighter's id to the opponent or to someone
    walking past, and those frames then carried the wrong name - a wrong
    label is worse than a missing one. Gaps at the start / end of the clip,
    and gaps longer than GAP_FILL_S, stay empty: nothing anchors them."""
    max_gap = max(1, round(GAP_FILL_S * fps))
    out = {"interpolated": 0}

    def slot_det(f, k):
        return next((p for p in f["people"] if p.get("player_slot") == k), None)

    for k in range(n):
        labelled = [i for i, f in enumerate(frames) if slot_det(f, k) is not None]
        for a, b in zip(labelled, labelled[1:]):
            if b - a - 1 <= 0 or b - a - 1 > max_gap:
                continue
            start, end = slot_det(frames[a], k), slot_det(frames[b], k)
            if not start.get("centroid") or not end.get("centroid"):
                continue
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                kp = []
                for (xa, ya, sa), (xb, yb, sb) in zip(start["keypoints"], end["keypoints"]):
                    if sa > 0.1 and sb > 0.1:
                        kp.append((xa + t * (xb - xa), ya + t * (yb - ya), min(sa, sb)))
                    else:
                        kp.append((0.0, 0.0, 0.0))
                cen = (start["centroid"][0] + t * (end["centroid"][0] - start["centroid"][0]),
                       start["centroid"][1] + t * (end["centroid"][1] - start["centroid"][1]))
                frames[i]["people"].append({
                    "openpose_person_id": -1, "bbox": None, "bbox_score": None, "appearance": None,
                    "reid_idx": None, "keypoints": kp,
                    "mean_confidence": round(sum(s for _, _, s in kp) / len(kp), 4),
                    "valid_keypoints": sum(1 for _, _, s in kp if s > 0.1),
                    "centroid": cen, "player_slot": k, "interpolated": True,
                })
                out["interpolated"] += 1
    return out


def follow_players(parsed: dict, n_players: int, log: JobLogger, fps: float = 30.0) -> dict | None:
    frames = parsed["frames"]
    if not frames:
        return None
    fps = fps or 30.0
    cents = [p["centroid"] for f in frames for p in f["people"] if p.get("centroid")]
    if not cents:
        return None
    diag = math.hypot(max(c[0] for c in cents) - min(c[0] for c in cents),
                      max(c[1] for c in cents) - min(c[1] for c in cents)) or 1.0
    picked = _pick_players(parsed, n_players, fps, log)
    if picked is None:
        return None
    start_frame, players, key, dist, _ = picked
    start = next(i for i, f in enumerate(frames) if f["frame"] == start_frame)
    thr = _calibrate(frames, key, dist, log) if key else None
    if thr is None or any(pl["ref"] is None for pl in players):
        key = None  # too little data to trust looks: follow by motion + tracker id only
    thrs = [thr] * len(players)
    if key:
        # Per player: how close does ANYONE ELSE in this video come to their
        # look? Measured on the frames of the start piece (where we know which
        # detection is the player); the threshold sits just below that, so it
        # adapts to the clip, the camera and how alike the clothes are.
        for k, pl in enumerate(players):
            others = []
            for f in frames:
                me = next((p for p in f["people"] if p.get("track_id") == pl["tid"]), None)
                if me is None:
                    continue
                # skip crops mixed with the player (clinches): they are partly him
                others += [dist(p[key], pl["ref"]) for p in f["people"]
                           if p is not me and p.get(key) is not None
                           and _overlap(p, me) < 0.1 and _overlap(me, p) < 0.1]
            if len(others) >= 30:
                thrs[k] = max(thr, OTHERS_MARGIN * float(np.percentile(others, 2)))
        log.info("players: look threshold per player " + ", ".join(f"{t:.3f}" for t in thrs)
                 + " (just below the closest other person in this video)", stage="skeleton")
    reacquire = max(1, round(REACQUIRE_S * fps))
    allowed = [pl.get("allowed", set()) for pl in players]
    forbidden = [pl.get("forbidden", set()) for pl in players]
    # Each player's reference look stays FIXED for the whole clip. (Learning
    # extra views on the way was tried: a team-mate in the same shirt slipped
    # into the references and then took the label - a wrong label is worse
    # than a missing one.)
    refs = [pl["ref"] for pl in players]

    def look_at(sig, k):
        return dist(sig, refs[k])

    # Tracker ids KNOWN to be somebody else: seen as a separate person while
    # BOTH players were labelled elsewhere in the same frame (coach, referee,
    # team-mates...). Such an id can never become a player - whatever the
    # clothes look like. Ids are never reused by the tracker, so this holds
    # for the whole clip; shared by the forward and backward passes.
    third: dict = {}
    third_min = max(2, round(THIRD_S * fps))

    def run(order: list[int]):
        """Track both players through frames in the given order (forward or
        backward from the initialisation point)."""
        state = [{"pos": pl["last"]["centroid"], "vel": (0.0, 0.0), "h": _height(pl["last"]),
                  "tid": pl["tid"], "miss": 0}
                 for pl in players]
        for i in order:
            f = frames[i]
            cands = [p for p in f["people"] if p.get("centroid")]
            k_n = len(state)

            def cost(c, k):
                s = state[k]
                pred = (s["pos"][0] + s["vel"][0] * min(s["miss"] + 1, fps),
                        s["pos"][1] + s["vel"][1] * min(s["miss"] + 1, fps))
                gate = min(s["h"] * (GATE_H + GATE_GROW_H * s["miss"] / fps), GATE_MAX_FRAC * diag)
                if key and s["miss"] > LOST_S * fps:
                    gate = 2 * diag  # long gone: could be anywhere - the look decides
                if third.get(c.get("track_id"), 0) >= third_min:
                    return None  # a known third person
                if _body_center_and_scale(c["keypoints"]) is None:
                    return None  # no visible torso: a fragment or a false detection, not a body
                standing = players_identity._is_standing(c["keypoints"])
                if standing is False:
                    return None  # kumite is fought standing: someone sitting is not a player
                if standing is None and s["miss"] > reacquire:
                    return None  # re-acquiring needs legs we can see standing, not a half-hidden body
                if c.get("track_id") in forbidden[k]:
                    return None  # this piece was grouped into somebody else
                d = math.dist(c["centroid"], pred)
                if d > gate:
                    return None
                look = None
                if key and c.get(key) is not None:
                    look = look_at(c[key], k)
                    if look > thrs[k]:
                        return None  # does not look like this player
                    if any(look_at(c[key], j) < look for j in range(k_n) if j != k):
                        return None  # looks more like the OTHER player
                    if s["miss"] > reacquire:
                        if look > REACQUIRE_LOOK * thrs[k]:
                            return None  # re-acquiring needs a clearly matching look
                        # after losing them: must look more like THIS player
                        # than like the other one (no coach / team-mate takeover)
                        if any(look_at(c[key], j) <= look for j in range(k_n) if j != k):
                            return None
                elif s["miss"] > reacquire and c.get("track_id") != s["tid"]:
                    return None  # no look to check: only the same tracker id may re-acquire
                c_val = 0.5 * d / gate + (0.5 * look / thrs[k] if look is not None else 0.25)
                if c.get("track_id") is not None and c.get("track_id") == s["tid"]:
                    c_val -= SAME_ID_BONUS
                if c.get("track_id") in allowed[k]:
                    c_val -= GROUP_BONUS  # grouped with this player by look over the clip
                return c_val

            table = [[cost(c, k) for k in range(k_n)] for c in cands]
            opts = [None] + list(range(len(cands)))
            combos = ([(a,) for a in opts] if k_n == 1 else
                      [(a, b) for a in opts for b in opts if a is None or b is None or a != b])
            best, best_c = None, None
            for combo in combos:
                tot, ok = 0.0, True
                for k, ci in enumerate(combo):
                    if ci is None:
                        tot += 0.6  # leaving a player unlabelled costs, but less than a bad match
                    elif table[ci][k] is None:
                        ok = False
                        break
                    else:
                        tot += table[ci][k]
                if ok and (best_c is None or tot < best_c):
                    best, best_c = combo, tot
            for k, ci in enumerate(best or [None] * k_n):
                s = state[k]
                if ci is None:
                    s["miss"] += 1
                    continue
                c = cands[ci]
                c["player_slot"] = k
                n_since = s["miss"] + 1
                v = ((c["centroid"][0] - s["pos"][0]) / n_since, (c["centroid"][1] - s["pos"][1]) / n_since)
                s["vel"] = (0.7 * s["vel"][0] + 0.3 * v[0], 0.7 * s["vel"][1] + 0.3 * v[1])
                s["pos"], s["h"], s["tid"], s["miss"] = c["centroid"], _height(c), c.get("track_id"), 0
            if best and all(ci is not None for ci in best):
                mine = [cands[ci] for ci in best]
                for c in cands:
                    tid = c.get("track_id")
                    if (tid is not None and c not in mine and tid not in (st["tid"] for st in state)
                            and all(_overlap(c, m) < 0.3 and _overlap(m, c) < 0.3 for m in mine)):
                        third[tid] = third.get(tid, 0) + 1

    for f in frames:
        for p in f["people"]:
            p.pop("player_slot", None)
    run(list(range(start, len(frames))))       # forward from the start point
    run(list(range(start - 1, -1, -1)))        # and backward to the first frame

    # --- whole-clip 1<->2 check (only between the two players) -------------------
    swaps = 0
    if key and len(players) == 2:
        ev = {}
        for i, f in enumerate(frames):
            d = {p["player_slot"]: p for p in f["people"] if p.get("player_slot") is not None}
            if 0 in d and 1 in d and d[0].get(key) is not None and d[1].get(key) is not None:
                keep = look_at(d[0][key], 0) + look_at(d[1][key], 1)
                swap = look_at(d[0][key], 1) + look_at(d[1][key], 0)
                ev[i] = keep - swap
        # one second of look evidence of threshold size is needed to swap the labels
        switch = SWITCH_COST_S * fps * min(thrs)
        cost, back = [0.0, switch], []
        for i in range(len(frames)):
            e = ev.get(i, 0.0)
            em = (max(e, 0.0), max(-e, 0.0))
            new, ptr = [0.0, 0.0], [0, 0]
            for st in (0, 1):
                stay, flip = cost[st], cost[1 - st] + switch
                new[st], ptr[st] = (stay, st) if stay <= flip else (flip, 1 - st)
                new[st] += em[st]
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

    # --- short gaps: same player on both sides -> fill them ------------------------
    filled = _fill_gaps(frames, len(players), fps)
    log.info(f"players: short gaps (<= {GAP_FILL_S:.1f}s, same player on both sides) filled by "
             f"interpolation - {filled['interpolated']} frames (skeleton only, for the action model; "
             f"not drawn)", stage="skeleton")

    # --- the two players become the first tracks (sent to the action model) -----
    tagged = [0] * len(players)
    for f in frames:
        for p in f["people"]:
            k = p.get("player_slot")
            if k is not None:
                p["track_id"] = PLAYER_TRACK_BASE + k
                tagged[k] += 1
    tracks = parsed.get("tracks") or []
    head = [{"track_id": PLAYER_TRACK_BASE + k, "frames_present": tagged[k], "mean_confidence": 1.0,
             "score": 1e9 - k, "player": f"Player {k + 1}"} for k in range(len(players))]
    parsed["tracks"] = head + [t for t in tracks if t["track_id"] < PLAYER_TRACK_BASE]
    parsed["track_count"] = len(parsed["tracks"])
    parsed["tracker"] = (parsed.get("tracker") or "tracker") + " + follow-the-two-players"
    log.info("players: followed from the start - " + ", ".join(
        f"Player {k + 1} in {tagged[k]}/{len(frames)} frames" for k in range(len(players)))
        + f"; whole-clip 1<->2 corrections: {swaps}", stage="skeleton")
    return {"players": len(players), "frames_tagged": tagged, "swap_corrections": swaps,
            "signature": key, "look_threshold": thrs}
