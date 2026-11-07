"""Step 3: ask a local VLM (llama.cpp server) which technique each clip shows.

Starts llama-server with one model, sends every clip image with the list of
techniques, forces the answer to be one of the 12 labels (JSON schema), and
saves results_<model>.json. One model at a time (4 GB GPU).

usage: python run_vlm.py <model key> [clip names ...]
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
VLM = Path(r"D:/Users/karate-training/vlm")
SERVER = VLM / "llama" / "llama-server.exe"
PORT = 8090

MODELS = {
    "qwen3vl": ("Qwen3-VL-4B-Instruct", "qwen3vl-4b.gguf", "qwen3vl-4b-mmproj.gguf"),
    "qwen35": ("Qwen3.5-4B", "qwen35-4b.gguf", "qwen35-4b-mmproj.gguf"),
    "gemma3": ("Gemma 3 4B", "gemma3-4b.gguf", "gemma3-4b-mmproj.gguf"),
}

LABELS = [
    "FEINT", "GYAKU_ZUKI", "KIZAMI_GYAKU_COMBINATION", "KIZAMI_MAWASHI_GERI_CHUDAN",
    "KIZAMI_MAWASHI_GERI_JODAN", "KIZAMI_URA_MAWASHI_GERI_JODAN", "KIZAMI_ZUKI",
    "MAWASHI_GERI_CHUDAN", "MAWASHI_GERI_JODAN", "MOVEMENT", "URA_MAWASHI_GERI_JODAN",
    "YOKO_GERI_CHUDAN",
]

PROMPT = """You are an expert WKF karate kumite referee and technique analyst.

THE IMAGE
It shows 8 frames of a 1.2 second clip of a kumite bout, in time order. Frame numbers are in the top-left corner of each frame: row 1 = frames 1-2, row 2 = frames 3-4, row 3 = frames 5-6, row 4 = frames 7-8 (left to right, top to bottom). Successive frames are about 0.17 s apart, so a fast kick may be visible in only 1-3 frames.
- PLAYER 1 is the fighter inside the BLUE box.
- PLAYER 2 is the fighter inside the ORANGE box.
- A box can be missing in some frames (the fighter is hidden or blurred). If a fighter has no box in any frame, answer MOVEMENT for that fighter with confidence 0.
- Ignore everyone else (referee, coaches, spectators sitting around the mat).

KEY IDEAS
- LEAD (front) side = the hand / leg closer to the opponent in the fighting stance. REAR side = the one further back.
- KIZAMI means the technique is done with the LEAD hand or LEAD leg. Without "KIZAMI", kicks are done with the REAR leg.
- CHUDAN = target is the body (stomach, ribs, chest). JODAN = target is the head / neck (the foot or fist reaches head height).
- A technique counts only if it is clearly launched toward the opponent. Normal guarding, bouncing and stepping are MOVEMENT.

THE 12 LABELS
1. KIZAMI_ZUKI - lead-hand straight punch (jab). The front arm shoots straight out toward the opponent's head or body, usually with a forward slide of the front foot; the rear hand stays at the chin; hips stay mostly sideways.
2. GYAKU_ZUKI - rear-hand straight punch (reverse punch). The arm on the side of the BACK leg extends straight; hips and shoulders rotate to face the opponent; usually a lunge forward with the front knee bent.
3. KIZAMI_GYAKU_COMBINATION - two punches in a row inside the clip: first a lead-hand jab, then immediately a rear-hand reverse punch. You must see both arms extend one after the other.
4. MAWASHI_GERI_CHUDAN - roundhouse kick with the REAR leg to the body. The back knee lifts and comes forward/across, the hip turns over, the lower leg snaps around horizontally; the foot (instep) reaches stomach / rib height.
5. MAWASHI_GERI_JODAN - the same rear-leg roundhouse kick, but the foot reaches the opponent's head height (higher than the shoulders).
6. KIZAMI_MAWASHI_GERI_CHUDAN - roundhouse kick with the FRONT leg to the body. The lead knee lifts (often after a small skip forward of the back foot), no full step through; the foot snaps around to stomach / rib height.
7. KIZAMI_MAWASHI_GERI_JODAN - the same front-leg roundhouse kick, to head height.
8. URA_MAWASHI_GERI_JODAN - reverse roundhouse / hook kick with the REAR leg to the head. The leg swings ACROSS the body and the heel or sole hooks BACKWARD toward the opponent's head (opposite direction to mawashi geri); the kicker often turns the back partly toward the opponent.
9. KIZAMI_URA_MAWASHI_GERI_JODAN - the same hook kick to the head, done with the FRONT leg: the lead knee lifts, the leg extends across and the heel hooks back to the head.
10. YOKO_GERI_CHUDAN - side kick to the body. The knee chambers up, then the leg thrusts out STRAIGHT in a line; hip, knee and foot edge are aligned and the body leans sideways away from the kick.
11. FEINT - a fake attack: the fighter starts a punch or kick (a sharp shoulder, hand, knee or hip movement toward the opponent) but stops early and does not extend it to the target.
12. MOVEMENT - no attack at all: footwork, stepping in or out, bouncing in stance, circling, guarding, blocking, standing, walking, resting, the bout being stopped, or the fighter not visible.

HOW TO DECIDE (fill the answer fields in this order, for each player)
1. what_happened: describe in one sentence what the fighter's hands and legs do, frame by frame.
2. attack: "none" (no attack), "feint" (attack started but not completed), "punch" or "kick".
3. side: which hand / leg attacks: "lead", "rear", or "lead_then_rear" (jab followed by reverse punch). "none" when there is no attack.
4. kick_type: for kicks only: "roundhouse" (mawashi), "hook" (ura mawashi), "side" (yoko). Otherwise "none".
5. height: "chudan" (body) or "jodan" (head). "none" when there is no attack.
6. label: the technique that matches steps 2-5:
   punch + lead = KIZAMI_ZUKI | punch + rear = GYAKU_ZUKI | punch + lead_then_rear = KIZAMI_GYAKU_COMBINATION
   kick + rear + roundhouse + chudan = MAWASHI_GERI_CHUDAN | kick + rear + roundhouse + jodan = MAWASHI_GERI_JODAN
   kick + lead + roundhouse + chudan = KIZAMI_MAWASHI_GERI_CHUDAN | kick + lead + roundhouse + jodan = KIZAMI_MAWASHI_GERI_JODAN
   kick + rear + hook + jodan = URA_MAWASHI_GERI_JODAN | kick + lead + hook + jodan = KIZAMI_URA_MAWASHI_GERI_JODAN
   kick + side + chudan = YOKO_GERI_CHUDAN | feint = FEINT | none = MOVEMENT
7. confidence: 0 to 1, how sure you are of the label.
If two techniques happen in the clip, describe the main / most complete attack. When you are not sure a real attack happened, answer MOVEMENT."""

PLAYER = {
    "type": "object",
    "properties": {  # order matters: the model reasons through these before the label
        "what_happened": {"type": "string"},
        "attack": {"type": "string", "enum": ["none", "feint", "punch", "kick"]},
        "side": {"type": "string", "enum": ["none", "lead", "rear", "lead_then_rear"]},
        "kick_type": {"type": "string", "enum": ["none", "roundhouse", "hook", "side"]},
        "height": {"type": "string", "enum": ["none", "chudan", "jodan"]},
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["what_happened", "attack", "side", "kick_type", "height", "label", "confidence"],
}

# label -> the decision steps it implies (to flag answers that contradict themselves)
IMPLIES = {
    "KIZAMI_ZUKI": ("punch", "lead", None, None), "GYAKU_ZUKI": ("punch", "rear", None, None),
    "KIZAMI_GYAKU_COMBINATION": ("punch", "lead_then_rear", None, None),
    "MAWASHI_GERI_CHUDAN": ("kick", "rear", "roundhouse", "chudan"),
    "MAWASHI_GERI_JODAN": ("kick", "rear", "roundhouse", "jodan"),
    "KIZAMI_MAWASHI_GERI_CHUDAN": ("kick", "lead", "roundhouse", "chudan"),
    "KIZAMI_MAWASHI_GERI_JODAN": ("kick", "lead", "roundhouse", "jodan"),
    "URA_MAWASHI_GERI_JODAN": ("kick", "rear", "hook", "jodan"),
    "KIZAMI_URA_MAWASHI_GERI_JODAN": ("kick", "lead", "hook", "jodan"),
    "YOKO_GERI_CHUDAN": ("kick", None, "side", "chudan"),
    "FEINT": ("feint", None, None, None), "MOVEMENT": ("none", None, None, None),
}


def consistent(a: dict) -> bool:
    want = IMPLIES.get(a.get("label"))
    if not want:
        return False
    got = (a.get("attack"), a.get("side"), a.get("kick_type"), a.get("height"))
    return all(w is None or w == g for w, g in zip(want, got))


SCHEMA = {"type": "object", "properties": {"player1": PLAYER, "player2": PLAYER},
          "required": ["player1", "player2"]}


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited, see the server log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready")


def main() -> int:
    key = sys.argv[1]
    names = sys.argv[2:] or ["match1", "match2", "match3", "balkas"]
    title, model, mmproj = MODELS[key]
    log = open(HERE / f"server_{key}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(VLM / "models" / model), "--mmproj", str(VLM / "models" / mmproj),
         "-ngl", "99", "-c", "8192", "--port", str(PORT), "--jinja", "-fa", "on", "-np", "1"],
        stdout=log, stderr=subprocess.STDOUT)
    results = []
    try:
        wait_ready(proc)
        for name in names:
            data = json.loads((HERE / f"windows_{name}.json").read_text(encoding="utf-8"))
            only = os.environ.get("ONLY")  # e.g. ONLY=54,55,56 for a quick check
            for w in (data["windows"] if not only else [data["windows"][int(i)] for i in only.split(",")]):
                img = base64.b64encode((HERE / w["image"]).read_bytes()).decode()
                body = {
                    "messages": [{"role": "user", "content": [
                        # fixed text first, so the server reuses it (prompt cache) for every window
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}],
                    "temperature": 0, "max_tokens": int(os.environ.get("MAX_TOKENS", 500)),
                    "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}},
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                t0 = time.time()
                try:
                    r = post("/v1/chat/completions", body)
                    ans = json.loads(r["choices"][0]["message"]["content"])
                except Exception as e:  # keep going; record the failure
                    err = {"label": None, "confidence": None, "what_happened": f"error: {e}"}
                    ans = {"player1": err, "player2": dict(err)}
                for k in (0, 1):  # a player with no box in the whole window was not seen
                    if not any(w["boxes"][k]):
                        ans[f"player{k + 1}"] = {"what_happened": "not visible in this window", "attack": "none",
                                                 "side": "none", "kick_type": "none", "height": "none",
                                                 "label": "MOVEMENT", "confidence": 0}
                    pl = ans[f"player{k + 1}"]
                    pl["consistent"] = consistent(pl) if pl.get("label") else None
                ans.update({"video": name, "image": w["image"], "start_s": w["start_s"],
                            "end_s": w["end_s"], "seconds": round(time.time() - t0, 1)})
                results.append(ans)
                print(f"{key} {name} {w['start_s']:>6}s  P1 {ans['player1']['label']}  "
                      f"P2 {ans['player2']['label']}  {ans['seconds']}s", flush=True)
                # save as we go: a long run survives a crash
                (HERE / f"results_{key}{os.environ.get('TAG', '')}.json").write_text(json.dumps({"model": title, "results": results},
                                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()
    (HERE / f"results_{key}{os.environ.get('TAG', '')}.json").write_text(json.dumps({"model": title, "results": results},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
