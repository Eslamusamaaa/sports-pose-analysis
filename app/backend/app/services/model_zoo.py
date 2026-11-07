"""Registry of skeleton models and the concept map that lets them be combined.

WHY AN ENSEMBLE AT ALL
----------------------
No public checkpoint knows karate. But several know *striking*:

    NTU RGB+D 60   punching/slapping other person, kicking other person
    HMDB51         punch, kick, hit
    UCF101         Punch, BoxingPunchingBag, BoxingSpeedBag
    Kinetics-400   punching person (boxing), side kick, high kick, drop kicking

Each is a weak, out-of-domain signal on its own. Asking all of them and
combining agreement is more informative than trusting any single one - if three
independent models all fire "kick" in the same second, that is real evidence
something kick-shaped happened, even though none of them can name the technique.

WHY SCORES CANNOT SIMPLY BE AVERAGED
------------------------------------
The models have different label spaces (60 / 51 / 101 / 400 classes). Index 50
means "kicking other person" in one and something unrelated in another, so
averaging raw score vectors is meaningless. Instead each model's relevant
classes are mapped onto a small set of shared CONCEPTS, and evidence is pooled
per concept.

Probabilities are also not comparable across models: a 400-class softmax spreads
mass far thinner than a 51-class one, so 0.10 from Kinetics is a much stronger
signal than 0.10 from HMDB51. Scores are therefore normalised by each model's
uniform baseline (1/num_classes) before pooling - see ensemble_service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import PARENT_DIR, PROJECT_ROOT

CONFIGS = PARENT_DIR / "mmaction2-main" / "configs" / "skeleton"
MODELS = PROJECT_ROOT / "models" / "mmaction"

# Shared concepts. Deliberately coarse: these checkpoints cannot distinguish
# gyaku-zuki from oi-zuki, and pretending otherwise would be dishonest.
PUNCH = "punch"
KICK = "kick"
GRAPPLE = "grapple/push"
MARTIAL = "martial-arts-like"
MOVE = "moving/footwork"

CONCEPT_ORDER = [PUNCH, KICK, GRAPPLE, MARTIAL, MOVE]

CONCEPT_AR = {
    PUNCH: "لكمة",
    KICK: "ركلة",
    GRAPPLE: "دفع/اشتباك",
    MARTIAL: "حركة قتالية",
    MOVE: "تحرك/فوت ورك",
}


# Default ensemble membership.
#
# MEASURED on a real karate match (job 8c1eb71c1cd4, 15-30s segment):
# the HMDB51 / UCF101 / Kinetics-400 PoseC3D checkpoints returned labels like
# "HulaHoop", "MoppingFloor", "riding unicycle", "catching or throwing frisbee"
# and "robot dancing" - not weak predictions, essentially random ones. Pooling
# them pushed the meaningless "martial-arts-like" concept (fed by TaiChi and
# capoeira) to a higher mean lift (70.2) than "punch" (19.1), i.e. they added
# noise, not signal.
#
# Diagnosis: those models are trained on single-subject clips where the person
# fills the frame, and PoseC3D rasterises the skeleton into a 64x64 heatmap. In
# a 1920x1080 wide shot with two distant athletes plus officials, that is far
# outside their training distribution.
#
# The two NTU60 checkpoints stayed coherent and, importantly, AGREED with each
# other across different architectures (ST-GCN / GCN vs PoseC3D / 3D-CNN) on the
# same punch at t=21s. Cross-architecture agreement on one dataset is the only
# signal here that survived scrutiny, so it is the default.
#
# The others stay in the registry - re-enable via ENSEMBLE_MODELS if the footage
# is closer to their domain (single athlete, close framing).
DEFAULT_ENSEMBLE = ("stgcn_ntu60", "posec3d_ntu60")


@dataclass
class ModelSpec:
    key: str
    name: str
    dataset: str
    config: Path
    checkpoint: Path
    label_map: Path
    num_classes: int
    # Label text -> concept. Matched case-insensitively against the label map at
    # load time, so a renamed or reordered class map cannot silently mis-map.
    concepts: dict[str, str] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.config.exists() and self.checkpoint.exists() and self.label_map.exists()


REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="stgcn_ntu60",
        name="ST-GCN",
        dataset="NTU RGB+D 60",
        config=CONFIGS / "stgcn" / "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
        checkpoint=MODELS / "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
        label_map=MODELS / "label_map_ntu60.txt",
        num_classes=60,
        concepts={
            "punching/slapping other person": PUNCH,
            "kicking other person": KICK,
            "kicking something": KICK,
            "pushing other person": GRAPPLE,
            "hugging other person": GRAPPLE,
            "walking towards each other": MOVE,
            "walking apart from each other": MOVE,
        },
    ),
    ModelSpec(
        key="posec3d_ntu60",
        name="PoseC3D",
        dataset="NTU RGB+D 60",
        config=CONFIGS / "posec3d" / "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py",
        checkpoint=MODELS / "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth",
        label_map=MODELS / "label_map_ntu60.txt",
        num_classes=60,
        concepts={
            "punching/slapping other person": PUNCH,
            "kicking other person": KICK,
            "kicking something": KICK,
            "pushing other person": GRAPPLE,
            "walking towards each other": MOVE,
            "walking apart from each other": MOVE,
        },
    ),
    ModelSpec(
        key="posec3d_hmdb51",
        name="PoseC3D",
        dataset="HMDB51",
        config=CONFIGS / "posec3d"
        / "slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py",
        checkpoint=MODELS
        / "slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth",
        label_map=MODELS / "label_map_hmdb51.txt",
        num_classes=51,
        concepts={
            "punch": PUNCH,
            "hit": PUNCH,
            "kick": KICK,
            "kick_ball": KICK,
            "sword_exercise": MARTIAL,
            "sword": MARTIAL,
            "walk": MOVE,
            "run": MOVE,
        },
    ),
    ModelSpec(
        key="posec3d_ucf101",
        name="PoseC3D",
        dataset="UCF101",
        config=CONFIGS / "posec3d"
        / "slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.py",
        checkpoint=MODELS
        / "slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint_20220815-9972260d.pth",
        label_map=MODELS / "label_map_ucf101.txt",
        num_classes=101,
        concepts={
            "Punch": PUNCH,
            "BoxingPunchingBag": PUNCH,
            "BoxingSpeedBag": PUNCH,
            "SumoWrestling": GRAPPLE,
            "TaiChi": MARTIAL,
            "Nunchucks": MARTIAL,
            "Fencing": MARTIAL,
            "WalkingWithDog": MOVE,
        },
    ),
    ModelSpec(
        key="posec3d_k400",
        name="PoseC3D",
        dataset="Kinetics-400",
        config=CONFIGS / "posec3d" / "slowonly_r50_8xb32-u48-240e_k400-keypoint.py",
        checkpoint=MODELS / "slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth",
        label_map=MODELS / "label_map_k400.txt",
        num_classes=400,
        concepts={
            "punching person (boxing)": PUNCH,
            "punching bag": PUNCH,
            "side kick": KICK,
            "high kick": KICK,
            "drop kicking": KICK,
            "wrestling": GRAPPLE,
            "arm wrestling": GRAPPLE,
            "capoeira": MARTIAL,
            "tai chi": MARTIAL,
            "jogging": MOVE,
            "walking the dog": MOVE,
        },
    ),
]


def available_models() -> list[ModelSpec]:
    return [m for m in REGISTRY if m.available]


def resolve_concept_indices(spec: ModelSpec) -> dict[int, str]:
    """Map class INDEX -> concept by looking the label text up in the map file.

    Resolved from the file rather than hardcoded indices so that a label map
    which is reordered, or does not match the checkpoint, fails loudly (missing
    concepts) instead of silently mapping the wrong classes.
    """
    try:
        labels = [
            ln.strip() for ln in spec.label_map.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except OSError:
        return {}
    lower = {lbl.lower(): i for i, lbl in enumerate(labels)}
    out: dict[int, str] = {}
    for text, concept in spec.concepts.items():
        idx = lower.get(text.lower())
        if idx is not None:
            out[idx] = concept
    return out
