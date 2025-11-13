from pathlib import Path

FPS = 10
DURATION = 5.0
OUTDIR = Path("shot_out")
VIDEO_NAME = "shot.mp4"
FRAME_SIZE = (int(1.6 * 720), 720)
CAMERA_NAME = "7_foot_offcenter"
SUMMARY_NAME = "shot_summary.json"
FRAME_PREFIX = "frame"
FRAME_PATTERN = f"{FRAME_PREFIX}_%06d.png"
POCKET_ORDER = ("lb", "lc", "lt", "rb", "rc", "rt")
POCKET_COLOR_MAP: dict[str, str] = {
    "lb": "red",
    "lc": "orange",
    "lt": "yellow",
    "rb": "green",
    "rc": "blue",
    "rt": "purple",
}
CUSHION_COLOR_LOOKUP: dict[int, str] = {
    1: "red-green-wall",
    2: "orange-red-wall",
    3: "orange-red-wall",
    4: "orange-red-wall",
    5: "yellow-orange-wall",
    6: "yellow-orange-wall",
    7: "yellow-orange-wall",
    8: "purple-yellow-wall",
    9: "purple-yellow-wall",
    10: "purple-yellow-wall",
    11: "blue-purple-wall",
    12: "blue-purple-wall",
    13: "blue-purple-wall",
    14: "green-blue-wall",
    15: "green-blue-wall",
    16: "green-blue-wall",
    17: "red-green-wall",
    18: "red-green-wall",
    19: "purple-yellow-wall",
    20: "blue-purple-wall",
    21: "blue-purple-wall",
    22: "green-blue-wall",
    23: "green-blue-wall",
    24: "red-green-wall",
    25: "red-green-wall",
    26: "orange-red-wall",
    27: "orange-red-wall",
    28: "yellow-orange-wall",
    29: "yellow-orange-wall",
    30: "purple-yellow-wall",
}
