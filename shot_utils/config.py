from pathlib import Path

FPS = 15
BASE_OUTPUT = Path("outputs")
GLOBAL_INDEX_PATH = BASE_OUTPUT / "global_index.json"
FRAME_SIZE = (int(1.6 * 720), 720)
# 7_foot_corner_pocket_zoom
# 7_foot_offcenter (default)
# 7_foot_overhead
# 7_foot_overhead_zoom
# 7_foot_side_pocket_zoom
CAMERA_NAME = "7_foot_offcenter"
# Move the camera slightly farther from the fixation point so the full table and
# pocket markers remain in frame even when markers sit outside the rails.
CAMERA_DISTANCE_OFFSET = 0.35
FRAME_PREFIX = "frame"
FRAME_PATTERN = f"{FRAME_PREFIX}_%06d.png"
POCKET_ORDER = ("lb", "lc", "lt", "rb", "rc", "rt")
POCKET_COLOR_MAP: dict[str, str] = {
    "lb": "red",
    "lc": "orange",
    "lt": "grey",
    "rb": "green",
    "rc": "blue",
    "rt": "purple",
}
CUSHION_COLOR_LOOKUP: dict[int, str] = {
    1: "red-green-wall",
    2: "orange-red-wall",
    3: "orange-red-wall",
    4: "orange-red-wall",
    5: "grey-orange-wall",
    6: "grey-orange-wall",
    7: "grey-orange-wall",
    8: "purple-grey-wall",
    9: "purple-grey-wall",
    10: "purple-grey-wall",
    11: "blue-purple-wall",
    12: "blue-purple-wall",
    13: "blue-purple-wall",
    14: "green-blue-wall",
    15: "green-blue-wall",
    16: "green-blue-wall",
    17: "red-green-wall",
    18: "red-green-wall",
    19: "purple-grey-wall",
    20: "blue-purple-wall",
    21: "blue-purple-wall",
    22: "green-blue-wall",
    23: "green-blue-wall",
    24: "red-green-wall",
    25: "red-green-wall",
    26: "orange-red-wall",
    27: "orange-red-wall",
    28: "grey-orange-wall",
    29: "grey-orange-wall",
    30: "purple-grey-wall",
}
