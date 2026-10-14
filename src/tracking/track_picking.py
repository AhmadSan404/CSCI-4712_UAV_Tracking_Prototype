"""Picking a track with the mouse (click) or the keyboard (Tab)."""

from src.tracking.track import Track

CLICK_MIN_PADDING_PX = 12.0  # extra clickable margin around a box, so tiny top-down cars can still be clicked
CLICK_PADDING_FRACTION = 0.15  # or this share of the box's longest side, whichever is larger


def find_track_at_point(tracks:list[Track], x:float, y:float) -> Track | None:
    """Finds the track under a click, preferring the smallest box when several overlap."""
    hits = []
    for track in tracks:
        x1, y1, x2, y2 = track.box
        padding = max(CLICK_MIN_PADDING_PX, CLICK_PADDING_FRACTION * max(x2 - x1, y2 - y1))
        if x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding:
            hits.append(((x2 - x1) * (y2 - y1), track))
    # If boxes overlap, the smallest is most likely the one the user meant.
    return min(hits, key=lambda hit: hit[0])[1] if hits else None


def find_next_track(tracks:list[Track], current_track_id:int|None) -> Track | None:
    """The next track left to right after the current one, wrapping around (Tab key). Coasting tracks are skipped."""
    candidates = sorted((track for track in tracks if not track.is_coasting), key=lambda track: track.center[0])
    if not candidates:
        return None
    current = next((track for track in tracks if track.track_id == current_track_id), None)
    if current is None:
        return candidates[0]
    return next((track for track in candidates if track.center[0] > current.center[0]), candidates[0])
