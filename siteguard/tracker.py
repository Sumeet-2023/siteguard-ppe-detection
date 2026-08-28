from collections import defaultdict, deque


class ViolationMonitor:
    """Turns per-frame detections into stable, deduplicated events."""

    def __init__(self, fps: float, min_violation_sec: float = 2.0):
        self.min_frames = int(fps * min_violation_sec)
        self.history = defaultdict(lambda: deque(maxlen=30))
        self.streak = defaultdict(int)
        self.reported = set()
        self.events = []

    def update(self, track_id: int, cls_name: str, frame_idx: int) -> dict | None:
        has_helmet = cls_name == "helmet"
        self.history[track_id].append(has_helmet)
        self.streak[track_id] = 0 if has_helmet else self.streak[track_id] + 1

        if self.streak[track_id] >= self.min_frames and track_id not in self.reported:
            self.reported.add(track_id)
            event = {"track_id": track_id, "type": "no_helmet", "frame": frame_idx}
            self.events.append(event)
            return event
        return None

    def summary(self) -> dict:
        compliant = sum(
            sum(h) >= len(h) / 2 for h in self.history.values() if h
        )
        n = max(len(self.history), 1)
        return {"unique_people": len(self.history),
                "violations": len(self.events),
                "compliance_rate": round(compliant / n, 3)}
