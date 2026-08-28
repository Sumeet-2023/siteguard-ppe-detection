import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from siteguard.tracker import ViolationMonitor


def test_no_event_before_min_duration():
    # fps=10, min_violation_sec=2.0 -> needs 20 consecutive no-helmet frames
    mon = ViolationMonitor(fps=10, min_violation_sec=2.0)
    for i in range(19):
        event = mon.update(track_id=1, cls_name="head", frame_idx=i)
        assert event is None


def test_event_fires_once_at_threshold():
    mon = ViolationMonitor(fps=10, min_violation_sec=2.0)
    event = None
    for i in range(25):
        event = mon.update(track_id=1, cls_name="head", frame_idx=i) or event
    assert event is not None
    assert event["track_id"] == 1
    assert event["type"] == "no_helmet"
    # streak keeps growing but the event must not re-fire
    assert len(mon.events) == 1


def test_helmet_resets_streak_and_prevents_event():
    mon = ViolationMonitor(fps=10, min_violation_sec=2.0)
    for i in range(15):
        mon.update(track_id=1, cls_name="head", frame_idx=i)
    event = mon.update(track_id=1, cls_name="helmet", frame_idx=15)
    assert event is None
    assert mon.streak[1] == 0


def test_separate_tracks_are_independent():
    mon = ViolationMonitor(fps=10, min_violation_sec=2.0)
    for i in range(25):
        mon.update(track_id=1, cls_name="head", frame_idx=i)
    for i in range(5):
        mon.update(track_id=2, cls_name="helmet", frame_idx=i)

    summary = mon.summary()
    assert summary["unique_people"] == 2
    assert summary["violations"] == 1


def test_compliance_rate_majority_vote():
    mon = ViolationMonitor(fps=10, min_violation_sec=2.0)
    # mostly helmet, one bad frame -> still compliant
    for cls in ["helmet", "helmet", "helmet", "head"]:
        mon.update(track_id=1, cls_name=cls, frame_idx=0)
    assert mon.summary()["compliance_rate"] == 1.0
