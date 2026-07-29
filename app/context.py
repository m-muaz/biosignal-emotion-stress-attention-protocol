import random
from dataclasses import dataclass

from app.eventlog.event_logger import EventLogger


@dataclass
class SessionContext:
    """Bag of everything a task module needs, passed through main.py.

    demo_scale multiplies every duration read via ctx.scaled(seconds) --
    keep it at 1.0 for a real session, set e.g. 0.1 to walk the whole
    protocol in a few minutes for a demo run.
    """

    config: dict
    event_logger: EventLogger
    participant_id: str
    demo_scale: float
    rng: random.Random

    def scaled(self, seconds: float) -> float:
        return seconds * self.demo_scale
