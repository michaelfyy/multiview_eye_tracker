# utils/timer.py
import time
import logging

logger = logging.getLogger(__name__)

class Timer:
    """A simple timer class to track epoch durations and total training time."""

    def __init__(self):
        self._total_start_time = time.monotonic() # Record when training starts overall
        self._epoch_start_time = None
        self._last_epoch_duration = None
        logger.debug("Timer initialized.")

    def start_epoch(self):
        """Records the start time of an epoch."""
        self._epoch_start_time = time.monotonic()
        self._last_epoch_duration = None # Reset duration for the new epoch
        # logger.debug("Epoch timer started.")

    def end_epoch(self):
        """Records the end time of an epoch and calculates its duration."""
        if self._epoch_start_time is None:
            logger.warning("Timer end_epoch() called before start_epoch(). Duration will be invalid.")
            self._last_epoch_duration = 0.0
        else:
            epoch_end_time = time.monotonic()
            self._last_epoch_duration = epoch_end_time - self._epoch_start_time
            # logger.debug(f"Epoch timer stopped. Duration: {self._last_epoch_duration:.2f}s")
        # Reset epoch start time after calculating duration
        self._epoch_start_time = None

    def get_epoch_duration(self) -> float:
        """Returns the duration of the last completed epoch in seconds."""
        if self._last_epoch_duration is None:
            logger.warning("get_epoch_duration() called before end_epoch() was successfully called for the epoch.")
            return 0.0
        return self._last_epoch_duration

    def get_total_duration(self) -> float:
        """Returns the total time elapsed since the Timer was initialized."""
        current_time = time.monotonic()
        return current_time - self._total_start_time