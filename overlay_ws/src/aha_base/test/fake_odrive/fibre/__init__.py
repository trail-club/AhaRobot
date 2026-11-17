"""Stand-in for odrive's bundled fibre: only Event is used."""

import threading


class Event:
    def __init__(self):
        self._evt = threading.Event()
        self._subscribers = []

    def is_set(self):
        return self._evt.is_set()

    def set(self):
        if not self._evt.is_set():
            self._evt.set()
            for cb in list(self._subscribers):
                cb()

    def subscribe(self, cb):
        self._subscribers.append(cb)

    def unsubscribe(self, cb):
        self._subscribers.remove(cb)
