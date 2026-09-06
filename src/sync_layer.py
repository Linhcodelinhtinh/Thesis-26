#!/usr/bin/env python3
"""
Synchronization Layer
---------------------
Coordinates asynchronous sharing of inputs between components running at different frequencies.
- Thread 1 (Tracker): 10-20 Hz
- Thread 2 (Chronicler): 5-10 Hz
- Main Thread (VLA Engine): 10 Hz
"""
import threading

class SyncLayer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_prompt = None

    def update_image(self, painted_image):
        """
        Updates the latest painted image. Called by the Tracker thread.
        """
        with self.lock:
            self.latest_image = painted_image

    def update_prompt(self, dynamic_prompt):
        """
        Updates the latest logical prompt. Called by the Chronicler thread.
        """
        with self.lock:
            self.latest_prompt = dynamic_prompt

    def get_latest_state(self):
        """
        Retrieves the latest available image and prompt.
        Called by the VLA inference engine. Non-blocking.
        """
        with self.lock:
            return self.latest_image, self.latest_prompt

if __name__ == "__main__":
    sync = SyncLayer()
    sync.update_image("dummy_image")
    sync.update_prompt("dummy_prompt")
    img, prt = sync.get_latest_state()
    print(f"Sync Layer test: Image='{img}', Prompt='{prt}'")
