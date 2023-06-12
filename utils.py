from collections import deque
from typing import Union

import numpy as np
import skimage
from numpy import ndarray


class StateWrapper:
    # contain last 3 frames to detect motion
    def __init__(self, resolution, state_shape):
        self.resolution = resolution
        self.state_shape = state_shape
        self.last_frames = deque([], maxlen=3)
        self.last_stats = deque([], maxlen=3)
        self.reset()

    def __call__(self, frame: Union[ndarray, None] = None, state: Union[ndarray, None] = None) -> tuple[ndarray, ndarray]:
        if frame is not None and state is not None:
            self.add_frame(frame, state)
        return self.get_state()

    def _preprocess(self, frame):
        frame = skimage.color.rgb2gray(frame, channel_axis=0)
        frame = frame / 255.0
        frame = skimage.transform.resize(frame, self.resolution)
        frame = frame.astype(np.float32)
        return frame

    def get_state(self) -> tuple[ndarray, ndarray]:
        return np.concatenate(self.last_frames, axis=2), np.concatenate(self.last_stats, axis=0)

    def add_frame(self, frame, state):
        self.last_frames.append(self._preprocess(frame))
        self.last_stats.append(state)

    def reset(self) -> tuple[ndarray, ndarray]:
        self.last_frames.clear()
        self.last_stats.clear()
        for _ in range(3):
            self.last_frames.append(np.zeros(self.resolution))
            self.last_stats.append(np.zeros(self.state_shape))
        return self.get_state()
