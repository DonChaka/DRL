from collections import deque
from typing import Union

import numpy as np
import skimage
from numpy import ndarray


class StateWrapper:
    # contain last 3 frames to detect motion
    def __init__(self, resolution):
        self.resolution = resolution
        self.last_frames = deque([], maxlen=3)
        self.reset()

    def __call__(self, frame: Union[ndarray, None] = None) -> ndarray:
        if frame is not None:
            self.add_frame(frame)
        return self.get_state()

    def _preprocess(self, frame):
        frame = skimage.color.rgb2gray(frame, channel_axis=0)
        frame = frame / 255.0
        frame = skimage.transform.resize(frame, self.resolution)
        frame = frame.astype(np.float32)
        return frame

    def get_state(self) -> ndarray:
        return np.concatenate(self.last_frames, axis=2)

    def add_frame(self, frame):
        self.last_frames.append(self._preprocess(frame))

    def reset(self) -> ndarray:
        self.last_frames.clear()
        for _ in range(3):
            self.last_frames.append(np.zeros(self.resolution))
        return self.get_state()
