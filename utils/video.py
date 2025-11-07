"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Any, Final, Literal, Iterator, Mapping, Optional, Sequence, Tuple, Union
from fractions import Fraction
from dataclasses import dataclass, replace
import torch as th
import av
import numpy as np
import os

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_FRAMERATE: Final[int] = 30
DEFAULT_CRF: Final[int] = 23
DEFAULT_VCODEC: Final[str] = "libx264"
DEFAULT_ACODEC: Final[str] = "aac"
DEFAULT_AUDIO_LAYOUT: Final[str] = "mono"

DataType = Union[np.typing.NDArray, th.Tensor]


@dataclass(frozen=True)
class VideoObj:
    """Contains media data to add to a video file
    Args:
        audio: optional np.ndarray with audio data (n_channels, n_samples)
            n_channels can be 1 (mono) or 2 (stereo)
        video: optional np.ndarray with video data
            MP4File -> (n_frames, h, w, n_channels)
            StreamingMP4File.write() -> (h, w, n_channels)
        video_ts: optional list of timestamps in seconds for each frame of the video
            first frame should start with a timestamp of 0.0. The size of the sequence must be equal
            to video.shape[0] if provided. If not provided, will assume monotonically increasing
            timestamps based on framerate
    """

    audio: Optional[np.typing.NDArray] = None
    video: Optional[np.typing.NDArray] = None
    video_ts: Optional[Sequence[float]] = None


class StreamingFileType:
    name = "abstract"

    def __init__(self, path: str, mode: str = "r", **kwargs: Any) -> None:
        self.path = path
        self.mode = mode
        if self.mode not in {"r", "w"}:
            raise ValueError('Streaming file must be opened in either "r" or "w" mode.')
        self._closed = False

    def write(self, obj: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"Not sure how to write object into streaming file {self.name}.")

    def flush(self) -> None:
        pass

    def read(self, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Not sure how to index streaming file {self.name}.")

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Cannot seek streaming files.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError(f"Not sure how to iterate streaming file {self.name}.")

    @property
    def closed(self):
        return getattr(self, "_closed", True)

    @closed.setter
    def closed(self, c):
        self._closed = c
    
    def close(self) -> None:
        pass

    def __del__(self) -> None:
        if not self.closed:
            self.close()

    def __enter__(self) -> "StreamingFileType":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class StreamingMP4File(StreamingFileType):
    name = "mp4"

    def __init__(
        self,
        path: str,
        mode: str = "r",
        with_video: bool = True,
        with_audio: bool = False,
        shape: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Creates a streaming file that can be used to read or write an MP4
        without storing all frames in memory at once.

        Args:
            path: Path to the MP4 file.

            mode: Open mode. "r" or "w".

            with_video: Hint that the stream will contain video data. Default
                is True. If you plan to write video data, set this to True or
                write some video data in the first write() call.

            with_audio: Hint that the stream will contain audio data. Default
                is False. If you plan to write audio data, set this to True or
                write some audio data in the first write() call.

            video_kwargs: Args to pass to pyav as video options. Examples
                include framerate, or crf. Framerate will be passed to the
                stream, all other options go to the codec.

            audio_kwargs: Args to pass to pyav as audio options. Examples
                include sample_rate. Sample rate will be passed to the stream,
                all other options go to the codec.

            shape: If you will be writing video frames on the fly and do not
                plan to write a frame on the first write call, specify the video
                shape here.
        """
        assert mode in ["r", "w"]
        super().__init__(path, mode=mode)

        if mode == "w":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        self.output_file = open(self.path, mode + "b")

        self.container = av.open(
            self.output_file,
            mode=mode,
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        )
        self.video_stream = None
        self.audio_stream = None
        self.shape = shape[:2] if shape is not None else shape
        self.with_video = with_video
        self.with_audio = with_audio

        self.video_pts: int = 0
        self.audio_pts: int = 0

        self.audio_kwargs = dict(kwargs.get("audio_kwargs", None) or {})
        self.video_kwargs = dict(kwargs.get("video_kwargs", {}))
        self.framerate: int = self.video_kwargs.pop("framerate", DEFAULT_FRAMERATE)
        self.sample_rate: int = self.audio_kwargs.pop("sample_rate", DEFAULT_SAMPLE_RATE)
        self.vcodec: str = self.video_kwargs.pop("codec", DEFAULT_VCODEC)
        self.acodec: str = self.audio_kwargs.pop("codec", DEFAULT_ACODEC)
        self.audio_layout: str = self.audio_kwargs.pop("layout", DEFAULT_AUDIO_LAYOUT)
        self.video_kwargs.setdefault("crf", DEFAULT_CRF)

        # PyAV expects options to contain exclusively strings.
        self.video_kwargs = {k: str(v) for k, v in self.video_kwargs.items()}
        self.audio_kwargs = {k: str(v) for k, v in self.audio_kwargs.items()}

        if mode == "r":
            self.video_stream = self.container.streams.video[0]
            if hasattr(self.container.streams.video[0], "guessed_rate"):
                self.framerate = self.container.streams.video[0].guessed_rate
            else:
                self.framerate = self.container.streams.video[0].framerate
            self.video_iter = iter(self.container.decode(video=0))

    def __getstate__(self):
        raise RuntimeError("Tried to pickle streaming MP4 file.")

    def read(self, **kwargs: Any) -> np.typing.NDArray:
        if self.mode != "r" or self.closed:
            raise ValueError("Streaming file not opened in read mode.")
        return self.__next__()
    
    def write(self, obj: VideoObj, **kwargs: Any) -> None:
        """
        Args:
        - obj: Video object to write, with the following attributes:
            - video: if provided, must be a single frame np.ndarry of size: (H, W, N_CHANNELS)
            - video_ts: if provided, must be a sequence with a single float giving the relative
            timestamp for that frame in seconds. If not provided, the timestamps will be uniformly
            incremented based on the specified frame rate
            - audio: Cf. VideoObj doc
        Example:
            ```
            obj = VideoObj(
                video=np.ones((100, 100, 3), dtype=np.uint8),
                video_ts=[0.42],
                audio=np.random.rand(2, 1234),
            )
            streamer.write(obj)
            ```
        """
        if self.mode != "w" or self.closed:
            raise ValueError("Streaming file not opened in write mode.")

        if not isinstance(obj, VideoObj):
            obj = self.strictify_video_obj(obj)
        # Create streams. Both streams must be created before writing to either
        # one if we will have both audio / video data.
        if self.video_stream is None and (obj.video is not None or self.with_video):
            if obj.video is not None:
                video_frame = convert_to_rgb24(obj.video)
                h, w = video_frame.shape[:2]
                self.shape = (h, w)
            else:
                if self.shape is None:
                    raise RuntimeError(
                        "No video frame was present in the first write() call and the "
                        "shape was not given to the open() call."
                    )
                h, w = self.shape

            self.video_stream = self.container.add_stream(
                self.vcodec,
                width=w,
                height=h,
                pix_fmt="yuv420p",
                framerate=self.framerate,
                options=self.video_kwargs,
            )
            self.video_stream.codec_context.time_base = Fraction(1, self.framerate)

        if self.audio_stream is None and (obj.audio is not None or self.with_audio):
            if obj.audio is not None:
                given_layout = get_audio_layout(obj.audio)
                if given_layout != self.audio_layout:
                    print(
                        f"The requested audio layout (mono vs stereo) does not match the "
                        f"layout of the initial audio packet: {self.audio_layout} != {given_layout}"
                    )
            self.audio_stream = self.container.add_stream(
                self.acodec,
                sample_rate=self.sample_rate,
                layout=self.audio_layout,
            )
            self.audio_stream.options = self.audio_kwargs
        
        # Write data to streams.
        if obj.video is not None:
            if self.video_stream is None:
                raise RuntimeError(
                    "Tried to write a video frame to the MP4 stream before the video stream "
                    "was created. You must either open the stream with `with_video=True` or "
                    "write a video frame in the first call."
                )

            video_frame = convert_to_rgb24(obj.video)
            if tuple(video_frame.shape) != self.shape + (3,):
                raise ValueError(
                    f"Expected video frame with shape {self.shape + (3,)}, got shape {video_frame.shape}."
                )

            video_frame = av.VideoFrame.from_ndarray(video_frame, format="rgb24")
            this_pts: int = self.video_pts
            if obj.video_ts is not None:
                if len(obj.video_ts) != 1:
                    raise TypeError(
                        "obj.video_ts should be a sequence of 1 element exaclty, "
                        f"got {len(obj.video_ts)} elements"
                    )
                this_pts = round(obj.video_ts[0] / self.video_stream.codec_context.time_base)
                print(
                    f"Not writing frame at {obj.video_ts[0]} because the "
                    "presentation timestamp would be lower than the previous frame's"
                )

            if this_pts >= self.video_pts:
                video_frame.pts = this_pts
                self.video_pts = this_pts + 1  # next frame will have this pts at least

                for packet in self.video_stream.encode(video_frame):
                    self.container.mux(packet)

        if obj.audio is not None:
            if self.audio_stream is None:
                raise RuntimeError(
                    "Tried to write an audio sample to the MP4 stream before the audio stream "
                    "was created. You must either open the stream with `with_audio=True` or "
                    "write an audio sample in the first call."
                )

            audio_data = convert_to_fltp(obj.audio)
            audio_frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio_data),
                format="fltp",
                layout=get_audio_layout(audio_data),
            )
            audio_frame.sample_rate = self.sample_rate
            audio_frame.time_base = Fraction(1, self.sample_rate)
            audio_frame.pts = self.audio_pts
            self.audio_pts += audio_data.shape[1]

            for packet in self.audio_stream.encode(audio_frame):
                self.container.mux(packet)
    
    def strictify_video_obj(self, obj: Mapping[str, np.typing.NDArray]) -> VideoObj:
        if isinstance(obj, VideoObj):
            return obj
        strict_obj: VideoObj = VideoObj()
        if isinstance(obj, th.Tensor):
            strict_obj = replace(strict_obj, video=obj.data.cpu().numpy())
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                if isinstance(v, th.Tensor):
                    v = v.data.cpu().numpy()
                assert k in ["audio", "video"]
                strict_obj = replace(strict_obj, **{k: v})
        else:
            strict_obj = replace(strict_obj, video=obj)
        return strict_obj
    
    def __iter__(self) -> "StreamingMP4File":
        if self.mode != "r" or self.closed:
            raise ValueError(
                "Cannot create an iterator for a streaming video not opened in read mode."
            )

        return self

    def get_n_video_frames(self) -> int:
        return self.video_stream.frames

    def __next__(self) -> np.typing.NDArray:
        self.video_pts += 1
        return next(self.video_iter).to_ndarray(format="rgb24")

    def close(self) -> None:
        if self.closed:
            return

        if self.mode == "w":
            if self.video_stream is not None:
                for packet in self.video_stream.encode():
                    self.container.mux(packet)

            if self.audio_stream is not None:
                for packet in self.audio_stream.encode():
                    self.container.mux(packet)

        self.container.close()
        self.output_file.close()
        self.closed = True


def convert_to_rgb24(obj: np.typing.NDArray) -> np.typing.NDArray:
    if obj.ndim == 3 and obj.shape[0] in [1, 3]:
        obj = obj.transpose(1, 2, 0)
    if obj.ndim == 2:
        obj = np.dstack(3 * [obj])
    elif obj.shape[2] == 1:
        obj = np.dstack(3 * [obj[..., 0]])
    if obj.dtype != np.uint8 and obj.max() <= 1:
        obj = obj * 255
    return obj.astype(np.uint8)


def convert_to_fltp(obj: np.typing.NDArray) -> np.typing.NDArray:
    if obj.ndim == 1:
        obj = obj[None]
    _verify_audio_shape(obj)

    orig_dtype = obj.dtype
    obj = obj.astype(np.float32)
    if orig_dtype not in [np.float32, np.float64]:
        obj = obj / float(np.iinfo(orig_dtype).max)
    return obj


def _verify_audio_shape(obj: np.typing.NDArray) -> None:
    if obj.ndim > 2:
        raise ValueError(
            "Audio data should be a 2D array of samples with shape [n_channels, "
            f"n_samples]. Got shape: {obj.shape}"
        )
    elif obj.shape[0] not in {1, 2}:
        raise ValueError(
            f"Currently only mono / stereo (1 / 2 channel) audio is supported. "
            f"Got shape: {obj.shape}"
        )


def get_audio_layout(obj: np.typing.NDArray) -> Literal["mono", "stereo"]:
    if obj.ndim == 1:
        obj = obj[None]
    _verify_audio_shape(obj)

    return "mono" if obj.shape[0] == 1 else "stereo"
             