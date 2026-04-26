from __future__ import annotations

import os
import subprocess
from typing import Optional


def extract_audio_copy(ffmpeg_cmd: str, input_video: str, temp_audio: str) -> bool:
    subprocess.run(
        [
            ffmpeg_cmd,
            "-y",
            "-i",
            input_video,
            "-vn",
            "-acodec",
            "copy",
            temp_audio,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0


def open_encoder(
    ffmpeg_cmd: str,
    width: int,
    height: int,
    fps: float,
    temp_video: str,
    temp_audio: Optional[str] = None,
) -> subprocess.Popen:
    encode_cmd = [
        ffmpeg_cmd,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if temp_audio:
        encode_cmd.extend(["-i", temp_audio])

    encode_cmd.extend(
        [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-tune",
            "hq",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if temp_audio:
        encode_cmd.extend(["-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest"])
    encode_cmd.append(temp_video)

    return subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def open_decoder(ffmpeg_cmd: str, input_video: str, fps: float) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffmpeg_cmd,
            "-hwaccel",
            "cuda",
            "-i",
            input_video,
            "-fps_mode",
            "cfr",
            "-r",
            str(fps),
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def terminate_process(process: subprocess.Popen | None):
    if process is not None and process.poll() is None:
        process.kill()


def close_wait_process(process: subprocess.Popen | None):
    if process is None:
        return
    if process.stdin is not None:
        process.stdin.close()
    if process.stdout is not None:
        process.stdout.close()
    process.wait()

