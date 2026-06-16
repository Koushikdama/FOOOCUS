from enum import Enum


class VideoQuality(Enum):
    STANDARD = "480p"
    HIGH = "720p"

    @classmethod
    def list(cls) -> list:
        return [member.value for member in cls]


VIDEO_MODELS = {
    VideoQuality.STANDARD: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    VideoQuality.HIGH: "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
}

VIDEO_RESOLUTIONS = {
    VideoQuality.STANDARD: (832, 480),
    VideoQuality.HIGH: (1280, 720),
}

DEFAULT_VIDEO_QUALITY = VideoQuality.STANDARD
DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_VIDEO_GUIDANCE_SCALE = 5.0
DEFAULT_VIDEO_STEPS = 30
DEFAULT_VIDEO_NEGATIVE_PROMPT = "low quality, blurry, distorted, static, frozen, jittery"

MIN_DURATION_SEC = 2
MAX_DURATION_SEC = 10
MIN_FPS = 8
MAX_FPS = 24

FPS_CHOICES = [8, 12, 16, 24]

VRAM_THRESHOLD_FOR_OFFLOAD_GB = 20
