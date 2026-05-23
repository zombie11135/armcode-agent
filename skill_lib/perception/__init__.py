# skill_lib/perception/__init__.py

from skill_lib.perception.capture import (
    CaptureGlobalImageSkill,
    CaptureWristRGBDSkill,
)

from skill_lib.perception.vdm import VDMSkill


PERCEPTION_SKILLS = [
    CaptureGlobalImageSkill,
    CaptureWristRGBDSkill,
    VDMSkill,
]