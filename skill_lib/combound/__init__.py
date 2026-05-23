# skill_lib/compound/__init__.py

from skill_lib.combound.select_obj_by_text import SelectObjectByTextSkill
from skill_lib.combound.find_obj_by_text import FindObjectByTextSkill


COMPOUND_SKILLS = [
    SelectObjectByTextSkill,
    FindObjectByTextSkill,
]