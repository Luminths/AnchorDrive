"""Role compatibility for English goals and archived Chinese goals."""
import re


def is_helper_role(role):
    value = str(role).casefold()
    return ("\u8f85\u52a9" in value or
            bool(re.search(r"\b(helper|auxiliary|assist|support)(?: vehicle| car)?\b", value)))
