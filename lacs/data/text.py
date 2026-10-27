"""Text normalisation for Reddit-style messages.

Quoted lines (starting with '>' or '&gt;') are removed: they repeat someone
else's words and would pull a user's position towards the person they quote.
"""
from __future__ import annotations

import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS = re.compile(r"\s+")
_DELETED = {"[deleted]", "[removed]", "deleted", "removed"}


def clean_text(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("&gt;", ">")
    lines = [ln for ln in s.splitlines() if not ln.lstrip().startswith(">")]
    s = "\n".join(lines)
    s = html.unescape(s)
    s = _MD_LINK.sub(r"\1", s)
    s = _URL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s.lower() in _DELETED:
        return ""
    return s
