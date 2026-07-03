from __future__ import annotations

import os


ROLE_ORDER = {
    "viewer": 0,
    "researcher": 1,
    "expert": 2,
    "admin": 3,
}


def role_for_ui_token(token: str | None) -> str | None:
    return _role_for_token(token, os.getenv("APP_TOKENS"), os.getenv("APP_AUTH_TOKEN"), "expert")


def role_for_api_token(token: str | None) -> str | None:
    return _role_for_token(token, os.getenv("API_TOKENS"), os.getenv("API_AUTH_TOKEN"), "admin")


def has_role(role: str | None, minimum: str) -> bool:
    if role is None:
        return False
    return ROLE_ORDER.get(role, -1) >= ROLE_ORDER.get(minimum, 0)


def _role_for_token(token: str | None, token_map: str | None, legacy_token: str | None, legacy_role: str) -> str | None:
    if token_map:
        for item in token_map.split(","):
            if not item.strip():
                continue
            secret, _, role = item.partition(":")
            if token and token == secret.strip():
                return (role or legacy_role).strip()
        if legacy_token:
            return legacy_role if token == legacy_token else None
        return None
    if legacy_token:
        return legacy_role if token == legacy_token else None
    return "admin"
