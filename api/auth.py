"""JWT Bearer authentication compatible with the Node backend.

The Node controller signs ``jwt.sign({ id }, process.env.JWT_SECRET, ...)`` with
HS256 by default, so ``JWT_USER_CLAIM=id`` and ``JWT_ALGORITHM=HS256`` are the
local defaults.  Both are configurable in ``core.config.Settings``.
"""

from __future__ import annotations

from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import get_settings
from core.logging import get_logger
from memory.store import get_conversation

logger = get_logger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_UNAUTHORIZED_HEADERS,
    )


def decode_token(token: str) -> str:
    """Decode a Node-issued JWT and return the configured user-id claim."""
    settings = get_settings()
    if not settings.jwt_secret:
        logger.error("jwt_secret_missing_while_auth_enabled")
        raise _unauthorized("Authentication is not configured")

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": [settings.jwt_user_claim]},
        )
    except jwt.PyJWTError as error:
        logger.info("jwt_rejected", extra={"error_type": type(error).__name__})
        raise _unauthorized("Invalid authentication token") from error

    user_id = payload.get(settings.jwt_user_claim)
    if user_id is None or not str(user_id):
        raise _unauthorized("Invalid authentication token")
    return str(user_id)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    """FastAPI dependency: return user id, or ``None`` when auth is disabled."""
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    if credentials is None:
        raise _unauthorized("Missing authentication token")
    return decode_token(credentials.credentials)


def require_conversation_owner(
    conversation_id: str, user_id: str | None, *, allow_missing: bool = False
) -> None:
    """Verify ownership of an existing conversation.

    ``allow_missing`` is used by ``POST /v1/chat`` so a client may supply a new
    UUID that the pipeline will create; read/delete routes keep strict 404s.
    """
    settings = get_settings()
    conversation = get_conversation(conversation_id)
    if conversation is None:
        if allow_missing:
            return
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if settings.auth_enabled and conversation.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your conversation")


__all__ = [
    "bearer_scheme",
    "decode_token",
    "get_current_user",
    "require_conversation_owner",
]
