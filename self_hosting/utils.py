from pathlib import Path
from typing import Optional


__all__ = ["TokenHandler"]


class TokenHandler:

    def __init__(self, workdir: Path):
        self._token_filename = workdir / ".api_token"
        self._token: Optional[str] = None
        if self._token_filename.exists():
            self._token = self._token_filename.read_text()

    def set(self, token: str):
        if self._token is not None:
            raise RuntimeError("token is already setted")
        self._token_filename.write_text(token)
        self._token = self._token_filename.read_text()

    def get(self) -> Optional[str]:
        return self._token
