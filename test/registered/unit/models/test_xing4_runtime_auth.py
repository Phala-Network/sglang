"""Final runtime's explicit env-auth path, without credentials in process argv."""

import json
import secrets

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware, auth_level


def test_env_token_resolution_redaction_and_http_auth(monkeypatch):
    token = secrets.token_hex(24)
    monkeypatch.setenv("PIG_AUTH_FROM_TOKEN", "1")
    monkeypatch.setenv("TOKEN", token)
    args = ServerArgs(model_path="dummy")
    args.resolve_once()
    view = resolving_view(args)
    assert view.api_key is not None and secrets.compare_digest(view.api_key, token)
    assert view.admin_api_key is not None and secrets.compare_digest(view.admin_api_key, token)
    assert token not in json.dumps(args.diagnostic_dict())
    assert args.api_key is None and args.admin_api_key is None

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list"}

    @app.get("/admin")
    @auth_level(AuthLevel.ADMIN_FORCE)
    def admin():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    add_api_key_middleware(app, api_key=view.api_key, admin_api_key=view.admin_api_key)
    with TestClient(app) as client:
        for path in ("/v1/models", "/admin"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
            assert client.get(path, headers={"Authorization": "Bearer " + token}).status_code == 200
        # Backend probes are exempt; public ingress must authenticate them.
        assert client.get("/health").status_code == 200
