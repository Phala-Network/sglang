"""Copy public diagnostics without exposing credentials or launch arguments."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_REDACTED = "<redacted>"

def _sensitive_key(key):
    key = str(key).lower().replace("-", "_")
    return (
        key in {"apikey", "accesskeyid", "awsaccesskeyid", "launch_command", "authorization", "proxy_authorization", "headers", "http_headers", "token", "key", "sig", "signature", "x_amz_credential", "x_amz_signature"}
        or "password" in key or "secret" in key or "credential" in key or "signature" in key
        or key.endswith(("_api_key", "_token", "_key", "_extra_config", "_extra_config_json"))
    )

def sanitize_server_info(value):
    """Return a new recursively sanitized diagnostics value; never mutate config."""
    if isinstance(value, dict):
        return {key: _REDACTED if _sensitive_key(key) else sanitize_server_info(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_server_info(item) for item in value]
    if isinstance(value, str):
        if "bearer " in value.lower() or "authorization:" in value.lower():
            return _REDACTED
        try:
            parsed = urlsplit(value)
            if parsed.scheme and parsed.netloc:
                netloc = parsed.netloc.rsplit("@", 1)[-1]
                query = parse_qsl(parsed.query, keep_blank_values=True)
                safe_query = [(key, _REDACTED if _sensitive_key(key) else item) for key, item in query]
                fragment = parse_qsl(parsed.fragment, keep_blank_values=True)
                safe_fragment = [(key, _REDACTED if _sensitive_key(key) else item) for key, item in fragment]
                if netloc != parsed.netloc or safe_query != query or safe_fragment != fragment:
                    fragment_text = urlencode(safe_fragment) if safe_fragment != fragment else parsed.fragment
                    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(safe_query), fragment_text))
        except ValueError:
            if "://" in value:
                return _REDACTED
    return value
