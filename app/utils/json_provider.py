from typing import Any

from flask.json.provider import DefaultJSONProvider


class JSONProvider(DefaultJSONProvider):
    """JSON written the way this API has always written it: keys in insertion
    order, non-ASCII characters kept as they are, no separators padding and no
    trailing newline (and never indented, whatever DEBUG_RAG_API says)."""

    sort_keys = False
    ensure_ascii = False

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("separators", (",", ":"))
        return super().dumps(obj, **kwargs)

    def response(self, *args: Any, **kwargs: Any) -> Any:
        obj = self._prepare_response_obj(args, kwargs)
        return self._app.response_class(self.dumps(obj), mimetype="application/json")
