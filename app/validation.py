# app/validation.py
"""Request parsing and validation.

The route signature used to declare where each value came from and what shape it
had, and the framework enforced it. Under Flask the same checks are written out,
but they raise :class:`~app.errors.RequestValidationError`, so a malformed
request still answers 422 with the same body it always did.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from flask import request
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.errors import APIError, RequestValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

STRING_LIST = TypeAdapter(List[str])


def _missing_error(loc: Sequence[Any]) -> Dict[str, Any]:
    return {
        "type": "missing",
        "loc": list(loc),
        "msg": "Field required",
        "input": None,
    }


def _missing(loc: Sequence[Any]) -> RequestValidationError:
    return RequestValidationError([_missing_error(loc)])


def _from_pydantic(exc: ValidationError, source: str) -> RequestValidationError:
    # The errors are passed on exactly as pydantic reports them: a body that was
    # not JSON stays raw bytes in "input", as it always did.
    return RequestValidationError(
        [
            {**error, "loc": [source, *error.get("loc", ())]}
            for error in exc.errors(include_url=False)
        ]
    )


def _json_payload() -> Any:
    """Read the body the way the route signature used to have it read.

    It is decoded as JSON when the request declares JSON (``application/json``,
    ``application/*+json``) or declares no content type at all. Any other body is
    validated as it is, so it fails validation instead of being guessed at.
    """
    body = request.get_data()
    if not body:
        raise _missing(["body"])
    if request.headers.get("Content-Type") and not request.is_json:
        return body
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ["body", exc.pos],
                    "msg": "JSON decode error",
                    "input": {},
                    "ctx": {"error": exc.msg},
                }
            ]
        ) from exc
    except ValueError as exc:
        raise APIError(
            status_code=400, detail="There was an error parsing the body"
        ) from exc
    if payload is None:
        raise _missing(["body"])
    return payload


def _validate(adapter: TypeAdapter, payload: Any) -> Any:
    try:
        return adapter.validate_python(payload, from_attributes=True)
    except ValidationError as exc:
        raise _from_pydantic(exc, "body") from exc


def json_body(model: Type[ModelT]) -> ModelT:
    """Validate the JSON body against a pydantic model."""
    return _validate(TypeAdapter(model), _json_payload())


def json_body_as(adapter: TypeAdapter) -> Any:
    """Validate the JSON body against a bare type (e.g. ``List[str]``)."""
    return _validate(adapter, _json_payload())


def query_param(name: str, default: Optional[str] = None) -> Optional[str]:
    return request.args.get(name, default)


def required_query_params(*names: str) -> List[str]:
    """Read required query params, reporting every missing one at once."""
    values = [request.args.get(name) for name in names]
    missing = [
        _missing_error(["query", name])
        for name, value in zip(names, values)
        if value is None
    ]
    if missing:
        raise RequestValidationError(missing)
    return values


def required_query_param(name: str) -> str:
    return required_query_params(name)[0]


def required_query_list(name: str) -> List[str]:
    values = request.args.getlist(name)
    if not values:
        raise _missing(["query", name])
    return values


def form_field(name: str, default: Optional[str] = None) -> Optional[str]:
    return request.form.get(name, default)


def required_multipart(*parts: Tuple[str, str]) -> List[Any]:
    """Read required multipart parts, e.g. ``("form", "file_id"), ("file", "file")``.

    Every missing part is reported at once, in the order the parts are given.
    """
    values, missing = [], []
    for kind, name in parts:
        value = (request.files if kind == "file" else request.form).get(name)
        if value is None:
            missing.append(_missing_error(["body", name]))
        values.append(value)
    if missing:
        raise RequestValidationError(missing)
    return values
