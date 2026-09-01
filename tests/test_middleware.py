import os
import jwt
import pytest
from flask import Flask, g
from app.middleware import security_middleware

# Minimal app to provide the request context the middleware reads from.
app = Flask(__name__)


@pytest.fixture
def valid_jwt_header():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {"id": "testuser", "exp": 9999999999}
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def invalid_jwt_header():
    return {"Authorization": "Bearer invalidtoken"}


def test_security_middleware_valid(valid_jwt_header):
    with app.test_request_context("/protected", headers=valid_jwt_header):
        # Returning None lets the request continue to the view.
        assert security_middleware() is None
        assert g.get("user") is not None
        assert g.user["id"] == "testuser"


def test_security_middleware_invalid(invalid_jwt_header):
    with app.test_request_context("/protected", headers=invalid_jwt_header):
        response = security_middleware()
        assert response is not None
        assert app.make_response(response).status_code == 401


def test_verified_user_reaches_the_view(valid_jwt_header):
    """The WSGI layer verifies the token; the view still sees it as ``g.user``."""
    from app.middleware import AuthenticateBeforeApp, restore_user

    wrapped = Flask(__name__)
    wrapped.before_request(restore_user)

    @wrapped.get("/whoami")
    def whoami():
        return {"id": g.user["id"]}

    wrapped.wsgi_app = AuthenticateBeforeApp(wrapped)
    response = wrapped.test_client().get("/whoami", headers=valid_jwt_header)
    assert response.status_code == 200
    assert response.json == {"id": "testuser"}
