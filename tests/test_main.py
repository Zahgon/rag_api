import os
import jwt
import datetime
import pytest
from langchain_core.documents import Document

from main import app
from app.routes import document_routes

client = app.test_client()


def _auth_header():
    jwt_secret = os.environ.setdefault("JWT_SECRET", "testsecret")
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, jwt_secret, algorithm='HS256')}"}


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes

    # Clear the LRU cache and patch the cached function to return dummy embeddings
    document_routes.get_cached_query_embedding.cache_clear()

    def dummy_get_cached_query_embedding(query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", dummy_get_cached_query_embedding
    )

    # Override get_all_ids as an async function - patch at CLASS level to bypass run_in_executor
    async def dummy_get_all_ids(self, owners=None, executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(self, ids, owners=None, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(self, ids, owners=None, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function with a dummy that doesn't call OpenAI
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def dummy_similarity_search_with_score_by_vector(self, embedding, k, filter):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1") if filter else "testid1",
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        AsyncPgVector,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(self, docs, ids):
        return ids

    async def dummy_aadd_documents(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "add_documents", dummy_add_documents)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(self, ids=None, collection_only=False, executor=None):
        return None

    async def dummy_delete_scoped(self, ids=None, owners=None, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)
    monkeypatch.setattr(AsyncPgVector, "delete_scoped", dummy_delete_scoped)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", query_string={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.delete("/documents", json=["testid1"], headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Monkeypatch RAG_UPLOAD_DIR so the file is within the allowed directory.
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))

    # Create a temporary file inside the patched upload dir.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": "test.txt",
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    json_data = response.json
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={
                "file_id": "testid1",
                "entity_id": "testuser",
                "file": (f, "test_embed.txt", "text/plain"),
            },
            content_type="multipart/form-data",
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    json_data = response.json
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    content = response.get_data(as_text=True)
    assert "testid1" in content or "Test content" in content


def test_load_document_context_restores_parallel_chunk_order(auth_headers, monkeypatch):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def out_of_order_documents(self, ids, owners=None, executor=None):
        return [
            Document(
                page_content="third",
                metadata={
                    "_rag_chunk_index": 2,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
            Document(
                page_content="first",
                metadata={
                    "_rag_chunk_index": 0,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
            Document(
                page_content="second",
                metadata={
                    "_rag_chunk_index": 1,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
        ]

    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", out_of_order_documents)

    response = client.get("/documents/testid1/context", headers=auth_headers)

    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    assert response.json == "firstsecondthird"


def test_load_document_context_groups_repeated_ingestion_attempts(
    auth_headers, monkeypatch
):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    def marked(content, chunk_index, attempt_id, started_at_ns):
        return Document(
            page_content=content,
            metadata={
                "_rag_chunk_index": chunk_index,
                "_rag_ingestion_attempt_id": attempt_id,
                "_rag_ingestion_attempt_started_at_ns": started_at_ns,
            },
        )

    async def interleaved_attempts(self, ids, owners=None, executor=None):
        return [
            marked("old-second", 1, "old", 100),
            marked("new-second", 1, "new", 200),
            marked("old-first", 0, "old", 100),
            marked("new-first", 0, "new", 200),
        ]

    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", interleaved_attempts)

    response = client.get("/documents/testid1/context", headers=auth_headers)

    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    assert response.json == "old-firstold-secondnew-firstnew-second"


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={
                "file_id": "testid1",
                "entity_id": "testuser",
                "uploaded_file": (f, "upload_test.txt", "text/plain"),
            },
            content_type="multipart/form-data",
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    json_data = response.json
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    json_data = response.json
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={
                "file_id": "test_text_123",
                "entity_id": "testuser",
                "file": (f, "test_text_extraction.txt", "text/plain"),
            },
            content_type="multipart/form-data",
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.get_data(as_text=True)}"
    json_data = response.json

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types


# ---------------------------------------------------------------------------
# Wire contract of the previous framework: validation envelopes, body parsing,
# routing and middleware order, compared byte-for-byte against the original.
# ---------------------------------------------------------------------------


def test_query_malformed_json_reports_json_invalid(auth_headers):
    response = client.post(
        "/query", data="{", content_type="application/json", headers=auth_headers
    )
    assert response.status_code == 422
    assert response.json == {
        "detail": [
            {
                "type": "json_invalid",
                "loc": ["body", 1],
                "msg": "JSON decode error",
                "input": {},
                "ctx": {"error": "Expecting property name enclosed in double quotes"},
            }
        ],
        "message": "Request validation failed",
    }


def test_query_body_that_is_not_an_object(auth_headers):
    response = client.post(
        "/query", data="[]", content_type="application/json", headers=auth_headers
    )
    assert response.status_code == 422
    assert response.json["detail"] == [
        {
            "type": "model_attributes_type",
            "loc": ["body"],
            "msg": "Input should be a valid dictionary or object to extract fields from",
            "input": [],
        }
    ]


def test_query_body_without_content_type_is_read_as_json(auth_headers):
    response = client.post(
        "/query",
        data=b'{"query": "Test query", "file_id": "testid1"}',
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json[0][0]["page_content"] == "Queried content"


def test_query_body_not_declared_as_json_is_not_parsed(auth_headers):
    response = client.post(
        "/query",
        data=b'{"query": "Test query", "file_id": "testid1"}',
        content_type="text/plain",
        headers=auth_headers,
    )
    # the body is never read as JSON, and the raw body in the validation error is
    # what it always was: not renderable, so the request ends as a plain-text 500
    assert response.status_code == 500
    assert response.content_type.startswith("text/plain")


def test_embed_reports_every_missing_multipart_field(auth_headers):
    response = client.post("/embed", headers=auth_headers)
    assert response.status_code == 422
    assert response.json["detail"] == [
        {"type": "missing", "loc": ["body", "file_id"], "msg": "Field required", "input": None},
        {"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None},
    ]


def test_trailing_slash_redirects_to_the_route(auth_headers):
    response = client.get("/ids/?entity_id=x", headers=auth_headers)
    assert response.status_code == 307
    assert response.headers["Location"] == "http://localhost/ids?entity_id=x"


def test_unknown_route_is_json(auth_headers):
    response = client.get("/nope", headers=auth_headers)
    assert response.status_code == 404
    assert response.json == {"detail": "Not Found"}


def test_rejected_request_carries_no_cors_headers():
    os.environ["JWT_SECRET"] = "testsecret"
    response = client.get("/ids", headers={"Origin": "http://example.com"})
    assert response.status_code == 401
    assert "Access-Control-Allow-Origin" not in response.headers

    preflight = client.options(
        "/ids",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 401
    assert "Access-Control-Allow-Origin" not in preflight.headers


def test_authorised_request_carries_cors_headers(auth_headers):
    response = client.get("/ids", headers={**auth_headers, "Origin": "http://example.com"})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


def test_request_with_cookies_is_answered_with_its_own_origin(auth_headers):
    client.set_cookie("session", "x", domain="localhost")
    try:
        response = client.get(
            "/ids", headers={**auth_headers, "Origin": "http://example.com"}
        )
    finally:
        client.delete_cookie("session", domain="localhost")
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://example.com"
    assert "Origin" in response.headers["Vary"]


def test_cors_preflight_is_answered_directly(auth_headers):
    response = client.options(
        "/ids",
        headers={
            **auth_headers,
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "OK"
    assert response.headers["Access-Control-Allow-Origin"] == "http://example.com"
    assert response.headers["Access-Control-Allow-Methods"] == (
        "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"
    )
    assert response.headers["Access-Control-Allow-Headers"] == "authorization"
    assert response.headers["Access-Control-Max-Age"] == "600"


def test_openapi_schema_is_served_without_authentication():
    os.environ["JWT_SECRET"] = "testsecret"
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.content_type == "application/json"
    schema = response.json
    assert schema["openapi"] == "3.1.0"
    assert schema["info"] == {"title": "FastAPI", "version": "0.1.0"}
    assert "/query" in schema["paths"]


def test_docs_pages_are_served():
    os.environ["JWT_SECRET"] = "testsecret"
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert docs.content_type.startswith("text/html")
    assert "swagger-ui" in docs.get_data(as_text=True)

    # /redoc was never exempt from authentication
    assert client.get("/redoc").status_code == 401
    assert client.get("/redoc", headers=_auth_header()).status_code == 200


def test_health_reports_a_failing_check_as_before(monkeypatch):
    async def unhealthy():
        return False

    monkeypatch.setattr(document_routes, "is_health_ok", unhealthy)
    response = client.get("/health")
    # the (body, status) pair is sent as a two-element array with status 200
    assert response.status_code == 200
    assert response.json == [{"status": "DOWN"}, 503]
