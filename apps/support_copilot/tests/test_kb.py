from tests.conftest import upload


def test_upload_creates_document(client):
    resp = upload(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == 1
    assert body["filename"] == "delivery.txt"
    assert len(body["doc_id"]) > 8
    assert body["chunks"] >= 1
    assert body["duplicate"] is False
    assert body["skipped_pages"] == []
    assert client.app.state.rag.stats()["documents"] == 1


def test_list_documents(client):
    assert client.get("/kb").json() == []
    first = upload(client).json()
    second = upload(client, "returns.txt", "Возврат товара возможен в течение 14 дней.").json()

    listed = client.get("/kb").json()
    assert [d["filename"] for d in listed] == ["delivery.txt", "returns.txt"]
    assert [d["doc_id"] for d in listed] == [first["doc_id"], second["doc_id"]]
    assert set(listed[0]) == {"id", "filename", "doc_id", "created_at"}


def test_duplicate_upload_returns_existing_document(client):
    first = upload(client).json()
    resp = upload(client, "copy.txt")  # same bytes, different name

    assert resp.status_code == 200
    assert resp.json()["duplicate"] is True
    assert resp.json()["id"] == first["id"]
    assert len(client.get("/kb").json()) == 1


def test_client_directories_are_stripped_from_filename(client):
    resp = upload(client, "..\\..\\secret/folder\\faq.txt")
    assert resp.json()["filename"] == "faq.txt"


def test_too_large_file_is_rejected_with_413(client):
    resp = upload(client, "big.txt", b"a" * (1024 * 1024 + 1))  # max_upload_mb=1 in tests
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "file_too_large"
    assert client.get("/kb").json() == []


def test_file_at_the_limit_is_accepted(client):
    text = ("Доставка по Бишкеку бесплатная. " * 40_000).encode()[: 1024 * 1024]
    assert upload(client, "limit.txt", text.decode(errors="ignore")).status_code == 201


def test_unsupported_type_is_rejected_with_422(client):
    resp = upload(client, "photo.png", b"\x89PNG....")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unsupported_type"


def test_empty_file_is_rejected_with_422(client):
    resp = upload(client, "empty.txt", b"")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "empty_file"


def test_missing_file_is_rejected_with_422(client):
    assert client.post("/kb/upload").status_code == 422


def test_document_limit_is_409(make_client):
    client = make_client(max_documents=1)
    upload(client)
    resp = upload(client, "returns.txt", "Возврат возможен в течение 14 дней.")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "document_limit"


def test_embedding_failure_is_503(client, monkeypatch):
    def broken(texts):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(client.app.state.rag.embedder, "embed_documents", broken)
    resp = upload(client)
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "embedding_unavailable"
    assert client.get("/kb").json() == []
