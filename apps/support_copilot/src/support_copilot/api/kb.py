"""Knowledge base: upload documents into the RAG index and list them."""

from __future__ import annotations

from pathlib import PureWindowsPath

from fastapi import APIRouter, Depends, Response, UploadFile, status
from rag_core.ingestion import IngestionError
from rag_core.service import DocumentLimitError, EmbeddingError
from sqlmodel import select

from support_copilot.api.deps import RagDep, SessionDep, SettingsDep, api_error
from support_copilot.api.ratelimit import rate_limit
from support_copilot.models import KnowledgeDocument
from support_copilot.schemas import DocumentRead, DocumentUploaded, ErrorDetail

router = APIRouter(prefix="/kb", tags=["knowledge base"])


def clean_filename(name: str | None) -> str:
    # PureWindowsPath splits on both / and \, so no client-side directory survives.
    return PureWindowsPath(name or "").name.strip()[:255]


@router.post(
    "/upload",
    response_model=DocumentUploaded,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Already in the knowledge base", "model": DocumentUploaded},
        409: {"model": ErrorDetail},
        413: {"model": ErrorDetail},
        422: {"model": ErrorDetail},
        429: {"model": ErrorDetail},
        503: {"model": ErrorDetail},
    },
    dependencies=[Depends(rate_limit)],
)
def upload_document(
    file: UploadFile, response: Response, session: SessionDep, rag: RagDep, settings: SettingsDep
) -> DocumentUploaded:
    filename = clean_filename(file.filename)
    if not filename:
        raise api_error(422, "missing_filename", "The uploaded file has no name.")
    # Read one byte past the limit: enough to tell "too big" without loading a huge file.
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise api_error(
            413, "file_too_large", f"The file is larger than {settings.max_upload_mb} MB."
        )

    try:
        added = rag.add_document(data, filename)
    except IngestionError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    except DocumentLimitError as exc:
        raise api_error(409, "document_limit", str(exc)) from exc
    except EmbeddingError as exc:
        raise api_error(503, "embedding_unavailable", str(exc)) from exc

    document = session.exec(
        select(KnowledgeDocument).where(KnowledgeDocument.doc_id == added.doc_id)
    ).first()
    if document is None:
        document = KnowledgeDocument(filename=added.filename, doc_id=added.doc_id)
        session.add(document)
        session.commit()
        session.refresh(document)
    else:
        response.status_code = status.HTTP_200_OK
    return DocumentUploaded(
        **DocumentRead.model_validate(document).model_dump(),
        chunks=added.chunks,
        duplicate=added.duplicate,
        skipped_pages=added.skipped_pages,
    )


@router.get("", response_model=list[DocumentRead])
def list_documents(session: SessionDep) -> list[KnowledgeDocument]:
    return list(session.exec(select(KnowledgeDocument).order_by(KnowledgeDocument.id)))
