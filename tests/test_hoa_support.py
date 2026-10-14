from __future__ import annotations

from pathlib import Path

from hoa_support.bm25_index import BM25Index
from hoa_support.domain import Document
from hoa_support.service import ERROR_MESSAGE, HoaSupportService, reciprocal_rank_fusion
from hoa_support.splitter import SemanticDocumentSplitter


def _service(fakes, tmp_path, ocr_enabled=False):
    return HoaSupportService(
        splitter=SemanticDocumentSplitter(fakes["embedding"], threshold=0.2),
        vector_store=fakes["vector_store"],
        keyword_index=BM25Index(tmp_path / "bm25.sqlite3"),
        chat_model=fakes["chat"],
        checksum_store=fakes["checksums"],
        ocr_enabled=ocr_enabled,
    )


def _make_pdf(path: Path, with_image: bool = False) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    lines = [
        "Maple Grove HOA Rules",  # header line (dropped)
        "Commercial vehicles may not be parked overnight in the community.",
        "Residents may keep up to two pets per household.",
        "Quiet hours are from 10 PM to 7 AM every day.",
        "Page 1",  # footer line (dropped)
    ]
    y = 740
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    if with_image:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (1400, 200), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 60)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 60), "Fines are 50 dollars per violation", fill="black", font=font)
        img_path = path.with_suffix(".png")
        img.save(img_path)
        c.drawImage(str(img_path), 72, 400, width=450, height=64)
    c.save()
    return path


def test_rrf_rewards_documents_found_by_both_retrievers():
    a, b, c = Document("a", id="a"), Document("b", id="b"), Document("c", id="c")
    fused = reciprocal_rank_fusion([[a, b], [c, b]], limit=10, k=60)
    assert [d.id for d in fused][0] == "b"
    assert {d.id for d in fused} == {"a", "b", "c"}


def test_splitter_breaks_on_topic_change(fakes):
    splitter = SemanticDocumentSplitter(fakes["embedding"], threshold=0.5)
    chunks = splitter.split(Document("Pets must be leashed. Pets must be licensed. Parking is limited to two cars."))
    assert chunks == ["Pets must be leashed. Pets must be licensed.", "Parking is limited to two cars."]


def test_bm25_search(tmp_path):
    index = BM25Index(tmp_path / "idx.sqlite3")
    index.add([Document("Dogs must be on a leash", {"source": "x.pdf"}), Document("Pool closes at 9 PM")])
    results = index.search("leash rules for dogs?", 5)
    assert results[0].text == "Dogs must be on a leash"
    assert results[0].metadata == {"source": "x.pdf"}
    assert index.search("!!!", 5) == []


def test_ingest_and_search_end_to_end(fakes, tmp_path):
    service = _service(fakes, tmp_path)
    pdf = _make_pdf(tmp_path / "rules.pdf")

    assert service.add_document(str(pdf)) == "Document added to vector store successfully."
    assert service.add_document(str(pdf)) == "Document has already been processed."

    stored_texts = " ".join(d.text for d, _ in fakes["vector_store"].items)
    assert "Maple Grove HOA Rules" not in stored_texts  # header removed
    assert "Commercial vehicles" in stored_texts

    answer = service.get_hoa_support_info("Can I park a commercial vehicle overnight?")
    assert answer.startswith("ANSWER")
    system_prompt, _ = fakes["chat"].calls[-1]
    assert "Commercial vehicles may not be parked overnight" in system_prompt


def test_add_document_rejects_bad_paths(fakes, tmp_path):
    service = _service(fakes, tmp_path)
    assert "invalid" in service.add_document(str(tmp_path / "missing.pdf"))
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    assert service.add_document(str(txt)) == "Only PDF documents are supported."


def test_ocr_fallback_indexes_image_text(fakes, tmp_path):
    service = _service(fakes, tmp_path, ocr_enabled=True)
    pdf = _make_pdf(tmp_path / "scanned.pdf", with_image=True)
    service.add_document(str(pdf))
    ocr_docs = [d for d, _ in fakes["vector_store"].items if d.metadata.get("type") == "pdf-ocr"]
    assert ocr_docs, "expected OCR chunks"
    assert "violation" in " ".join(d.text for d in ocr_docs).lower()


def test_load_folder_skips_already_processed(fakes, tmp_path):
    service = _service(fakes, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    _make_pdf(docs / "rules.pdf")
    service.load_folder(str(docs))
    count = len(fakes["vector_store"].items)
    service.load_folder(str(docs))
    assert len(fakes["vector_store"].items) == count


def test_search_returns_friendly_error(fakes, tmp_path):
    service = _service(fakes, tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("chroma down")

    service.vector_store.similarity_search = boom
    assert service.get_hoa_support_info("pets?") == ERROR_MESSAGE


def test_chat_ui_and_api(fakes, tmp_path):
    from starlette.testclient import TestClient

    from hoa_support.config import Settings
    from hoa_support.server import create_server

    service = _service(fakes, tmp_path)
    service.ingest_pdf(_make_pdf(tmp_path / "rules.pdf"))
    app = create_server(service, Settings()).streamable_http_app()

    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "HOA Support" in page.text

        res = client.post("/api/chat", json={"question": "How many pets can I keep?"})
        assert res.status_code == 200
        assert res.json()["answer"] == "ANSWER for: How many pets can I keep?"

        assert client.post("/api/chat", json={"question": "  "}).status_code == 400
        assert client.post("/api/chat", content=b"not json").status_code == 400
