import os

import pytest

from peakwhale_helm.documents import DocumentError, DocumentStore, chunk_text, terms


def test_chunk_text_keeps_paragraphs_and_overlaps():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("Paragraph 0.")
    # neighbours share some text
    assert any(chunks[i][-20:].strip() in chunks[i + 1] for i in range(len(chunks) - 1))


def test_chunk_text_splits_very_long_paragraphs():
    chunks = chunk_text("x" * 2000, size=500, overlap=0)
    assert [len(c) for c in chunks] == [500, 500, 500, 500]


def test_terms_stem_consistently():
    assert terms("declares")[0] == terms("declared")[0] == terms("declare")[0]
    assert terms("transactions") == terms("transaction")
    assert terms("the of and") == []


def test_demo_documents_are_indexed(demo_dir):
    store = DocumentStore(demo_dir / "vault")
    docs = {d.source: d for d in store.list_documents()}
    assert set(docs) == {"2631d00b_loan_app.pdf", "f81dbaa1_loan_app.pdf", "credit_policy.md"}
    assert docs["2631d00b_loan_app.pdf"].kind == "pdf" and docs["2631d00b_loan_app.pdf"].chunks == 1


def test_search_with_source_filter(demo_dir):
    store = DocumentStore(demo_dir / "vault")
    hits = store.search("stated income source", source_filter="f81dbaa1")
    assert [h.chunk.source for h in hits] == ["f81dbaa1_loan_app.pdf"]
    assert "W2 Employment" in hits[0].chunk.text
    assert hits[0].chunk.page == 1 and hits[0].chunk.index == 0
    assert hits[0].chunk.label == "f81dbaa1_loan_app.pdf, chunk 0 (page 1)"


def test_search_ranks_the_right_file(demo_dir):
    store = DocumentStore(demo_dir / "vault")
    hits = store.search("manual review for gambling transactions")
    assert hits[0].chunk.source == "credit_policy.md"


def test_no_match_returns_nothing_but_filter_returns_file_start(demo_dir):
    store = DocumentStore(demo_dir / "vault")
    assert store.search("zebra giraffe") == []
    hits = store.search("zebra giraffe", source_filter="2631d00b")
    assert [h.chunk.source for h in hits] == ["2631d00b_loan_app.pdf"]


def test_markdown_txt_subfolders_and_skipped_files(tmp_path):
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / "acme.md").write_text("# Acme\n\nPayment terms are 45 days.")
    (tmp_path / "notes.txt").write_text("The warehouse in Lyon closes in August.")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".secret.txt").write_text("hidden payment terms")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    store = DocumentStore(tmp_path)
    docs = {d.source: d for d in store.list_documents()}
    assert set(docs) == {"contracts/acme.md", "notes.txt", "broken.pdf"}
    assert docs["broken.pdf"].chunks == 0 and docs["broken.pdf"].error
    hits = store.search("payment terms")
    assert [h.chunk.source for h in hits] == ["contracts/acme.md"]
    assert store.search("Lyon warehouse")[0].chunk.source == "notes.txt"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_symlink_pointing_outside_is_ignored(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret salary list")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.txt").write_text("public note")
    os.symlink(outside, docs / "link.txt")
    store = DocumentStore(docs)
    assert [d.source for d in store.list_documents()] == ["ok.txt"]


def test_index_is_rebuilt_when_files_change(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    store = DocumentStore(tmp_path)
    assert store.search("beta") == []
    (tmp_path / "b.txt").write_text("beta")
    assert store.search("beta")[0].chunk.source == "b.txt"


def test_missing_folder_has_a_clear_message(tmp_path):
    with pytest.raises(DocumentError, match="helm-demo"):
        DocumentStore(tmp_path / "nope").search("x")
    with pytest.raises(DocumentError, match="No documents folder"):
        DocumentStore(None).search("x")
    with pytest.raises(DocumentError, match="empty"):
        DocumentStore(tmp_path).search("  ")


class FakeEmbedder:
    """Counts a few topic words. Enough to test the embeddings path without a model."""

    TOPICS = ["income", "gambling", "trading", "weather"]

    def _embed(self, text):
        words = text.lower()
        return [float(words.count(t)) + 0.01 for t in self.TOPICS]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_embeddings_retriever_with_injected_model(demo_dir):
    store = DocumentStore(demo_dir / "vault", retriever="embeddings", embedder=FakeEmbedder())
    hits = store.search("gambling", top_k=1)
    assert store.retriever_note == "embeddings"
    assert hits[0].chunk.source == "credit_policy.md"


def test_embeddings_fall_back_to_bm25_when_model_is_missing(demo_dir, monkeypatch):
    import peakwhale_helm.documents as documents

    def broken(_name):
        raise DocumentError("The embeddings retriever needs the ml extra: uv sync --extra ml")

    monkeypatch.setattr(documents, "SentenceTransformerEmbedder", broken)
    store = DocumentStore(demo_dir / "vault", retriever="embeddings")
    assert store.search("income source", source_filter="2631d00b")
    assert store.retriever_note.startswith("bm25 (embeddings unavailable")
