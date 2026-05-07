# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the multi-modal ingestion image pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from src.ingest.contracts.converters import ImageDescriptionPayload
from src.ingest.infrastructure.converters._pdf import (
    PdfImageProcessor,  # pyright: ignore[reportPrivateUsage]
)
from src.ingest.infrastructure.image_cache import (
    ExtractedImageStore,
    ImageDescriptionCache,
    compute_image_hash,
)
from src.ingest.infrastructure.image_description import (
    ImageDescriptionService,
    ImageFilterConfig,
)
from src.ingest.infrastructure.vision import VisionDescriber


def _png_bytes(*, width: int, height: int, color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    """Build a deterministic PNG in-memory at the given size."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeDescriber:
    """Test double for :class:`VisionDescriber` recording calls."""

    def __init__(self, response: str = "x" * 80) -> None:
        self.response = response
        self.calls: list[tuple[bytes, str | None]] = []

    def describe(self, image_bytes: bytes, *, language_hint: str | None = None) -> str:
        self.calls.append((image_bytes, language_hint))
        return self.response


def _assert_protocol_compatible() -> None:
    """Compile-time-only sanity check that the fake satisfies the protocol."""
    _: VisionDescriber = _FakeDescriber()


_assert_protocol_compatible()


class TestComputeImageHash:
    def test_is_deterministic(self) -> None:
        data = b"hello world"
        assert compute_image_hash(data) == compute_image_hash(data)

    def test_differs_for_different_inputs(self) -> None:
        assert compute_image_hash(b"a") != compute_image_hash(b"b")


class TestImageDescriptionCache:
    def test_get_returns_none_on_miss(self, tmp_path: Path) -> None:
        cache = ImageDescriptionCache(tmp_path)
        assert cache.get("deadbeef") is None

    def test_put_then_get_roundtrips(self, tmp_path: Path) -> None:
        cache = ImageDescriptionCache(tmp_path)
        cache.put("h1", "described content")
        assert cache.get("h1") == "described content"


class TestExtractedImageStore:
    def test_store_writes_to_deterministic_path(self, tmp_path: Path) -> None:
        store = ExtractedImageStore(tmp_path)
        path = store.store(
            pdf_hash="abc123",
            page=2,
            image_index=5,
            image_bytes=b"PNGDATA",
            suffix=".png",
        )
        assert path == tmp_path / "abc123" / "page0002_img005.png"
        assert path.read_bytes() == b"PNGDATA"

    def test_store_is_idempotent(self, tmp_path: Path) -> None:
        store = ExtractedImageStore(tmp_path)
        first = store.store(pdf_hash="abc", page=1, image_index=0, image_bytes=b"X", suffix=".png")
        second = store.store(pdf_hash="abc", page=1, image_index=0, image_bytes=b"X", suffix=".png")
        assert first == second


class TestImageDescriptionService:
    def _make(
        self,
        tmp_path: Path,
        describer: _FakeDescriber,
        *,
        min_dimension: int = 64,
        min_description_length: int = 40,
    ) -> ImageDescriptionService:
        return ImageDescriptionService(
            describer=describer,
            cache=ImageDescriptionCache(tmp_path),
            filter_config=ImageFilterConfig(
                min_dimension=min_dimension,
                min_description_length=min_description_length,
            ),
        )

    def test_describes_and_caches_on_miss(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="x" * 80)
        svc = self._make(tmp_path, describer)
        img = _png_bytes(width=128, height=128)

        result = svc.describe(img, language_hint="de")
        assert result is not None
        assert result.description == "x" * 80
        assert result.image_hash == compute_image_hash(img)
        assert svc.vision_call_count == 1

    def test_cache_hit_skips_vision_call(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="x" * 80)
        svc = self._make(tmp_path, describer)
        img = _png_bytes(width=128, height=128)

        svc.describe(img)
        svc.describe(img)
        svc.describe(img)

        assert svc.vision_call_count == 1

    def test_cache_hit_returns_cached_description_unfiltered(self, tmp_path: Path) -> None:
        # Pre-seed the cache with a short description; later runs with a
        # higher floor must still return it (cache contents are authoritative).
        cache = ImageDescriptionCache(tmp_path)
        img = _png_bytes(width=128, height=128)
        h = compute_image_hash(img)
        cache.put(h, "short")

        describer = _FakeDescriber()
        svc = ImageDescriptionService(
            describer=describer,
            cache=cache,
            filter_config=ImageFilterConfig(min_dimension=64, min_description_length=999),
        )

        result = svc.describe(img)

        assert result is not None
        assert result.description == "short"
        assert describer.calls == []

    def test_skips_too_small_image(self, tmp_path: Path) -> None:
        describer = _FakeDescriber()
        svc = self._make(tmp_path, describer, min_dimension=64)
        img = _png_bytes(width=32, height=32)

        assert svc.describe(img) is None
        assert describer.calls == []

    def test_skips_short_description(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="hi")
        svc = self._make(tmp_path, describer, min_description_length=40)
        img = _png_bytes(width=128, height=128)

        assert svc.describe(img) is None
        # Vision was called, but description was filtered and not cached.
        assert svc.vision_call_count == 1

    def test_swallows_vision_errors(self, tmp_path: Path) -> None:
        class _BoomDescriber:
            def describe(self, image_bytes: bytes, *, language_hint: str | None = None) -> str:
                raise RuntimeError("vision down")

        svc = ImageDescriptionService(
            describer=_BoomDescriber(),
            cache=ImageDescriptionCache(tmp_path),
            filter_config=ImageFilterConfig(min_dimension=64, min_description_length=40),
        )
        img = _png_bytes(width=128, height=128)

        assert svc.describe(img) is None

    def test_skips_unparseable_image(self, tmp_path: Path) -> None:
        describer = _FakeDescriber()
        svc = self._make(tmp_path, describer)

        assert svc.describe(b"not an image") is None
        assert describer.calls == []


class TestMakeImageDescriptionPayload:
    def test_payload_shape(self, tmp_path: Path) -> None:
        payload = ImageDescriptionPayload(
            description="A pie chart.",
            source="corpus/example.pdf",
            image_path=tmp_path / "abc" / "page0001_img000.png",
            image_hash="hashvalue",
            base_meta={"language": "de", "title": "Example"},
            extra_meta={"page": "1", "image_index": "0"},
        )

        assert payload.description == "A pie chart."
        assert payload.source == "corpus/example.pdf"
        assert payload.image_hash == "hashvalue"
        assert str(payload.image_path).endswith("page0001_img000.png")
        assert payload.base_meta["language"] == "de"
        assert payload.base_meta["title"] == "Example"
        assert payload.extra_meta is not None
        assert payload.extra_meta["page"] == "1"
        assert payload.extra_meta["image_index"] == "0"


class TestPdfImageProcessor:
    def _make(self, tmp_path: Path, describer: _FakeDescriber) -> PdfImageProcessor:
        return PdfImageProcessor(
            service=ImageDescriptionService(
                describer=describer,
                cache=ImageDescriptionCache(tmp_path / "cache"),
                filter_config=ImageFilterConfig(min_dimension=64, min_description_length=40),
            ),
            store=ExtractedImageStore(tmp_path / "extracted"),
        )

    def test_emits_one_doc_per_unique_image(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="x" * 80)
        proc = self._make(tmp_path, describer)
        img_a = _png_bytes(width=128, height=128, color=(10, 20, 30))
        img_b = _png_bytes(width=128, height=128, color=(40, 50, 60))
        seen: set[str] = set()

        payloads = proc.emit_for_page(
            pdf_source="corpus/x.pdf",
            pdf_hash="pdfhash",
            page_number=1,
            page_images=[(0, ".png", img_a), (1, ".png", img_b)],
            base_meta={"language": "de"},
            seen_hashes=seen,
            language_hint="de",
        )

        assert len(payloads) == 2
        assert all(isinstance(payload, ImageDescriptionPayload) for payload in payloads)
        assert seen == {compute_image_hash(img_a), compute_image_hash(img_b)}
        # extracted bytes were stored per image
        assert (tmp_path / "extracted" / "pdfhash" / "page0001_img000.png").exists()
        assert (tmp_path / "extracted" / "pdfhash" / "page0001_img001.png").exists()

    def test_dedups_repeated_logo_across_pages(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="x" * 80)
        proc = self._make(tmp_path, describer)
        logo = _png_bytes(width=128, height=128)
        seen: set[str] = set()

        page1 = proc.emit_for_page(
            pdf_source="corpus/x.pdf",
            pdf_hash="pdfhash",
            page_number=1,
            page_images=[(0, ".png", logo)],
            base_meta={},
            seen_hashes=seen,
            language_hint=None,
        )
        page2 = proc.emit_for_page(
            pdf_source="corpus/x.pdf",
            pdf_hash="pdfhash",
            page_number=2,
            page_images=[(0, ".png", logo)],
            base_meta={},
            seen_hashes=seen,
            language_hint=None,
        )

        assert len(page1) == 1
        assert len(page2) == 0
        assert describer.calls and len(describer.calls) == 1

    def test_skips_filtered_images(self, tmp_path: Path) -> None:
        describer = _FakeDescriber(response="x" * 80)
        proc = self._make(tmp_path, describer)
        too_small = _png_bytes(width=16, height=16)
        ok = _png_bytes(width=128, height=128)

        payloads = proc.emit_for_page(
            pdf_source="corpus/x.pdf",
            pdf_hash="pdfhash",
            page_number=1,
            page_images=[(0, ".png", too_small), (1, ".png", ok)],
            base_meta={},
            seen_hashes=set(),
            language_hint=None,
        )

        assert len(payloads) == 1
        assert payloads[0].extra_meta is not None
        assert payloads[0].extra_meta["image_index"] == "1"


@pytest.mark.parametrize("language_hint", ["de", "en", None])
def test_describer_receives_language_hint(tmp_path: Path, language_hint: str | None) -> None:
    describer = _FakeDescriber(response="x" * 80)
    svc = ImageDescriptionService(
        describer=describer,
        cache=ImageDescriptionCache(tmp_path),
        filter_config=ImageFilterConfig(min_dimension=64, min_description_length=40),
    )
    img = _png_bytes(width=128, height=128)

    svc.describe(img, language_hint=language_hint)

    assert describer.calls[0][1] == language_hint
