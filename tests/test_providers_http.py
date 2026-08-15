"""Wire-mock tests for the HTTP-backed providers. The provider's network
client (GenAI SDK, gTTS, edge-tts WebSocket) is replaced with a fake that
records the request and returns canned responses — so we verify the exact
request shape each provider builds and that responses parse correctly,
without any network access or live API keys."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from core.domain import Transcript, TranscriptSegment
from providers.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from providers.translation.gemini_provider import (
    GeminiTranslationProvider,
    _extract_json,
)


def _transcript(n: int = 2) -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker_id=f"SPEAKER_{i % 2}",
                start_ms=i * 1000,
                end_ms=i * 1000 + 500,
                text=f"line {i}",
                language="en",
            )
            for i in range(n)
        ],
        source_language="en",
    )


def _json_response(translations: list[str]):
    return SimpleNamespace(text=json.dumps({"translations": translations}))


class _FakeModels:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    async def generate_content(self, *, model=None, contents=None, config=None):
        self.calls.append({"model": model, "contents": contents})
        result = self._responder(contents)
        if inspect.isawaitable(result):
            result = await result
        return result


class _FakeClient:
    def __init__(self, responder=None):
        self.aio = SimpleNamespace(models=_FakeModels(responder or (lambda c: _json_response([]))))


class _FakeClientError(Exception):
    def __init__(self, code=None, message="boom"):
        super().__init__(message)
        self.code = code


class _FakeServerError(Exception):
    pass


class _FakeAPIError(Exception):
    pass


def _make_provider(monkeypatch, responder, batch_size=25) -> GeminiTranslationProvider:
    import providers.translation.gemini_provider as gemini

    monkeypatch.setattr(gemini.genai, "Client", lambda **kw: _FakeClient(responder))
    monkeypatch.setattr(
        gemini,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_api_key="test-key",
            provider_timeout_seconds=60,
            translation_batch_size=batch_size,
        ),
    )
    monkeypatch.setattr(gemini, "ClientError", _FakeClientError)
    monkeypatch.setattr(gemini, "ServerError", _FakeServerError)
    monkeypatch.setattr(gemini, "APIError", _FakeAPIError)
    return GeminiTranslationProvider()


def _count_batch(contents: str) -> int:
    numbered = contents.split("\n\n", 1)[1]
    return sum(1 for ln in numbered.splitlines() if ln.strip())


# --- _extract_json -----------------------------------------------------------

def test_extract_json_plain():
    assert json.loads(_extract_json('{"translations": ["a", "b"]}')) == {
        "translations": ["a", "b"]
    }


def test_extract_json_fenced():
    text = '```json\n{"translations": ["a"]}\n```\nsome trailing prose'
    assert json.loads(_extract_json(text)) == {"translations": ["a"]}


def test_extract_json_prose_prefix_and_suffix():
    text = 'Sure, here is the result: {"translations": ["a", "b"]} let me know if you need more.'
    assert json.loads(_extract_json(text)) == {"translations": ["a", "b"]}


def test_extract_json_accepts_bare_list():
    assert json.loads(_extract_json('["a", "b"]')) == ["a", "b"]


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


# --- Gemini translation ------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_builds_prompt_and_parses_response(monkeypatch):
    calls = []

    def responder(contents):
        n = _count_batch(contents)
        calls.append(contents)
        return _json_response([f"t{i}" for i in range(n)])

    provider = _make_provider(monkeypatch, responder)

    result = await provider.translate(_transcript(3), "es")

    assert provider._client.aio.models.calls[0]["model"] == "gemini-3.5-flash"
    assert "es" in calls[0]  # target language formatted into the prompt
    assert "1. line 0" in calls[0] and "3. line 2" in calls[0]
    assert [s.translated_text for s in result.segments] == ["t0", "t1", "t2"]
    assert result.target_language == "es"
    # segment alignment (same order / speaker / timing as the source)
    assert [s.speaker_id for s in result.segments] == ["SPEAKER_0", "SPEAKER_1", "SPEAKER_0"]
    assert [s.start_ms for s in result.segments] == [0, 1000, 2000]


@pytest.mark.asyncio
async def test_gemini_batches_large_transcripts(monkeypatch):
    counter = [0]

    def responder(contents):
        n = _count_batch(contents)
        out = [f"t{counter[0] + i}" for i in range(n)]
        counter[0] += n
        return _json_response(out)

    provider = _make_provider(monkeypatch, responder, batch_size=2)

    result = await provider.translate(_transcript(5), "es")

    calls = provider._client.aio.models.calls
    assert len(calls) == 3  # 5 segments / batch of 2
    assert [s.translated_text for s in result.segments] == [f"t{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_gemini_maps_http_errors_to_typed_errors(monkeypatch):
    async def _auth(contents):
        raise _FakeClientError(code=401)

    with pytest.raises(ProviderAuthError):
        provider = _make_provider(monkeypatch, _auth)
        await provider.translate(_transcript(1), "es")

    async def _ratelimit(contents):
        raise _FakeClientError(code=429)

    with pytest.raises(ProviderRateLimitError):
        provider = _make_provider(monkeypatch, _ratelimit)
        await provider.translate(_transcript(1), "es")

    async def _down(contents):
        raise _FakeServerError()

    with pytest.raises(ProviderUnavailableError):
        provider = _make_provider(monkeypatch, _down)
        await provider.translate(_transcript(1), "es")


@pytest.mark.asyncio
async def test_gemini_rejects_invalid_json_and_count_mismatch(monkeypatch):
    async def _garbage(contents):
        return SimpleNamespace(text="not json at all")

    with pytest.raises(ProviderError, match="invalid JSON"):
        provider = _make_provider(monkeypatch, _garbage)
        await provider.translate(_transcript(1), "es")

    async def _mismatch(contents):
        return _json_response(["only one"])

    with pytest.raises(ProviderError, match="count mismatch"):
        provider = _make_provider(monkeypatch, _mismatch)
        await provider.translate(_transcript(2), "es")


# --- gTTS --------------------------------------------------------------------

class _FakeGTTS:
    instances = []

    def __init__(self, text, lang, slow):
        self.text, self.lang, self.slow = text, lang, slow
        _FakeGTTS.instances.append(self)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.text)


@pytest.mark.asyncio
async def test_gtts_synthesizes_and_uploads_per_segment(monkeypatch):
    from providers.tts.gtts_provider import GTTSProvider

    _FakeGTTS.instances = []
    monkeypatch.setattr("providers.tts.gtts_provider.gTTS", _FakeGTTS)

    from core.domain import TranslatedTranscript, TranslatedSegment

    translated = TranslatedTranscript(
        target_language="es",
        segments=[
            TranslatedSegment(
                speaker_id="SPEAKER_0", start_ms=0, end_ms=500,
                source_text="hello", translated_text="hola", target_language="es",
            ),
            TranslatedSegment(
                speaker_id="SPEAKER_1", start_ms=1000, end_ms=1500,
                source_text="world", translated_text="mundo", target_language="es",
            ),
        ],
    )

    result = await GTTSProvider().synthesize(translated)

    assert [i.text for i in _FakeGTTS.instances] == ["hola", "mundo"]
    assert all(i.lang == "es" and i.slow is False for i in _FakeGTTS.instances)
    assert len(result.segments) == 2
    assert all(s.audio_path.startswith("synthesized/") for s in result.segments)
    assert [s.speaker_id for s in result.segments] == ["SPEAKER_0", "SPEAKER_1"]


# --- edge-tts ----------------------------------------------------------------

class _FakeCommunicate:
    instances = []
    fail_voices = set()

    def __init__(self, text, voice):
        self.text, self.voice = text, voice
        self.fail = voice in _FakeCommunicate.fail_voices
        _FakeCommunicate.instances.append(self)

    async def save(self, path):
        if self.fail:
            raise RuntimeError(f"voice {self.voice} unavailable")
        with open(path, "w") as f:
            f.write(f"{self.voice}:{self.text}")


@pytest.mark.asyncio
async def test_edge_tts_maps_speaker_to_stable_voice(monkeypatch):
    from providers.tts.edge_tts_provider import EdgeTTSProvider

    _FakeCommunicate.instances = []
    _FakeCommunicate.fail_voices = set()
    monkeypatch.setattr("edge_tts.Communicate", _FakeCommunicate)

    from core.domain import TranslatedTranscript, TranslatedSegment

    translated = TranslatedTranscript(
        target_language="es",
        segments=[
            TranslatedSegment(
                speaker_id="SPEAKER_00", start_ms=0, end_ms=500,
                source_text="hi", translated_text="hola", target_language="es",
            ),
            TranslatedSegment(
                speaker_id="SPEAKER_01", start_ms=1000, end_ms=1500,
                source_text="bye", translated_text="adios", target_language="es",
            ),
        ],
    )

    result = await EdgeTTSProvider().synthesize(translated)

    # SPEAKER_00 -> voices[0], SPEAKER_01 -> voices[1] for es
    assert _FakeCommunicate.instances[0].voice == "es-ES-AlvaroNeural"
    assert _FakeCommunicate.instances[1].voice == "es-ES-ElviraNeural"
    assert result.segments[0].voice_id == "es-ES-AlvaroNeural"
    assert result.segments[1].voice_id == "es-ES-ElviraNeural"


@pytest.mark.asyncio
async def test_edge_tts_falls_back_when_assigned_voice_disappears(monkeypatch):
    from providers.tts.edge_tts_provider import EdgeTTSProvider

    _FakeCommunicate.instances = []
    _FakeCommunicate.fail_voices = {"es-ES-ElviraNeural"}  # assigned voice gone
    monkeypatch.setattr("edge_tts.Communicate", _FakeCommunicate)

    from core.domain import TranslatedTranscript, TranslatedSegment

    translated = TranslatedTranscript(
        target_language="es",
        segments=[
            TranslatedSegment(
                speaker_id="SPEAKER_01", start_ms=0, end_ms=500,
                source_text="bye", translated_text="adios", target_language="es",
            ),
        ],
    )

    result = await EdgeTTSProvider().synthesize(translated)

    # Retried with the language default (voices[0]); the failed attempt is
    # recorded too, proving the fallback path ran.
    assert [i.voice for i in _FakeCommunicate.instances] == ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"]
    assert result.segments[0].voice_id == "es-ES-AlvaroNeural"


@pytest.mark.asyncio
async def test_edge_tts_raises_when_fallback_also_fails(monkeypatch):
    from providers.tts.edge_tts_provider import EdgeTTSProvider

    _FakeCommunicate.instances = []
    _FakeCommunicate.fail_voices = {"es-ES-AlvaroNeural", "es-ES-ElviraNeural"}
    monkeypatch.setattr("edge_tts.Communicate", _FakeCommunicate)

    from core.domain import TranslatedTranscript, TranslatedSegment

    translated = TranslatedTranscript(
        target_language="es",
        segments=[
            TranslatedSegment(
                speaker_id="SPEAKER_01", start_ms=0, end_ms=500,
                source_text="bye", translated_text="adios", target_language="es",
            ),
        ],
    )

    with pytest.raises(RuntimeError):
        await EdgeTTSProvider().synthesize(translated)
