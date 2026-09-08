"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
    - disabled on Ollama  → extra_body.think = False + reasoning_effort=none
    - disabled elsewhere  → reasoning_effort=none, no think (strict APIs 422)
    - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect), passed through verbatim
                          including ``max``/``xhigh``
    - enabled + no effort → nothing emitted (endpoint's server default applies)
    - ollama_num_ctx      → extra_body.options.num_ctx, orthogonal to reasoning
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False on an Ollama URL → reasoning_effort='none' + think=False.

        Both fields are required on Ollama: /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → same dual emission on Ollama."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="qwen3",
            base_url="http://localhost:11434/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_omits_think_on_mistral(self, custom_profile):
        """Strict OpenAI-compat hosts forbid extra ``think`` (HTTP 422)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_omits_think_without_base_url(self, custom_profile):
        """Unknown custom endpoint — do not send the Ollama-only flag."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://127.0.0.1:8080/v1",
            "http://localhost:1234/v1",
            "https://api.groq.com/openai/v1",
        ],
    )
    def test_disabled_omits_think_on_non_ollama_relays(self, custom_profile, base_url):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"},
            model="llama3",
            base_url=base_url,
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_sends_think_false_on_ollama_cloud_host(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url="https://ollama.com/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://myhost:99999/v1",  # out-of-range port: OpenAI client accepts it
            "http://localhost:80a/v1",  # non-integer port
            "http://localhost:11434./v1",  # trailing-dot port
        ],
    )
    def test_malformed_port_does_not_raise(self, custom_profile, base_url):
        """Malformed ports must not raise — urlparse's ``port`` is ValueError-happy.

        The OpenAI client accepts ``http://myhost:99999/v1`` at construction
        (only httpx fails later), so these URLs reach ``build_api_kwargs_extras``
        in production. The heuristic must treat them as non-Ollama rather than
        killing the kwargs build.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url=base_url,
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb


    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}


class TestCustomReasoningRouteDeclaredEfforts:
    """A configured route's ``supported_efforts`` beats the generic wire vocabulary.

    An aggregator fronted by ``provider=custom`` may publish effort-pinned model aliases
    (``vendor/model-low``) that accept exactly one level, and may be permissive enough to
    return 200 for a level it silently ignores. Clamping to the declared set is what keeps
    the request honest; a silent route keeps the previous generic behaviour.
    """

    ROUTE = "https://aggregator.invalid/v1"

    @pytest.fixture
    def declared(self, monkeypatch):
        """Patch the config reader at its use site in the plugin module."""
        import providers

        module = type(providers.get_provider_profile("custom")).__module__
        import sys

        def _install(value):
            monkeypatch.setattr(
                sys.modules[module], "_declared_route_efforts", lambda model, base_url: value
            )

        return _install

    def test_effort_clamps_down_to_the_declared_level(self, custom_profile, declared):
        declared(("low",))
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="vendor/model-low",
            base_url=self.ROUTE,
        )
        assert tl == {"reasoning_effort": "low"}

    def test_effort_within_the_declared_set_passes_through(self, custom_profile, declared):
        declared(("low", "medium", "high"))
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="vendor/model",
            base_url=self.ROUTE,
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_hermes_only_effort_clamps_into_the_declared_set(self, custom_profile, declared):
        """``ultra`` is Hermes-internal; no wire takes it."""
        declared(("low", "high"))
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            model="vendor/model",
            base_url=self.ROUTE,
        )
        assert tl == {"reasoning_effort": "high"}

    def test_route_declaring_no_efforts_emits_nothing(self, custom_profile, declared):
        declared(())
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="vendor/plain",
            base_url=self.ROUTE,
        )
        assert eb == {}
        assert tl == {}

    @pytest.mark.parametrize(
        "reasoning_config",
        [{"enabled": False}, {"enabled": True, "effort": "none"}],
    )
    def test_disable_is_dropped_when_the_route_cannot_disable(
        self, custom_profile, declared, reasoning_config
    ):
        """Declared efforts without ``none`` mean thinking is mandatory.

        Emitting ``reasoning_effort="none"`` there is worse than emitting nothing: a permissive
        aggregator treats any reasoning field as an explicit client choice and skips the per-model
        default it would otherwise inject, so the request ships with no level at all.
        """
        declared(("low", "high"))
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config, model="vendor/model", base_url=self.ROUTE
        )
        assert eb == {}
        assert tl == {}

    def test_disable_survives_when_the_route_declares_none(self, custom_profile, declared):
        declared(("none", "low", "high"))
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="vendor/model", base_url=self.ROUTE
        )
        assert tl == {"reasoning_effort": "none"}

    def test_silent_route_keeps_the_generic_wire_behaviour(self, custom_profile, declared):
        declared(None)
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="glm-5.2",
            base_url=self.ROUTE,
        )
        assert tl == {"reasoning_effort": "max"}
        _, tl_off = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2", base_url=self.ROUTE
        )
        assert tl_off == {"reasoning_effort": "none"}

    def test_num_ctx_still_rides_along_when_the_effort_is_dropped(self, custom_profile, declared):
        declared(())
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="vendor/plain",
            base_url=self.ROUTE,
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}
