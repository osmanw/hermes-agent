"""Pinned non-ACP external-process providers must keep their own transport.

Regression for the CEO-lab defect: ``delegate_tool_config._resolve_child_runtime`` rewrote the
child's provider to ``copilot-acp`` whenever a pinned command was present, so a delegation pinned
to an out-of-tree external-process provider (Devin CLI OAuth, whose ProviderProfile supplies its
own HTTP inference client) spawned ``<cli> --acp --stdio`` and died with "ACP transport not
supported" — while the identical route worked on the parent/CLI path.
"""
import unittest
from unittest.mock import MagicMock, patch

from providers import register_provider
from providers.base import ProviderProfile
from tools.delegate_tool_config import _provider_owns_its_transport, _resolve_child_runtime


class _OwnTransportProfile(ProviderProfile):
    """Profile that supplies its own client (not the ACP stdio shim)."""

    def create_client(self, **client_kwargs):
        return MagicMock(name="own-transport-client")


_OWN = _OwnTransportProfile(
    name="ceolab-owntransport",
    api_mode="chat_completions",
    env_vars=(),
    base_url="ceolab://inference",
    auth_type="external_process",
    process_command="/bin/sh",
    process_args=(),
)
register_provider(_OWN)


def _parent():
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.acp_command = None
    parent.acp_args = []
    parent._fallback_chain = None
    parent.max_tokens = None
    return parent


def _runtime(**over):
    kwargs = dict(
        parent_agent=_parent(), delegation_cfg={}, parent_api_key="***", model="some-model",
        override_provider=None, override_base_url=None, override_api_key=None,
        override_api_mode=None, override_acp_command=None, override_acp_args=None,
        routing_cfg=None,
    )
    kwargs.update(over)
    with patch("shutil.which", return_value="/bin/sh"):
        return _resolve_child_runtime(**kwargs)


class TestPinnedProviderTransport(unittest.TestCase):
    def test_profile_with_own_client_owns_its_transport(self):
        self.assertTrue(_provider_owns_its_transport("ceolab-owntransport"))

    def test_plain_http_provider_does_not_own_transport(self):
        self.assertFalse(_provider_owns_its_transport("openrouter"))

    def test_unknown_provider_does_not_own_transport(self):
        self.assertFalse(_provider_owns_its_transport("nope-not-registered"))
        self.assertFalse(_provider_owns_its_transport(""))
        self.assertFalse(_provider_owns_its_transport(None))

    def test_pinned_own_transport_provider_is_not_rewritten_to_copilot_acp(self):
        rt = _runtime(override_provider="ceolab-owntransport", override_acp_command="/bin/sh",
                      override_acp_args=[])
        self.assertEqual(rt["provider"], "ceolab-owntransport")
        self.assertNotEqual(rt["provider"], "copilot-acp")
        self.assertEqual(rt["acp_command"], "/bin/sh")

    def test_bare_command_pin_without_provider_still_uses_copilot_acp(self):
        rt = _runtime(override_acp_command="/bin/sh", override_acp_args=["--acp", "--stdio"])
        self.assertEqual(rt["provider"], "copilot-acp")
        self.assertEqual(rt["api_mode"], "chat_completions")

    def test_pinned_provider_without_own_profile_still_uses_copilot_acp(self):
        rt = _runtime(override_provider="openrouter", override_acp_command="/bin/sh",
                      override_acp_args=["--acp", "--stdio"])
        self.assertEqual(rt["provider"], "copilot-acp")

    def test_pinned_provider_without_command_is_unchanged(self):
        rt = _runtime(override_provider="ceolab-owntransport")
        self.assertEqual(rt["provider"], "ceolab-owntransport")
        self.assertIsNone(rt["acp_command"])
        self.assertEqual(rt["acp_args"], [])


if __name__ == "__main__":
    unittest.main()
