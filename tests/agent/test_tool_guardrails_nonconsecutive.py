"""Non-consecutive identical-call replay detection (tool_guardrails).

The pre-existing identical-call streak in ``observe_call`` only counts
*consecutive* repeats; a model that alternates a repeat with other calls
(read A -> grep B -> read A) resets it and was never flagged. The DB audit
found 1,023 such non-consecutive excess calls vs 700 consecutive. The
``_repeat_call_counts`` tracker catches them.
"""

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)


def _obs(controller, tool, args, result, failed=False):
    return controller.observe_call(tool, args, result, failed=failed)


def test_nonconsecutive_identical_call_fires_notice():
    c = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "cat big.log"}
    # Interleave the repeated call with a DIFFERENT call so the consecutive
    # streak never reaches the threshold.
    notices = []
    for i in range(4):
        obs = _obs(c, "terminal", args, "same-output\n")
        if obs.notice:
            notices.append(obs.notice)
        _obs(c, "search_files", {"pattern": f"other{i}"}, "x\n")  # intervening
    assert notices, "no notice fired for a non-consecutive identical replay"
    assert "not consecutive" in notices[0]


def test_nonconsecutive_replay_after_result_change_restarts_count():
    c = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "cat f"}
    for _ in range(2):
        _obs(c, "terminal", args, "v1\n")
    # File changes -> result hash changes -> counter restarts, no notice.
    obs = _obs(c, "terminal", args, "v2\n")
    assert obs.notice is None
    obs = _obs(c, "terminal", args, "v2\n")
    assert obs.notice is None  # only 2nd sighting of v2


def test_nonconsecutive_replay_halts_under_hard_stop():
    c = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=4)
    )
    args = {"command": "ls"}
    for i in range(4):
        c.observe_call("terminal", args, "x\n", failed=False)
        if i < 3:
            c.observe_call("search_files", {"pattern": "q"}, "y\n", failed=False)
    halt = c.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.count == 4


def test_consecutive_streak_still_fires_first_notice():
    # Regression: consecutive identical calls still use the original notice text.
    c = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "pwd"}
    for _ in range(2):
        _obs(c, "terminal", args, "/home\n")
    obs = _obs(c, "terminal", args, "/home\n")
    assert obs.notice is not None and "consecutive identical call" in obs.notice
