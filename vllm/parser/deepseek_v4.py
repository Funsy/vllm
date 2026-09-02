# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
DSML_FOREIGN_TOOL_START = f"<{_DSML}function_calls>"
DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"

_ESCAPED_DSML = re.escape(_DSML)
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*?)"
    rf"(?:</{_ESCAPED_DSML}parameter>|(?=<{_ESCAPED_DSML}parameter\s+name=)|$)",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _dsml_args_are_well_formed(raw_args: str) -> bool:
    """Return whether every non-whitespace byte belongs to a DSML parameter."""
    end = 0
    found = False
    for match in _PARAM_RE.finditer(raw_args):
        if raw_args[end : match.start()].strip():
            return False
        found = True
        end = match.end()
    if raw_args[end:].strip():
        return False
    return found or not raw_args.strip()


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


def _convert_dsml_args(
    raw_args: str,
    partial: bool,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    result = _dsml_arg_converter(raw_args, partial)
    if not tools:
        return result
    unwrapped = _unwrap_wrapper_args(result, tools, func_name)
    if partial and unwrapped == result and func_name:
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, ValueError, TypeError):
            return result
        properties = find_tool_properties(tools, func_name)
        for wrapper in ("arguments", "input"):
            if set(parsed) == {wrapper} and wrapper not in properties:
                # Do not stream a wrapper key that may disappear once its
                # nested JSON value becomes complete and can be unwrapped.
                return ""
    return unwrapped


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
            "FOREIGN_START": DSML_FOREIGN_TOOL_START,
            "FOREIGN_END": DSML_FOREIGN_TOOL_END,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            # Keep V3.2 wrappers verbatim so their inner invokes cannot enter
            # the V4 orphan-recovery path.
            (ParserState.CONTENT, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.FOREIGN_BLOCK, "FOREIGN_END"): Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.FOREIGN_BLOCK, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_REASONING_BLOCK,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "FOREIGN_END",
            ): Transition(
                ParserState.REASONING,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "TOOL_START",
            ): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            # DeepSeek V4 can intermittently omit or corrupt the outer
            # <｜DSML｜tool_calls> wrapper while still emitting a complete
            # invoke. Hold this recovery path until the function name is
            # complete and verify that the request actually declared it.
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                provisional_tool_call=True,
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
                commit_provisional_tool_call=True,
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,
            ParserState.FOREIGN_REASONING_BLOCK: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
        validate_tool_names=True,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._recovery_request_tools: list[Tool] = list(tools or [])
        self._recovery_suppressed = False
        self._engine.recovery_tool_name_validator = self._can_recover_tool_name
        self._engine.recovery_tool_name_prefix_validator = (
            self._can_recover_tool_name_prefix
        )
        self._engine.recovery_tool_call_validator = self._can_recover_tool_call
        self._engine.allow_unvalidated_wrapped_tool_call = (
            not self._recovery_request_tools
        )

    def _check_skip_tool_parsing(self, request) -> None:
        super()._check_skip_tool_parsing(request)
        request_tools = getattr(request, "tools", None)
        if request_tools is not None:
            self._recovery_request_tools = list(request_tools)
        else:
            self._recovery_request_tools = list(self._tools or [])
        self._recovery_suppressed = self._suppress_tool_calls
        self._engine.allow_unvalidated_wrapped_tool_call = bool(
            not self._recovery_request_tools or self._recovery_suppressed
        )

    def _can_recover_tool_name(self, name: str) -> bool:
        return bool(
            name
            and self._recovery_request_tools
            and not self._recovery_suppressed
            and name in self._declared_tool_names()
        )

    @staticmethod
    def _tool_name_and_parameters(tool: Tool) -> tuple[str | None, object]:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("name"), function.get("parameters")
            return tool.get("name"), tool.get("parameters")
        function = getattr(tool, "function", None)
        name = getattr(function, "name", None) or getattr(tool, "name", None)
        parameters = getattr(function, "parameters", None) or getattr(
            tool, "parameters", None
        )
        return name, parameters

    def _declared_tool_names(self) -> list[str]:
        names: list[str] = []
        for tool in self._recovery_request_tools:
            name, _ = self._tool_name_and_parameters(tool)
            if isinstance(name, str):
                names.append(name)
        return names

    def _can_recover_tool_name_prefix(self, prefix: str) -> bool:
        return bool(
            not self._recovery_suppressed
            and len(prefix) <= 256
            and any(name.startswith(prefix) for name in self._declared_tool_names())
        )

    def _required_tool_params(self, name: str) -> set[str]:
        for tool in self._recovery_request_tools:
            tool_name, parameters = self._tool_name_and_parameters(tool)
            if tool_name != name:
                continue
            if isinstance(parameters, dict):
                required = parameters.get("required", [])
                if isinstance(required, list):
                    return {item for item in required if isinstance(item, str)}
            break
        return set()

    def _can_recover_tool_call(self, name: str, raw_args: str) -> bool:
        if not _dsml_args_are_well_formed(raw_args):
            return False
        try:
            parsed = json.loads(_dsml_arg_converter(raw_args, partial=False))
        except (json.JSONDecodeError, ValueError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        result = _unwrap_wrapper_args(
            json.dumps(parsed, ensure_ascii=False),
            self._recovery_request_tools,
            name,
        )
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, ValueError, TypeError):
            return False
        return self._required_tool_params(name).issubset(parsed)

    def _convert_tool_args(self, idx: int, raw_args: str, partial: bool) -> str:
        func_name = self._tool_slots[idx].name if idx < len(self._tool_slots) else None
        return _convert_dsml_args(raw_args, partial, self._tools, func_name)
