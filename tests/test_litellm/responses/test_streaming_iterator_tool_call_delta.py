"""
Some upstreams (e.g. codex-proxy-rs for parallel tool calls) stream
function calls as output_item.added -> function_call_arguments.done ->
output_item.done with no function_call_arguments.delta. Clients such as
LangChain only read tool arguments from .delta events, so the native
Responses streaming iterator must synthesize one from output_item.done.
"""
import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents


def _function_call_item(index, arguments):
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "read_file",
        "arguments": arguments,
        "status": "completed",
    }


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event)


def _parallel_without_delta(paths):
    events = []
    for index, args in enumerate(paths):
        events += [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": _function_call_item(index, ""),
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": index,
                "item_id": f"fc_{index}",
                "arguments": args,
            },
            {
                "type": "response.output_item.done",
                "output_index": index,
                "item": _function_call_item(index, args),
            },
        ]
    return [_sse(e) for e in events]


def _single_with_delta(args):
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": _function_call_item(0, ""),
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_0",
            "delta": args[:10],
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_0",
            "delta": args[10:],
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": _function_call_item(0, args),
        },
    ]
    return [_sse(e) for e in events]


def _make_logging_obj():
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return logging_obj


def _make_response():
    response = MagicMock()
    response.headers = {}
    return response


class _AsyncLines:
    def __init__(self, lines):
        self._it = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _summarize(events):
    types = [getattr(e.type, "value", e.type) for e in events]
    delta_args = {}
    done_args = {}
    for e in events:
        t = getattr(e.type, "value", e.type)
        if t == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA.value:
            delta_args[e.output_index] = delta_args.get(e.output_index, "") + e.delta
        if t == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value:
            done_args[e.output_index] = getattr(e.item, "arguments", None)
    return types, delta_args, done_args


async def _collect_async(lines):
    iterator = ResponsesAPIStreamingIterator(
        response=_make_response(),
        model="gpt-6.1-sol",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    iterator.stream_iterator = _AsyncLines(lines)
    iterator._call_post_streaming_deployment_hook = _identity_hook
    out = []
    async for e in iterator:
        out.append(e)
    return out


async def _identity_hook(chunk):
    return chunk


def _collect_sync(lines):
    iterator = SyncResponsesAPIStreamingIterator(
        response=_make_response(),
        model="gpt-6.1-sol",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    iterator.stream_iterator = iter(lines)
    iterator._call_post_streaming_deployment_hook = _identity_hook
    return list(iterator)


@pytest.mark.asyncio
async def test_async_parallel_tool_calls_without_delta_get_synthesized_delta():
    paths = ['{"path": "/etc/hosts"}', '{"path": "/etc"}']
    types, delta_args, done_args = _summarize(
        await _collect_async(_parallel_without_delta(paths))
    )
    assert delta_args == {0: paths[0], 1: paths[1]}
    assert done_args == {0: paths[0], 1: paths[1]}
    # synthesized delta must precede the real output_item.done for each item
    assert types == [
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.function_call_arguments.delta",
        "response.output_item.done",
    ] * 2


def test_sync_parallel_tool_calls_without_delta_get_synthesized_delta():
    paths = ['{"path": "/etc/hosts"}', '{"path": "/etc"}']
    types, delta_args, done_args = _summarize(
        _collect_sync(_parallel_without_delta(paths))
    )
    assert delta_args == {0: paths[0], 1: paths[1]}
    assert types.count("response.function_call_arguments.delta") == 2
    assert types.count("response.output_item.done") == 2


@pytest.mark.asyncio
async def test_tool_call_with_real_deltas_is_not_duplicated():
    args = '{"path": "/etc/hosts"}'
    types, delta_args, _ = _summarize(await _collect_async(_single_with_delta(args)))
    assert delta_args == {0: args}
    assert types.count("response.function_call_arguments.delta") == 2
    assert types.count("response.output_item.done") == 1


@pytest.mark.asyncio
async def test_non_function_call_items_untouched():
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_0", "role": "assistant", "content": []},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_0",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            },
        },
    ]
    types, delta_args, _ = _summarize(await _collect_async([_sse(e) for e in events]))
    assert delta_args == {}
    assert types == ["response.output_item.added", "response.output_item.done"]
