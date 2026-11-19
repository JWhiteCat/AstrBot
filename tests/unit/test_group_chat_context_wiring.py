import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.api.message_components import At, Image, Json, Plain
from astrbot.api.provider import LLMResponse
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.builtin_stars.astrbot.main import Main
from astrbot.core.astr_main_agent.config import MainAgentBuildConfig
from astrbot.core.astr_main_agent.request import collect_initial_request
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.platform.message_type import MessageType


def make_main_with_conversation_manager(conv_mgr):
    main = Main.__new__(Main)
    main.context = MagicMock()
    main.context.get_using_provider_async = AsyncMock(
        side_effect=lambda *args, **kwargs: main.context.get_using_provider(
            *args,
            **kwargs,
        )
    )
    main.context.conversation_manager = conv_mgr
    return main


def _make_extras_store():
    """Return a mutable dict and get_extra / set_extra side_effects bound to it."""
    store: dict[str, object] = {}
    get_extra = lambda key, default=None: store.get(key, default)  # noqa: E731
    set_extra = store.__setitem__  # type: ignore[assignment]
    return store, get_extra, set_extra


def make_event(
    umo: str = "aiocqhttp:GroupMessage:user_123_group_456",
    *,
    handlers_parsed_params: dict | None = None,
):
    event = MagicMock()
    event.unified_msg_origin = umo
    event.get_platform_id.return_value = "aiocqhttp"
    event.message_obj = SimpleNamespace(message=[Plain("hello")])
    event.message_str = "hello"
    event.session_id = "session-1"

    store, get_extra, set_extra = _make_extras_store()
    # Simulate WakingCheckStage output: an empty dict means no command matched.
    store["handlers_parsed_params"] = (
        {} if handlers_parsed_params is None else handlers_parsed_params
    )
    event.get_extra.side_effect = get_extra
    event.set_extra.side_effect = set_extra
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_text", [None, "", " \n"])
async def test_bare_group_mention_continues_recent_group_context(
    empty_text, monkeypatch
):
    """A bare mention must collect earlier group messages through normal hooks.

    Args:
        empty_text: Optional empty text component supplied by the adapter.
        monkeypatch: Fixture for stopping before any external model request.
    """
    conversation = SimpleNamespace(cid="cid-1", history="[]")
    main = make_main_with_conversation_manager(
        SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid-1"),
            get_conversation=AsyncMock(return_value=conversation),
        )
    )
    main.context.get_config.return_value = {
        "platform_settings": {},
        "wake_prefix": ["/"],
        "provider_ltm_settings": {
            "group_icl_enable": True,
            "active_reply": {"enable": False},
        },
    }
    group_context = GroupChatContext(MagicMock(), main.context)
    group_context.cfg = MagicMock(return_value={"group_message_max_cnt": 100})
    group_context.need_active_reply = AsyncMock(return_value=False)
    group_context._format_message = AsyncMock(
        side_effect=["[Alice]: Why are game maps so complex?", "[Alice]: @bot"]
    )
    main.group_chat_context = group_context
    previous = make_event()
    previous.get_message_type.return_value = MessageType.GROUP_MESSAGE
    await group_context.handle_message(previous)

    event = make_event()
    event.message_str = ""
    event.get_message_type.return_value = MessageType.GROUP_MESSAGE
    event.get_self_id.return_value = "bot"
    components = [At(qq="bot")]
    if empty_text is not None:
        components.append(Plain(empty_text))
    event.message_obj.message = components
    event.get_messages.return_value = components

    assert [item async for item in main.handle_empty_mention(event)] == []
    assert [item async for item in main.on_message(event)] == []
    stage = internal.InternalAgentSubStage()
    stage.streaming_response = False
    stage.show_reasoning = False
    event.send_typing = AsyncMock()
    event.stop_typing = AsyncMock()
    waiting_hook = AsyncMock(return_value=True)
    monkeypatch.setattr(internal, "call_event_hook", waiting_hook)
    monkeypatch.setattr(internal, "try_capture_follow_up", lambda event: None)
    assert [item async for item in stage.process(event, "")] == []
    waiting_hook.assert_awaited_once()

    req, _ = await collect_initial_request(
        event, main.context, MainAgentBuildConfig(tool_call_timeout=60)
    )
    assert req is not None
    await group_context.on_req_llm(event, req)

    assert req.prompt == event.get_extra("_empty_group_mention_prompt")
    assert req.prompt
    assert req.conversation is conversation
    assert "Why are game maps so complex?" in req.extra_user_content_parts[0].text
    assert event.message_str == ""
    assert event.message_obj.message == components
    event.stop_event.assert_not_called()
    event.request_llm.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "components",
    [[At(qq="other")], [At(qq="bot"), Plain("hello")], [At(qq="bot"), Image(file="x")]],
)
async def test_contextual_mention_does_not_replace_other_message_types(components):
    """Preserve other mentions and explicit text or media inputs.

    Args:
        components: A message that should not become a contextual bare mention.
    """
    main = make_main_with_conversation_manager(SimpleNamespace())
    main.context.get_config.return_value = {"platform_settings": {}, "wake_prefix": []}
    event = make_event()
    event.get_self_id.return_value = "bot"
    event.get_messages.return_value = components
    assert [item async for item in main.handle_empty_mention(event)] == []
    assert event.get_extra("_empty_group_mention_prompt") is None


@pytest.mark.asyncio
async def test_active_reply_does_not_create_conversation_when_current_missing():
    conv_mgr = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=None),
        new_conversation=AsyncMock(),
        get_conversation=AsyncMock(),
    )
    main = make_main_with_conversation_manager(conv_mgr)
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_icl_enable": False,
            "active_reply": {"enable": True},
        },
    }
    main.context.get_using_provider.return_value = object()
    main.group_chat_context = SimpleNamespace(
        need_active_reply=AsyncMock(return_value=True),
        handle_message=AsyncMock(),
    )
    event = make_event()

    results = [item async for item in main.on_message(event)]

    assert results == []
    conv_mgr.get_curr_conversation_id.assert_awaited_once_with(event.unified_msg_origin)
    conv_mgr.new_conversation.assert_not_called()
    conv_mgr.get_conversation.assert_not_called()
    event.request_llm.assert_not_called()


@pytest.mark.asyncio
async def test_active_reply_reuses_current_umo_conversation():
    conv = SimpleNamespace(cid="cid-1")
    conv_mgr = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="cid-1"),
        new_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value=conv),
    )
    main = make_main_with_conversation_manager(conv_mgr)
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_icl_enable": False,
            "active_reply": {"enable": True},
        },
    }
    main.context.get_using_provider.return_value = object()
    main.group_chat_context = SimpleNamespace(
        need_active_reply=AsyncMock(return_value=True),
        handle_message=AsyncMock(),
    )
    event = make_event("aiocqhttp:GroupMessage:user_999_group_456")
    llm_request = object()
    event.request_llm.return_value = llm_request

    results = [item async for item in main.on_message(event)]

    assert results == [llm_request]
    conv_mgr.get_curr_conversation_id.assert_awaited_once_with(event.unified_msg_origin)
    conv_mgr.new_conversation.assert_not_called()
    conv_mgr.get_conversation.assert_awaited_once_with(
        event.unified_msg_origin,
        "cid-1",
    )
    event.request_llm.assert_called_once_with(
        prompt="hello",
        session_id="session-1",
        image_urls=[],
        conversation=conv,
    )


@pytest.mark.asyncio
async def test_on_message_does_not_clear_group_context_on_first_enabled_message():
    main = Main.__new__(Main)
    main.context = MagicMock()
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_icl_enable": True,
            "active_reply": {"enable": False},
        },
    }
    main.group_chat_context = SimpleNamespace(
        need_active_reply=AsyncMock(return_value=False),
        handle_message=AsyncMock(),
        remove_session=AsyncMock(),
    )
    event = make_event()

    async for _ in main.on_message(event):
        pass

    main.group_chat_context.need_active_reply.assert_awaited_once_with(event)
    main.group_chat_context.handle_message.assert_awaited_once_with(event)
    main.group_chat_context.remove_session.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_records_json_card_and_checks_active_reply():
    main = Main.__new__(Main)
    main.context = MagicMock()
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_icl_enable": True,
            "active_reply": {"enable": False},
        },
    }
    main.group_chat_context = SimpleNamespace(
        need_active_reply=AsyncMock(return_value=False),
        handle_message=AsyncMock(),
    )
    event = make_event()
    event.message_obj.message = [Json(data={"meta": {"news": {"title": "News"}}})]

    async for _ in main.on_message(event):
        pass

    main.group_chat_context.need_active_reply.assert_awaited_once_with(event)
    main.group_chat_context.handle_message.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_on_message_skips_recording_when_command_handler_matched():
    """A slash-command message (handlers_parsed_params non-empty) must not be
    recorded into the group context buffer."""
    main = Main.__new__(Main)
    main.context = MagicMock()
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_icl_enable": True,
            "active_reply": {"enable": False},
        },
    }
    main.group_chat_context = SimpleNamespace(
        need_active_reply=AsyncMock(return_value=False),
        handle_message=AsyncMock(),
    )
    event = make_event(
        handlers_parsed_params={
            "astrbot.builtin_stars.builtin_commands.main_reset": {}
        },
    )

    async for _ in main.on_message(event):
        pass

    main.group_chat_context.need_active_reply.assert_awaited_once_with(event)
    main.group_chat_context.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_response_persists_final_complete_chain():
    """Persist the final runner response for an enabled group session."""
    main = Main.__new__(Main)
    main.context = MagicMock()
    main.context.get_config.return_value = {
        "provider_ltm_settings": {
            "group_message_history_enable": True,
            "group_message_history_max_cnt": 700,
        },
    }
    main.context.message_history_manager.insert_message_chain = AsyncMock()
    event = make_event()
    event.get_message_type.return_value = MessageType.GROUP_MESSAGE
    event.get_platform_name.return_value = "aiocqhttp"
    event.get_self_id.return_value = "bot-1"
    response = LLMResponse(
        role="assistant",
        result_chain=MessageChain([Plain("complete response")]),
    )

    await main.persist_llm_response(event, response)

    main.context.message_history_manager.insert_message_chain.assert_awaited_once_with(
        platform_id="aiocqhttp",
        user_id=event.unified_msg_origin,
        message_chain=response.result_chain,
        role="bot",
        sender_id="bot-1",
        sender_name="bot",
        max_messages=700,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("card_data", "expected"),
    [
        (
            {
                "meta": {
                    "detail_1": {
                        "title": "WeChat AI models",
                        "desc": "AI learning\nwith examples",
                        "qqdocurl": "https://example.com/detail",
                    }
                }
            },
            " [Shared Card: Title: WeChat AI models; Description: AI learning "
            "with examples; URL: https://example.com/detail]",
        ),
        (
            {
                "data": json.dumps(
                    {
                        "meta": {
                            "news": {
                                "title": "Wrapped card",
                                "jumpUrl": "https://example.com/news",
                            }
                        }
                    }
                )
            },
            " [Shared Card: Title: Wrapped card; URL: https://example.com/news]",
        ),
        ({"app": "com.example.unknown"}, " [Shared Card]"),
    ],
)
async def test_format_message_summarizes_json_card(card_data, expected):
    context = GroupChatContext(MagicMock(), MagicMock())
    event = MagicMock()
    event.message_obj = SimpleNamespace(sender=SimpleNamespace(nickname="Alice"))
    event.get_messages.return_value = [Json(data=card_data)]

    formatted = await context._format_message(event, {})

    assert formatted.endswith(expected)


@pytest.mark.asyncio
async def test_format_message_truncates_long_json_card_fields():
    context = GroupChatContext(MagicMock(), MagicMock())
    event = MagicMock()
    event.message_obj = SimpleNamespace(sender=SimpleNamespace(nickname="Alice"))
    event.get_messages.return_value = [
        Json(
            data={
                "meta": {"news": {"desc": "a" * 201}},
            }
        )
    ]

    formatted = await context._format_message(event, {})

    assert f"Description: {'a' * 200}...]" in formatted
