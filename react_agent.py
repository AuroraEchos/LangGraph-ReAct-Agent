"""A minimal LangGraph ReAct agent for learning purposes.

Core implementation ideas and structure are adapted from the open-source
LangGraph ReAct agent template:

    https://github.com/langchain-ai/react-agent

This file is intended only for personal study, code reading, and educational
experimentation. Please respect the original authors' work, preserve proper
attribution when reusing or modifying the code, and comply with the license of
the upstream repository.

This module defines a custom reasoning-and-acting agent graph.

Execution flow:

    START
      |
      v
    call_model
      |
      |-- if model emits tool calls --> tools --> call_model
      |
      |-- otherwise -----------------> END

The agent uses:
- Runtime context for configuration.
- State for conversation and execution history.
- ToolNode for tool execution.
- A simple conditional edge for ReAct-style looping.
"""


from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Callable, Literal, Sequence, cast

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime, get_runtime
from typing_extensions import Annotated


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. System time: {system_time}."
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5-20250929"
DEFAULT_MAX_SEARCH_RESULTS = 10


def _coerce_env_value(raw_value: str, default_value: Any) -> Any:
    """Convert an environment variable string to the type of its default value."""
    if isinstance(default_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}

    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)

    if isinstance(default_value, float):
        return float(raw_value)

    if isinstance(default_value, str):
        return raw_value

    return raw_value


@dataclass(kw_only=True)
class Context:
    """Runtime configuration for the agent.

    Context is static for a single graph run. It should contain configuration,
    not step-by-step execution data.

    Examples:
        - model name
        - system prompt
        - search result limit
    """

    system_prompt: str = field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "description": (
                "The system prompt to use for the agent's interactions. "
                "This prompt sets the context and behavior for the agent."
            )
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=DEFAULT_MODEL,
        metadata={
            "description": (
                "The language model to use for the agent. "
                "Expected format: provider/model-name."
            )
        },
    )

    max_search_results: int = field(
        default=DEFAULT_MAX_SEARCH_RESULTS,
        metadata={
            "description": (
                "The maximum number of search results to return for each search query."
            )
        },
    )

    def __post_init__(self) -> None:
        """Load environment variable overrides for fields left at default values.

        Example:
            MODEL="openai/gpt-4.1"
            MAX_SEARCH_RESULTS="5"

        Explicitly passed values take precedence over environment variables,
        unless the explicitly passed value is exactly equal to the default.
        """
        for config_field in fields(self):
            if not config_field.init:
                continue

            current_value = getattr(self, config_field.name)
            default_value = config_field.default

            if current_value != default_value:
                continue

            env_name = config_field.name.upper()
            raw_env_value = os.environ.get(env_name)

            if raw_env_value is None:
                continue

            try:
                coerced_value = _coerce_env_value(raw_env_value, default_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value for environment variable {env_name}: "
                    f"{raw_env_value!r}"
                ) from exc

            setattr(self, config_field.name, coerced_value)


async def search(query: str) -> dict[str, Any]:
    """Search the web for recent or source-backed information.

    Use this tool when answering requires:
    - current events
    - external facts
    - recent documentation
    - source-backed verification

    Do not use this tool for pure reasoning or information already available
    in the conversation.
    """
    runtime = get_runtime(Context)

    max_results = runtime.context.max_search_results

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = DEFAULT_MAX_SEARCH_RESULTS

    wrapped = TavilySearch(max_results=max_results)

    try:
        result = await wrapped.ainvoke({"query": query})
    except Exception as exc:
        return {
            "error": "search_failed",
            "query": query,
            "message": str(exc),
        }

    return cast(dict[str, Any], result)


TOOLS: list[Callable[..., Any]] = [search]


@dataclass
class InputState:
    """Public input state for the agent.

    This is the narrower interface exposed to external callers.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """Conversation and execution messages.

    Typical ReAct message pattern:

        1. HumanMessage
        2. AIMessage with tool_calls
        3. ToolMessage
        4. AIMessage without tool_calls
        5. HumanMessage

    The add_messages reducer appends new messages while also supporting
    message replacement by ID.
    """


@dataclass
class State(InputState):
    """Internal graph state for the agent.

    This extends InputState with graph-managed execution metadata.
    """

    is_last_step: IsLastStep = field(default=False)
    """Whether the graph is about to hit the recursion limit.

    This is controlled by LangGraph, not by user code.
    """


def get_message_text(message: BaseMessage) -> str:
    """Extract text content from a LangChain message.

    LangChain messages may contain:
    - plain strings
    - dict content blocks
    - lists of text / multimodal blocks

    Non-text blocks are ignored.
    """
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        text = content.get("text", "")
        return text if isinstance(text, str) else ""

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts).strip()

    return ""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a provider/model string.

    Args:
        fully_specified_name:
            Model name in the format "provider/model-name".

            Examples:
                "openai/gpt-4.1"
                "anthropic/claude-sonnet-4-5-20250929"
                "google_genai/gemini-2.5-pro"

    Returns:
        A LangChain BaseChatModel instance.
    """
    if "/" not in fully_specified_name:
        raise ValueError(
            "Model name must be in the format 'provider/model-name', "
            f"got {fully_specified_name!r}."
        )

    provider, model = fully_specified_name.split("/", maxsplit=1)

    if not provider or not model:
        raise ValueError(
            "Both provider and model must be non-empty in model name "
            f"{fully_specified_name!r}."
        )

    return init_chat_model(model, model_provider=provider)


async def call_model(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, list[AnyMessage]]:
    """Call the LLM that powers the agent.

    This node:
    1. Loads the configured chat model.
    2. Binds tools to the model.
    3. Builds the system prompt.
    4. Invokes the model with the current message history.
    5. Returns the model response as a state update.
    """
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    system_prompt = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    messages: list[AnyMessage] = [
        SystemMessage(content=system_prompt),
        *state.messages,
    ]

    response = cast(AIMessage, await model.ainvoke(messages))

    if state.is_last_step and response.tool_calls:
        fallback_response = AIMessage(
            id=response.id,
            content=(
                "Sorry, I could not find an answer to your question within "
                "the allowed number of steps."
            ),
        )
        return {"messages": [fallback_response]}

    return {"messages": [response]}


def route_model_output(state: State) -> Literal["tools", "__end__"]:
    """Route after the model node.

    If the latest AI message contains tool calls, continue to the tools node.
    Otherwise, finish the graph.
    """
    if not state.messages:
        raise ValueError("Expected at least one message in state, but found none.")

    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError(
            "Expected the last message to be an AIMessage after call_model, "
            f"but got {type(last_message).__name__}."
        )

    if last_message.tool_calls:
        return "tools"

    return END


builder = StateGraph(
    State,
    input_schema=InputState,
    context_schema=Context,
)

builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "call_model")

builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

builder.add_edge("tools", "call_model")

graph = builder.compile(name="ReAct Agent")


if __name__ == "__main__":
    import asyncio
    import os
    from langchain_core.messages import HumanMessage

    os.environ["API_KEY"] = ""
    os.environ["BASE_URL"] = ""
    os.environ["MODEL"] = ""
    os.environ["TAVILY_API_KEY"] = ""

    async def test() -> None:
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="请搜索一下 LangGraph 是什么，并简要总结。"),
                ]
            },
            context=Context(
                max_search_results=5,
            ),
        )

        for i, message in enumerate(result["messages"], start=1):
            print(f"\n--- Message {i}: {type(message).__name__} ---")
            print(message)

    asyncio.run(test())