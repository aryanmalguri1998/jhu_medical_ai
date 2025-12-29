"""Helpers for running exported Agent Builder workflows locally."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import import_module
from typing import Any, Mapping

from agents import Agent, Runner

DEFAULT_WORKFLOW_MODULE = "app.workflow_export"
MAX_SESSION_TURNS = 40

_SESSION_CONVERSATIONS: dict[str, list[dict[str, Any]]] = {}


def _conversation_for(session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return []
    return list(_SESSION_CONVERSATIONS.get(session_id, []))


def _store_conversation(session_id: str, conversation: list[dict[str, Any]]) -> None:
    if not session_id:
        return
    if len(conversation) > MAX_SESSION_TURNS:
        conversation = conversation[-MAX_SESSION_TURNS:]
    _SESSION_CONVERSATIONS[session_id] = conversation


class WorkflowAgentError(RuntimeError):
    """Raised when the exported workflow agent cannot be loaded."""


def _module_name() -> str:
    return os.getenv("WORKFLOW_MODULE", DEFAULT_WORKFLOW_MODULE)


@lru_cache(maxsize=1)
def _load_workflow_module():
    module_name = _module_name()
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - configuration issue
        raise WorkflowAgentError(
            "Workflow module '{module}' was not found. Export your Agent Builder workflow "
            "to backend/app/workflow_export.py or set WORKFLOW_MODULE to a valid module path."
        .format(module=module_name)) from exc


@lru_cache(maxsize=1)
def _load_agent() -> Agent[Any]:
    module = _load_workflow_module()
    module_name = module.__name__

    agent = getattr(module, "root_agent", None) or getattr(module, "agent", None)
    if agent is None:
        raise WorkflowAgentError(
            f"Module '{module_name}' does not expose a 'root_agent'."
        )

    return agent


def _build_workflow_context(payload_text: str) -> Any | None:
    module = _load_workflow_module()

    builder = getattr(module, "build_workflow_context", None)
    if callable(builder):
        return builder(payload_text)

    context_cls = getattr(module, "ClinicalAgentContext", None)
    if context_cls is None:
        return None
    try:
        return context_cls(payload_text)
    except TypeError:
        return None


async def execute_workflow(
    payload: Mapping[str, Any], session_id: str | None = None
) -> Any:
    """Run the exported workflow with the provided payload."""

    agent = _load_agent()
    payload_text = json.dumps(payload)
    context = _build_workflow_context(payload_text)

    user_message = {
        "role": "user",
        "content": [{"type": "input_text", "text": payload_text}],
    }
    conversation_history = _conversation_for(session_id)
    baton = [*conversation_history, user_message]

    run_kwargs = {"input": baton}
    if context is not None:
        run_kwargs["context"] = context

    result = await Runner.run(agent, **run_kwargs)

    if session_id:
        updated = [*baton]
        new_items = getattr(result, "new_items", None)
        if new_items:
            for item in new_items:
                to_input = getattr(item, "to_input_item", None)
                if callable(to_input):
                    updated.append(to_input())
        _store_conversation(session_id, updated)

    final_output = getattr(result, "final_output", None)
    if final_output is not None:
        return final_output

    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()

    return result
