"""Structured replies of the agents that answer in JSON (see `formatting.parse_route`)."""

from typing import Any, Literal

from pydantic import BaseModel

Intent = Literal["data", "chat", "clarify"]


class RouteDecision(BaseModel):
    """What the router decided about a message."""

    intent: Intent  # data: write SQL · chat: not about the data · clarify: too vague to write SQL
    standalone_question: str = ""  # the message rewritten to stand without the conversation
    unclear: str = ""  # for "clarify": what is missing or ambiguous


def json_schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema to constrain a model's reply with. Every field is required there, so the
    model fills them all in (the defaults only serve replies that skip some)."""
    schema = model.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


ROUTE_SCHEMA = json_schema_of(RouteDecision)
