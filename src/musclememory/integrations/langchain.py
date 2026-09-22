"""LangChain / LangGraph: the learning tools as ``StructuredTool`` objects.

    from musclememory.integrations.langchain import langchain_tools
    tools = my_tools + langchain_tools(session)
    ...
    session.end_turn(state["messages"])   # LangChain message objects are read natively
"""

from __future__ import annotations

from ..session import Session
from ..tools import tool_specs


def langchain_tools(session: Session) -> list:
    from langchain_core.tools import StructuredTool

    def make(name: str):
        def run(**kwargs) -> str:
            return session.handle_tool_call(name, kwargs)
        return run

    return [
        StructuredTool.from_function(func=make(spec["name"]), name=spec["name"],
                                     description=spec["description"], args_schema=spec["parameters"])
        for spec in tool_specs(session._learner.config.description_max_chars)
    ]
