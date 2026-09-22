"""musclememory — a drop-in self-learning loop for any LLM agent.

The agent keeps two stores it reads in every future conversation — short memory entries and
on-demand skills — and a background reviewer distils each conversation into them::

    from musclememory import SelfLearner, anthropic_llm
    learner = SelfLearner("./.musclememory", llm=anthropic_llm(anthropic.Anthropic()))
    with learner.session() as session:
        system = BASE_PROMPT + "\\n\\n" + session.system_prompt()
        tools = MY_TOOLS + session.tools("anthropic")
        ...                       # route tool calls with session.handles / handle_tool_call
        session.end_turn(messages)  # after each reply
"""

from .config import LearnerConfig
from .learner import SelfLearner
from .library import Library, OpResult, SkillInfo
from .llm import LLMRefusal, anthropic_llm, openai_llm
from .review import ReviewResult, run_review
from .session import Session
from .store import FileStore

__all__ = [
    "FileStore", "LLMRefusal", "LearnerConfig", "Library", "OpResult", "ReviewResult", "SelfLearner",
    "Session", "SkillInfo", "anthropic_llm", "openai_llm", "run_review",
]
__version__ = "0.1.0"
