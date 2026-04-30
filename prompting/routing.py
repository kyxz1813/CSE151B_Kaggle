import re

from .models import PromptContext


class RuleBasedProblemRouter:
    def __init__(self, default_strategy_name="baseline"):
        self.default_strategy_name = default_strategy_name

    def route(self, problem):
        tags = set(problem.tags)

        if problem.is_mcq:
            route_name = "mcq"
        else:
            route_name = "free_form"

        return PromptContext(
            problem=problem,
            strategy_name=self.default_strategy_name,
            route_name=route_name,
            tags=tags,
            metadata={"answer_format": problem.answer_format},
        )


class FixedStrategyRouter(RuleBasedProblemRouter):
    def __init__(self, strategy_name="baseline"):
        super().__init__(default_strategy_name=strategy_name)