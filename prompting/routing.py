from .models import PromptContext
from .strategies import build_default_strategy_registry


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


class StrategyRouter:
    def __init__(self, strategy_name, registry=None):
        self.strategy = (registry or build_default_strategy_registry()).get(strategy_name)

    def route(self, problem):
        route = self.strategy.select_route(problem)
        category = route.category or problem.metadata.get("primary_category") or "general_math"
        tags = set(problem.tags)
        if category:
            tags.add(category)

        return PromptContext(
            problem=problem,
            strategy_name=self.strategy.name,
            route_name=route.name,
            tags=tags,
            metadata={
                "answer_format": problem.answer_format,
                "category": category,
                "qwen_categories": problem.metadata.get("qwen_categories") or [category],
                "strategy_label": self.strategy.label,
                "route_template_name": route.template_name or route.name,
            },
        )


class Baseline3Router(StrategyRouter):
    def __init__(self):
        super().__init__("baseline3")
