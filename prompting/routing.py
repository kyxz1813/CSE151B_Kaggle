from .models import PromptContext
from .strategies import BASELINE3_CATEGORIES, build_default_strategy_registry, normalize_category


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

    def _normalize_problem_category(self, problem):
        primary = problem.metadata.get("primary_category")

        if primary not in BASELINE3_CATEGORIES:
            categories = problem.metadata.get("qwen_categories") or []
            if isinstance(categories, str):
                categories = [categories]

            primary = None
            for category in categories:
                category = normalize_category(category)
                if category != "general_math":
                    primary = category
                    break

            if primary is None:
                for category in categories:
                    category = normalize_category(category)
                    if category in BASELINE3_CATEGORIES:
                        primary = category
                        break

        primary = normalize_category(primary)

        problem.metadata["primary_category"] = primary
        problem.metadata["qwen_categories"] = [primary]

        return primary

    def route(self, problem):
        category = self._normalize_problem_category(problem)
        route = self.strategy.select_route(problem)

        if route.category is not None:
            category = route.category

        tags = set(problem.tags)
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
                "derived_rules": problem.metadata.get("derived_rules") or [],
                "rule_annotations": problem.metadata.get("rule_annotations") or {},
                "strategy_label": self.strategy.label,
                "route_template_name": route.template_name or route.name,
            },
        )


class Baseline3Router(StrategyRouter):
    def __init__(self):
        super().__init__("baseline3")
