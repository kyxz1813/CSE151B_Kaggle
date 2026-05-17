from dataclasses import dataclass, field


BASELINE3_CATEGORIES = (
    "statistics_probability",
    "calculus",
    "geometry_trig",
    "linear_algebra",
    "discrete_algorithm",
    "arithmetic_algebra",
    "applied_word_problem",
    "general_math",
)


def normalize_category(category):
    category = str(category or "").strip()
    if category in BASELINE3_CATEGORIES:
        return category
    return "general_math"


@dataclass(frozen=True)
class RouteDefinition:
    name: str
    answer_format: str | None = None
    category: str | None = None
    template_name: str | None = None
    required_tags: frozenset[str] = field(default_factory=frozenset)

    def is_eligible(self, problem):
        if self.answer_format is not None and problem.answer_format != self.answer_format:
            return False

        if self.category is not None:
            primary_category = normalize_category(problem.metadata.get("primary_category"))
            if self.category != primary_category:
                return False

        if self.required_tags and not self.required_tags.issubset(problem.tags):
            return False

        return True


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    label: str
    routes: tuple[RouteDefinition, ...]
    default_route_name: str | None = None

    def select_route(self, problem):
        for route in self.routes:
            if route.is_eligible(problem):
                return route

        if self.default_route_name is not None:
            for route in self.routes:
                if route.name == self.default_route_name:
                    return route

        answer_format = problem.answer_format
        fallback_name = f"general_math_{answer_format}"

        for route in self.routes:
            if route.name == fallback_name:
                return route

        raise ValueError(f"No eligible route for strategy {self.name} and problem {problem.id}")


class StrategyRegistry:
    def __init__(self):
        self._strategies = {}

    def register(self, strategy):
        self._strategies[strategy.name] = strategy

    def get(self, name):
        return self._strategies[name]

    def names(self):
        return sorted(self._strategies)


def build_default_strategy_registry():
    registry = StrategyRegistry()

    for name, label in (
        ("baseline", "baseline_weakest"),
        ("baseline2", "baseline2_prompt_format"),
    ):
        registry.register(
            StrategyDefinition(
                name=name,
                label=label,
                routes=(
                    RouteDefinition(name="mcq", answer_format="mcq"),
                    RouteDefinition(name="free_form", answer_format="free_form"),
                ),
            )
        )

    baseline3_routes = []

    for category in BASELINE3_CATEGORIES:
        for answer_format in ("mcq", "free_form"):
            baseline3_routes.append(
                RouteDefinition(
                    name=f"{category}_{answer_format}",
                    answer_format=answer_format,
                    category=category,
                    template_name=f"baseline3_{category}_{answer_format}",
                )
            )

    baseline3_routes.extend((
        RouteDefinition(
            name="general_math_mcq",
            answer_format="mcq",
            category=None,
            template_name="baseline3_general_math_mcq",
        ),
        RouteDefinition(
            name="general_math_free_form",
            answer_format="free_form",
            category=None,
            template_name="baseline3_general_math_free_form",
        ),
    ))

    registry.register(
        StrategyDefinition(
            name="baseline3",
            label="baseline3_qwen_single_category_prompts",
            routes=tuple(baseline3_routes),
        )
    )

    def register_category_strategy(strategy_name, label, category):
        registry.register(
            StrategyDefinition(
                name=strategy_name,
                label=label,
                routes=(
                    RouteDefinition(
                        name=f"{category}_mcq",
                        answer_format="mcq",
                        category=category,
                        template_name=f"{strategy_name}_{category}_mcq",
                    ),
                    RouteDefinition(
                        name=f"{category}_free_form",
                        answer_format="free_form",
                        category=category,
                        template_name=f"{strategy_name}_{category}_free_form",
                    ),
                ),
            )
        )

    register_category_strategy(
        "calculus_v1_structured",
        "calculus_structured",
        "calculus",
    )

    register_category_strategy(
        "calculus_limit_asymptotic",
        "calculus_limit_asymptotic",
        "calculus",
    )

    register_category_strategy(
        "calculus_integral",
        "calculus_integral",
        "calculus",
    )

    register_category_strategy(
        "calculus_derivative_extrema",
        "calculus_derivative_extrema",
        "calculus",
    )

    register_category_strategy(
        "calculus_differential_equation",
        "calculus_differential_equation",
        "calculus",
    )

    register_category_strategy(
        "linear_algebra_v1_structured_verify",
        "linear_algebra_structured_verify",
        "linear_algebra",
    )

    register_category_strategy(
        "linear_algebra_mcq_option_verifier",
        "linear_algebra_mcq_option_verifier",
        "linear_algebra",
    )

    register_category_strategy(
        "linear_algebra_freeform_multi_answer",
        "linear_algebra_freeform_multi_answer",
        "linear_algebra",
    )

    register_category_strategy(
        "linear_algebra_lp_systems",
        "linear_algebra_lp_systems",
        "linear_algebra",
    )

    register_category_strategy(
        "discrete_algorithm_v1_subtype_router",
        "discrete_algorithm_subtype_router",
        "discrete_algorithm",
    )

    register_category_strategy(
        "discrete_sequence_option_verifier",
        "discrete_sequence_option_verifier",
        "discrete_algorithm",
    )

    register_category_strategy(
        "discrete_boolean_logic_v1",
        "discrete_boolean_logic_v1",
        "discrete_algorithm",
    )

    register_category_strategy(
        "discrete_counting_dp_v1",
        "discrete_counting_dp_v1",
        "discrete_algorithm",
    )

    register_category_strategy(
        "discrete_number_theory_v1",
        "discrete_number_theory_v1",
        "discrete_algorithm",
    )

    register_category_strategy(
        "general_math_v1_structured",
        "general_math_structured",
        "general_math",
    )

    register_category_strategy(
        "general_math_mcq_verifier",
        "general_math_mcq_verifier",
        "general_math",
    )

    register_category_strategy(
        "general_math_multi_answer",
        "general_math_multi_answer",
        "general_math",
    )

    register_category_strategy(
        "general_math_text_or_unit",
        "general_math_text_or_unit",
        "general_math",
    )

    registry.register(
        StrategyDefinition(
            name="baseline3_adaptive_rules",
            label="baseline3_adaptive_rules",
            routes=tuple(
                RouteDefinition(
                    name=f"{category}_{answer_format}",
                    answer_format=answer_format,
                    category=category,
                    template_name=f"baseline3_adaptive_rules_{category}_{answer_format}",
                )
                for category in BASELINE3_CATEGORIES
                for answer_format in ("mcq", "free_form")
            ),
        )
    )

    return registry
