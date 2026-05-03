from .models import PromptContext


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


CATEGORY_KEYWORDS = {
    "statistics_probability": (
        "hypothesis", "regression", "probability", "distribution", "confidence",
        "sample", "standard deviation", "variance", "mean", "median", "normal",
        "z score", "z-score", "t statistic", "p-value", "p value", "expected value",
        "random", "correlation", "anova", "significance", "null hypothesis",
    ),
    "calculus": (
        "integral", "integrate", "derivative", "differentiate", "limit",
        "antiderivative", "maximum", "minimum", "extrema", "critical point",
        "rate of change", "partial derivative", "\\int", "oint", "d/d",
    ),
    "geometry_trig": (
        "triangle", "circle", "circumcircle", "inradius", "angle", "area",
        "volume", "perimeter", "radius", "diameter", "arc", "polygon",
        "polyhedron", "rectangle", "altitude", "sine", "cosine", "tangent",
        "sin", "cos", "tan", "trig", "csc", "sec", "cot",
    ),
    "linear_algebra": (
        "matrix", "matrices", "vector", "rank", "linear transformation",
        "linearly independent", "basis", "eigen", "determinant", "system",
        "span", "null space", "row reduce",
    ),
    "discrete_algorithm": (
        "sequence", "recurrence", "combinatorics", "permutation", "combination",
        "partition", "modular", "modulo", "prime", "prime factor", "factorize", "integer",
        "algorithm", "x_list", "y_list", "number theory", "ways", "arrangement",
        "choose", "congruence",
    ),
    "arithmetic_algebra": (
        "simplify", "equation", "fraction", "polynomial", "quadratic",
        "linear", "slope", "intercept", "factor", "expression", "solve",
        "order of operations", "reduce",
    ),
    "applied_word_problem": (
        "temperature", "fahrenheit", "celsius", "kelvin", "rankine", "loan",
        "interest", "annuity", "principal", "dollars", "cost", "price",
        "miles", "hours", "minutes", "mixture", "population", "growth",
        "work together", "units",
    ),
}


def route_baseline3_category(problem):
    text_parts = [problem.question or ""]
    text_parts.extend(problem.options or [])
    text = " ".join(text_parts).lower()

    for category in BASELINE3_CATEGORIES:
        if category == "general_math":
            continue
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in text:
                return category

    return "general_math"


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


class Baseline3Router(RuleBasedProblemRouter):
    def __init__(self):
        super().__init__(default_strategy_name="baseline3")

    def route(self, problem):
        category = route_baseline3_category(problem)
        answer_format = problem.answer_format
        route_name = f"{category}_{answer_format}"
        tags = set(problem.tags)
        tags.add(category)

        return PromptContext(
            problem=problem,
            strategy_name=self.default_strategy_name,
            route_name=route_name,
            tags=tags,
            metadata={
                "answer_format": answer_format,
                "category": category,
            },
        )
