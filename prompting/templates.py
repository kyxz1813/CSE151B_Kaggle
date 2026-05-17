from .models import PromptSpec
from baseline.baseline2_prompts import BASELINE2_SYSTEM_PROMPT, build_baseline2_user_prompt
from baseline.baseline3_prompts import (
    CATEGORY_GUIDANCE,
    build_baseline3_system_prompt,
    build_baseline3_user_prompt,
    build_linear_algebra_structured_verify_system_prompt,
    build_linear_algebra_mcq_option_verifier_system_prompt,
    build_linear_algebra_freeform_multi_answer_system_prompt,
    build_linear_algebra_lp_systems_system_prompt,
    build_discrete_subtype_router_system_prompt,
    build_discrete_sequence_option_verifier_system_prompt,
    build_discrete_boolean_logic_system_prompt,
    build_discrete_counting_dp_system_prompt,
    build_discrete_number_theory_system_prompt,
    build_calculus_structured_system_prompt,
    build_calculus_limit_asymptotic_system_prompt,
    build_calculus_integral_system_prompt,
    build_calculus_derivative_extrema_system_prompt,
    build_calculus_differential_equation_system_prompt,
    build_general_math_structured_system_prompt,
    build_general_math_mcq_verifier_system_prompt,
    build_general_math_multi_answer_system_prompt,
    build_general_math_text_or_unit_system_prompt,
    build_adaptive_rule_user_prompt,
)


FREE_FORM_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

MCQ_SYSTEM_PROMPT = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output only the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


class PromptTemplate:
    def __init__(
        self,
        name,
        system_prompt,
        user_builder,
        few_shot_builder=None,
        generation_hints=None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.user_builder = user_builder
        self.few_shot_builder = few_shot_builder
        self.generation_hints = generation_hints or {}

    def render(self, context):
        few_shot_messages = []
        if self.few_shot_builder is not None:
            few_shot_messages = self.few_shot_builder(context)

        metadata = dict(context.metadata)
        metadata.update({
            "strategy_name": context.strategy_name,
            "route_name": context.route_name,
            "tags": sorted(context.tags),
        })

        return PromptSpec(
            name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=self.user_builder(context),
            few_shot_messages=few_shot_messages,
            generation_hints=dict(self.generation_hints),
            metadata=metadata,
        )


def build_user_prompt_free_form(context):
    return context.problem.question.strip()


def build_user_prompt_mcq(context):
    problem = context.problem
    options = problem.options or []
    option_lines = []

    for idx, opt in enumerate(options):
        letter = chr(ord("A") + idx)
        option_lines.append(f"{letter}. {opt}")

    option_block = "\n".join(option_lines)

    return (
        f"{problem.question.strip()}\n\n"
        f"Answer choices:\n{option_block}\n\n"
        "Remember: output only the final answer letter inside \\boxed{}."
    )


class PromptTemplateRegistry:
    def __init__(self):
        self._templates = {}

    def register(self, strategy_name, route_name, template):
        self._templates.setdefault(strategy_name, {})[route_name] = template

    def get(self, strategy_name, route_name):
        return self._templates[strategy_name][route_name]


def build_default_registry():
    registry = PromptTemplateRegistry()

    registry.register(
        "baseline",
        "free_form",
        PromptTemplate(
            name="baseline_free_form",
            system_prompt=FREE_FORM_SYSTEM_PROMPT,
            user_builder=build_user_prompt_free_form,
            generation_hints={
                "temperature": 0.6,
                "top_p": 0.95,
            },
        ),
    )

    registry.register(
        "baseline",
        "mcq",
        PromptTemplate(
            name="baseline_mcq",
            system_prompt=MCQ_SYSTEM_PROMPT,
            user_builder=build_user_prompt_mcq,
            generation_hints={
                "temperature": 0.6,
                "top_p": 0.95,
            },
        ),
    )

    registry.register(
        "baseline2",
        "free_form",
        PromptTemplate(
            name="baseline2_free_form",
            system_prompt=BASELINE2_SYSTEM_PROMPT,
            user_builder=build_baseline2_user_prompt,
            generation_hints={
                "temperature": 0.6,
                "top_p": 0.95,
            },
        ),
    )

    registry.register(
        "baseline2",
        "mcq",
        PromptTemplate(
            name="baseline2_mcq",
            system_prompt=BASELINE2_SYSTEM_PROMPT,
            user_builder=build_baseline2_user_prompt,
            generation_hints={
                "temperature": 0.6,
                "top_p": 0.95,
            },
        ),
    )

    for category in CATEGORY_GUIDANCE:
        for answer_format in ("free_form", "mcq"):
            registry.register(
                "baseline3",
                f"{category}_{answer_format}",
                PromptTemplate(
                    name=f"baseline3_{category}_{answer_format}",
                    system_prompt=build_baseline3_system_prompt(category),
                    user_builder=build_baseline3_user_prompt,
                    generation_hints={
                        "temperature": 0.6,
                        "top_p": 0.95,
                    },
                ),
            )

    def register_category_strategy_templates(strategy_name, category, mcq_system_prompt, free_form_system_prompt):
        registry.register(
            strategy_name,
            f"{category}_mcq",
            PromptTemplate(
                name=f"{strategy_name}_{category}_mcq",
                system_prompt=mcq_system_prompt,
                user_builder=build_baseline3_user_prompt,
                generation_hints={
                    "temperature": 0.6,
                    "top_p": 0.95,
                },
            ),
        )

        registry.register(
            strategy_name,
            f"{category}_free_form",
            PromptTemplate(
                name=f"{strategy_name}_{category}_free_form",
                system_prompt=free_form_system_prompt,
                user_builder=build_baseline3_user_prompt,
                generation_hints={
                    "temperature": 0.6,
                    "top_p": 0.95,
                },
            ),
        )

    register_category_strategy_templates(
        "calculus_v1_structured",
        "calculus",
        build_calculus_structured_system_prompt(),
        build_calculus_structured_system_prompt(),
    )

    register_category_strategy_templates(
        "calculus_limit_asymptotic",
        "calculus",
        build_calculus_limit_asymptotic_system_prompt(),
        build_calculus_limit_asymptotic_system_prompt(),
    )

    register_category_strategy_templates(
        "calculus_integral",
        "calculus",
        build_calculus_integral_system_prompt(),
        build_calculus_integral_system_prompt(),
    )

    register_category_strategy_templates(
        "calculus_derivative_extrema",
        "calculus",
        build_calculus_derivative_extrema_system_prompt(),
        build_calculus_derivative_extrema_system_prompt(),
    )

    register_category_strategy_templates(
        "calculus_differential_equation",
        "calculus",
        build_calculus_differential_equation_system_prompt(),
        build_calculus_differential_equation_system_prompt(),
    )

    register_category_strategy_templates(
        "linear_algebra_v1_structured_verify",
        "linear_algebra",
        build_linear_algebra_structured_verify_system_prompt(),
        build_linear_algebra_structured_verify_system_prompt(),
    )

    register_category_strategy_templates(
        "linear_algebra_mcq_option_verifier",
        "linear_algebra",
        build_linear_algebra_mcq_option_verifier_system_prompt(),
        build_linear_algebra_structured_verify_system_prompt(),
    )

    register_category_strategy_templates(
        "linear_algebra_freeform_multi_answer",
        "linear_algebra",
        build_linear_algebra_mcq_option_verifier_system_prompt(),
        build_linear_algebra_freeform_multi_answer_system_prompt(),
    )

    register_category_strategy_templates(
        "linear_algebra_lp_systems",
        "linear_algebra",
        build_linear_algebra_mcq_option_verifier_system_prompt(),
        build_linear_algebra_lp_systems_system_prompt(),
    )

    register_category_strategy_templates(
        "discrete_algorithm_v1_subtype_router",
        "discrete_algorithm",
        build_discrete_subtype_router_system_prompt(),
        build_discrete_subtype_router_system_prompt(),
    )

    register_category_strategy_templates(
        "discrete_sequence_option_verifier",
        "discrete_algorithm",
        build_discrete_sequence_option_verifier_system_prompt(),
        build_discrete_subtype_router_system_prompt(),
    )

    register_category_strategy_templates(
        "discrete_boolean_logic_v1",
        "discrete_algorithm",
        build_discrete_boolean_logic_system_prompt(),
        build_discrete_boolean_logic_system_prompt(),
    )

    register_category_strategy_templates(
        "discrete_counting_dp_v1",
        "discrete_algorithm",
        build_discrete_counting_dp_system_prompt(),
        build_discrete_counting_dp_system_prompt(),
    )

    register_category_strategy_templates(
        "discrete_number_theory_v1",
        "discrete_algorithm",
        build_discrete_number_theory_system_prompt(),
        build_discrete_number_theory_system_prompt(),
    )

    register_category_strategy_templates(
        "general_math_v1_structured",
        "general_math",
        build_general_math_structured_system_prompt(),
        build_general_math_structured_system_prompt(),
    )

    register_category_strategy_templates(
        "general_math_mcq_verifier",
        "general_math",
        build_general_math_mcq_verifier_system_prompt(),
        build_general_math_structured_system_prompt(),
    )

    register_category_strategy_templates(
        "general_math_multi_answer",
        "general_math",
        build_general_math_structured_system_prompt(),
        build_general_math_multi_answer_system_prompt(),
    )

    register_category_strategy_templates(
        "general_math_text_or_unit",
        "general_math",
        build_general_math_mcq_verifier_system_prompt(),
        build_general_math_text_or_unit_system_prompt(),
    )

    for category in CATEGORY_GUIDANCE:
        for route_name in [f"{category}_mcq", f"{category}_free_form"]:
            registry.register(
                "baseline3_adaptive_rules",
                route_name,
                PromptTemplate(
                    name=f"baseline3_adaptive_rules_{route_name}",
                    system_prompt=build_baseline3_system_prompt(category),
                    user_builder=build_adaptive_rule_user_prompt,
                    generation_hints={
                        "temperature": 0.6,
                        "top_p": 0.95,
                    },
                ),
            )

    return registry
