from .models import PromptSpec
from baseline.baseline2_prompts import BASELINE2_SYSTEM_PROMPT, build_baseline2_user_prompt


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

        return PromptSpec(
            name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=self.user_builder(context),
            few_shot_messages=few_shot_messages,
            generation_hints=dict(self.generation_hints),
            metadata={
                "strategy_name": context.strategy_name,
                "route_name": context.route_name,
                "tags": sorted(context.tags),
            },
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

    return registry
