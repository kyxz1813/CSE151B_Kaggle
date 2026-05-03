import json
import sys

from .models import problem_from_record
from .routing import Baseline3Router, FixedStrategyRouter
from .templates import build_default_registry


class RegistryBackedTemplateRenderer:
    def __init__(self, registry):
        self.registry = registry

    def render(self, context):
        template = self.registry.get(
            strategy_name=context.strategy_name,
            route_name=context.route_name,
        )
        return template.render(context)


class PromptChain:
    def __init__(self, router, renderer):
        self.router = router
        self.renderer = renderer

    def build_spec(self, problem):
        context = self.router.route(problem)
        return self.renderer.render(context)

    def build_messages(self, problem):
        return self.build_spec(problem).to_messages()


def build_prompt_chain(strategy_name="baseline"):
    registry = build_default_registry()
    if strategy_name == "baseline3":
        router = Baseline3Router()
    else:
        router = FixedStrategyRouter(strategy_name=strategy_name)
    renderer = RegistryBackedTemplateRenderer(registry)
    return PromptChain(router=router, renderer=renderer)

"""
In powershell:

MCQ:

$json = '{"id": 1, "question": "What is 2+2?", "options": ["1", "2", "4", "8"]}'
Set-Content sample_problem.json $json
python -m prompting.prompt_chain sample_problem.json

Free-form:

$json = '{"id": 2, "question": "Compute 3^2 + 4^2 = [ANS]"}'
Set-Content sample_problem.json $json
python -m prompting.prompt_chain sample_problem.json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m prompting.prompt_chain '<json_record_or_json_file>'")
        sys.exit(1)

    arg = sys.argv[1]

    if arg.endswith(".json"):
        with open(arg, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(arg)

    problem = problem_from_record(raw)
    chain = build_prompt_chain()
    spec = chain.build_spec(problem)

    print(json.dumps({
        "problem_id": problem.id,
        "template_name": spec.name,
        "metadata": spec.metadata,
        "generation_hints": spec.generation_hints,
        "messages": spec.to_messages(),
    }, indent=2))
"""
