import unittest

from baseline.category_tagging import normalize_category_tags
from prompting.models import problem_from_record
from prompting.prompt_chain import build_prompt_chain
from prompting.strategies import build_default_strategy_registry


class PromptStrategyTests(unittest.TestCase):
    def test_strategy_registry_contains_prompt_chain_variants(self):
        registry = build_default_strategy_registry()
        self.assertEqual(registry.names(), ["baseline", "baseline2", "baseline3"])

    def test_baseline3_uses_qwen_category_tag_for_route(self):
        problem = problem_from_record({
            "id": 1,
            "question": "Compute the derivative of x^2.",
            "primary_category": "calculus",
            "qwen_categories": ["calculus"],
        })
        spec = build_prompt_chain("baseline3").build_spec(problem)

        self.assertEqual(spec.name, "baseline3_calculus_free_form")
        self.assertEqual(spec.metadata["category"], "calculus")

    def test_baseline3_keeps_answer_format_eligibility(self):
        problem = problem_from_record({
            "id": 2,
            "question": "Which matrix has rank 1?",
            "options": ["A", "B", "C", "D"],
            "primary_category": "linear_algebra",
            "qwen_categories": ["linear_algebra"],
        })
        spec = build_prompt_chain("baseline3").build_spec(problem)

        self.assertEqual(spec.name, "baseline3_linear_algebra_mcq")
        self.assertEqual(spec.metadata["answer_format"], "mcq")

    def test_category_tag_parser_normalizes_json(self):
        parsed = normalize_category_tags(
            '{"categories":["general_math","geometry_trig"],"primary_category":"general_math"}'
        )

        self.assertEqual(parsed["categories"], ["geometry_trig"])
        self.assertEqual(parsed["primary_category"], "geometry_trig")
        self.assertTrue(parsed["tag_parse_ok"])


if __name__ == "__main__":
    unittest.main()
