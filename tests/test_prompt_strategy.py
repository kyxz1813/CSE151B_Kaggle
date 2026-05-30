import unittest

from baseline.category_tagging import normalize_category_tags
from baseline.category_work import calculus, general_math
from baseline.output_parsing import parse_model_output
from baseline.ved_category_analysis import build_failure_analysis_rows
from prompting.models import problem_from_record
from prompting.prompt_chain import build_prompt_chain
from prompting.strategies import build_default_strategy_registry


class PromptStrategyTests(unittest.TestCase):
    def test_strategy_registry_contains_prompt_chain_variants(self):
        registry = build_default_strategy_registry()
        names = registry.names()
        self.assertIn("baseline", names)
        self.assertIn("baseline2", names)
        self.assertIn("baseline3", names)
        self.assertIn("calculus_v1_structured", names)
        self.assertIn("general_math_v1_structured", names)

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
            '{"categories":["general_math","geometry_trig"],"primary_category":"calculus"}'
        )

        self.assertEqual(parsed["categories"], ["calculus"])
        self.assertEqual(parsed["primary_category"], "calculus")
        self.assertTrue(parsed["tag_parse_ok"])

    def test_ved_category_strategy_routes(self):
        calculus = problem_from_record({
            "id": 3,
            "question": "Evaluate the limit as x approaches 0.",
            "primary_category": "calculus",
        })
        general = problem_from_record({
            "id": 4,
            "question": "Which statement is true?",
            "options": ["A", "B", "C", "D"],
            "primary_category": "general_math",
        })

        calc_spec = build_prompt_chain("calculus_v1_structured").build_spec(calculus)
        gen_spec = build_prompt_chain("general_math_mcq_verifier").build_spec(general)

        self.assertEqual(calc_spec.name, "calculus_v1_structured_calculus_free_form")
        self.assertEqual(gen_spec.name, "general_math_mcq_verifier_general_math_mcq")

    def test_ved_category_work_registers(self):
        calc_info = calculus.register_all()
        gen_info = general_math.register_all()

        self.assertEqual(calc_info["category"], "calculus")
        self.assertEqual(gen_info["category"], "general_math")
        self.assertIn("calculus_v1_structured", calc_info["candidate_strategies"])
        self.assertIn("general_math_v1_structured", gen_info["candidate_strategies"])

    def test_ved_failure_analysis_classifies_calculus_integral(self):
        rows = build_failure_analysis_rows([{
            "id": 5,
            "question": "Evaluate the integral of x^2.",
            "category": "calculus",
            "gold": "1/3",
            "boxed_answer": "1/2",
            "correct": False,
            "extractable": True,
            "schema_valid": True,
            "strict_well_formed": True,
            "derived_rules": ["calculus_integral"],
        }])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["root_cause_classification"], "calculus_integral_error")

    def test_new_ved_rules_detect_target_patterns(self):
        endpoint_problem = {
            "question": "Find the absolute maximum of f(x) on the interval [0, 2].",
            "primary_category": "calculus",
        }
        percentage_problem = {
            "question": "A price increased by 20%. What is the final amount?",
            "primary_category": "general_math",
        }

        self.assertTrue(calculus.is_endpoint_extrema(endpoint_problem))
        self.assertTrue(general_math.is_percentage_or_rate(percentage_problem))

    def test_sanitize_prefers_last_complete_final_box_before_junk(self):
        output = (
            "Reasoning:\nFirst attempt.\n\n"
            "Final Answer: \\boxed{A}\n"
            "</think>\n"
            "Reasoning:\nSecond attempt.\n\n"
            "Final Answer: \\boxed{B}\n"
            "Reasoning:\nTrailing incomplete answer.\n\n"
            "Final Answer: \\boxed{"
        )

        parsed = parse_model_output(
            output,
            record={"question": "Pick one.", "options": ["a", "b", "c"]},
            sanitize=True,
        )

        self.assertEqual(parsed["boxed_answer"], "B")
        self.assertTrue(parsed["schema_valid"])
        self.assertNotIn("<think>", parsed["response_for_submission"])
        self.assertNotIn("</think>", parsed["response_for_submission"])

    def test_ved_prompts_include_strict_output_guard(self):
        spec = build_prompt_chain("calculus_v1_structured").build_spec(problem_from_record({
            "id": 6,
            "question": "Find the derivative of x^2.",
            "primary_category": "calculus",
        }))

        system_prompt = spec.to_messages()[0]["content"]

        self.assertIn("At most 4 concise lines", system_prompt)
        self.assertIn("Do not write <think> or </think>", system_prompt)
        self.assertIn("Stop immediately after Final Answer", system_prompt)


if __name__ == "__main__":
    unittest.main()
