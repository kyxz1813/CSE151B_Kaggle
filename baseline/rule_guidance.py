from dataclasses import dataclass


@dataclass
class RuleGuidance:
    rule_name: str
    category: str
    title: str
    guidance: str
    priority: int = 100


class RuleGuidanceRegistry:
    def __init__(self):
        self._items = {}

    def register(self, item):
        self._items[item.rule_name] = item

    def get(self, rule_name):
        return self._items.get(rule_name)

    def guidance_for_rules(self, rule_names, category=None):
        items = []

        for rule_name in rule_names or []:
            item = self.get(rule_name)

            if item is None:
                continue

            if category is not None and item.category != category:
                continue

            items.append(item)

        items.sort(key=lambda item: item.priority)
        return items


RULE_GUIDANCE_REGISTRY = RuleGuidanceRegistry()


def register_rule_guidance(item):
    RULE_GUIDANCE_REGISTRY.register(item)


def get_rule_guidance_text(rule_names, category=None, registry=None):
    registry = registry or RULE_GUIDANCE_REGISTRY
    items = registry.guidance_for_rules(rule_names, category=category)

    if not items:
        return ""

    parts = []
    parts.append("Subtype-specific guidance based on detected derived rules:")

    for item in items:
        parts.append(f"- {item.title}: {item.guidance.strip()}")

    return "\n".join(parts)


def register_default_linear_discrete_guidance():
    items = [
        RuleGuidance(
            rule_name="linear_algebra_rank_parameter_condition",
            category="linear_algebra",
            title="Rank parameter condition",
            guidance=(
                "For rank/nullity/linear-dependence with a parameter, determinant = 0 is only "
                "a necessary condition. Solve the candidate values, then verify each candidate "
                "against the exact rank/nullity/independence requirement. Reject values that "
                "make the rank too small or too large."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="linear_algebra_determinant_or_matrix_property",
            category="linear_algebra",
            title="Matrix property",
            guidance=(
                "State the requested matrix property exactly. Check dimensions and distinguish "
                "determinant, rank, eigenvalue, inverse, and singularity. For MCQ, map the "
                "computed result to the option letter."
            ),
            priority=30,
        ),
        RuleGuidance(
            rule_name="linear_algebra_vector_projection",
            category="linear_algebra",
            title="Vector projection",
            guidance=(
                "Distinguish scalar projection, vector projection, signed projection, and "
                "components in a basis. Check whether the problem asks for a vector, scalar, "
                "length, or sum over possible vectors."
            ),
            priority=30,
        ),
        RuleGuidance(
            rule_name="linear_algebra_linear_programming_word_problem",
            category="linear_algebra",
            title="Linear programming word problem",
            guidance=(
                "Define variables, constraints, and objective. Do not assume variables are "
                "integers unless the problem explicitly says integer or whole number. For "
                "continuous variables, evaluate feasible-region vertices."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="linear_algebra_system_word_multi_answer",
            category="linear_algebra",
            title="System word problem with multiple answers",
            guidance=(
                "Count all [ANS] placeholders before solving. Return exactly that many answers "
                "in the same order. If the prompt asks for equations and then a value, include "
                "the equations and the value."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="linear_algebra_combinatorial_matrix_determinant",
            category="linear_algebra",
            title="Combinatorial matrix determinant",
            guidance=(
                "For subset/intersection matrices, reason about determinant values carefully. "
                "For MCQ, compute or infer the mathematical value, then select the matching "
                "option letter rather than boxing the raw number."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="linear_algebra_matrix_transformation_or_mapping",
            category="linear_algebra",
            title="Matrix transformation or mapping",
            guidance=(
                "Track input and output dimensions. Multiply matrices in the correct order and "
                "verify the resulting shape before comparing to options."
            ),
            priority=30,
        ),
        RuleGuidance(
            rule_name="linear_algebra_column_dependence_multi_part",
            category="linear_algebra",
            title="Column dependence multi-part problem",
            guidance=(
                "For each requested column subset or matrix case, answer separately and preserve "
                "the requested output order. Do not collapse a multi-part answer into one value."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="linear_algebra_mistag_xlist_ylist_sequence",
            category="linear_algebra",
            title="Likely sequence mistag",
            guidance=(
                "This looks like an x_list/y_list sequence problem. Treat it as list-option "
                "verification rather than ordinary matrix algebra."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="linear_algebra_mcq_option_mapping",
            category="linear_algebra",
            title="MCQ option mapping",
            guidance=(
                "Because this is multiple-choice, never put the raw computed value in the final "
                "box. Compare the computed result to the choices and box only the final letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="linear_algebra_freeform_multi_answer",
            category="linear_algebra",
            title="Free-form multi-answer",
            guidance=(
                "Because this is free-form with multiple blanks, final answer must contain the "
                "same number of comma-separated entries as [ANS] placeholders."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="discrete_xlist_ylist_sequence",
            category="discrete_algorithm",
            title="x_list/y_list sequence",
            guidance=(
                "Parse x_list and treat answer choices as candidate y_lists. Verify the selected "
                "option term-by-term. Do not choose based on vague pattern matching."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="discrete_named_sequence",
            category="discrete_algorithm",
            title="Named sequence",
            guidance=(
                "Do not invent a recurrence or rely on unsupported named-sequence memory. Use "
                "only the definition and answer choices. If the definition is not fully usable, "
                "compare options conservatively and verify all terms you can."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="discrete_computable_definition_sequence",
            category="discrete_algorithm",
            title="Computable sequence definition",
            guidance=(
                "If the problem gives an exact rule, compute y for every x in x_list and compare "
                "against each candidate option."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="discrete_boolean_logic_calculus",
            category="discrete_algorithm",
            title="Boolean/logical calculus",
            guidance=(
                "Do not assume ordinary arithmetic. In logical calculus, 1 may mean True, 0 may "
                "mean False, + may mean OR, and multiplication/dot may mean AND. Evaluate options "
                "under the logical interpretation."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="discrete_counting_dp",
            category="discrete_algorithm",
            title="Counting or dynamic programming",
            guidance=(
                "Define the state, base cases, recurrence/case split, and boundary conditions. "
                "For MCQ, map the final count to the correct option letter."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="discrete_graph_grid_walk",
            category="discrete_algorithm",
            title="Graph or grid walk",
            guidance=(
                "Model positions/states explicitly. Use recurrence or symmetry carefully and "
                "avoid unsupported periodicity claims."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="discrete_number_theory",
            category="discrete_algorithm",
            title="Number theory",
            guidance=(
                "Use the exact definition. Factor or reduce modulo only when relevant. For "
                "x_list/y_list number theory sequences, compute each listed input independently "
                "when possible."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="discrete_numeric_root_mistag",
            category="discrete_algorithm",
            title="Numeric root mistag",
            guidance=(
                "This is a numerical equation-solving problem, not really discrete. Find all "
                "solutions and output as much precision as possible, even if the problem says "
                "two decimal places, because the judge may compare against high-precision gold."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="discrete_set_family_extremal",
            category="discrete_algorithm",
            title="Set family extremal problem",
            guidance=(
                "Translate the condition on sets carefully. Check small examples and avoid "
                "boxing a raw number in MCQ unless it is mapped to the option letter."
            ),
            priority=20,
        ),
        RuleGuidance(
            rule_name="discrete_mcq_option_mapping",
            category="discrete_algorithm",
            title="MCQ option mapping",
            guidance=(
                "Because this is multiple-choice, compare the final result to the choices and "
                "box only the letter. Never box the raw count/value."
            ),
            priority=1,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items