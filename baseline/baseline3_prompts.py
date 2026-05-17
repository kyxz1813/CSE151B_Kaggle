from .baseline2_prompts import BASELINE2_SYSTEM_PROMPT, build_baseline2_user_prompt
from baseline.rule_guidance import get_rule_guidance_text

CATEGORY_GUIDANCE = {
    "statistics_probability": (
        "Use statistical notation carefully. Identify hypotheses, parameters, test statistics, "
        "critical values, p-values, regression quantities, and probability assumptions before computing."
    ),
    "calculus": (
        "Identify the relevant calculus operation first. For integrals, choose substitutions, symmetry, "
        "series, or standard antiderivatives as appropriate. For derivatives or extrema, state the function "
        "and conditions before simplifying."
    ),
    "geometry_trig": (
        "Track geometric definitions, diagrams implied by the text, and trigonometric identities. "
        "Use exact values and simplify radicals or ratios when possible."
    ),
    "linear_algebra": (
        "Translate the problem into matrices, vectors, ranks, bases, systems, or transformations. "
        "Keep dimensions and variable order consistent."
    ),
    "discrete_algorithm": (
        "Look for recurrence, counting, modular arithmetic, sequence, or algorithm structure. "
        "Use exact integer reasoning and verify list-style answers against the requested order."
    ),
    "arithmetic_algebra": (
        "Simplify expressions step by step. Respect order of operations, signs, fractions, "
        "equation solving, and algebraic simplification."
    ),
    "applied_word_problem": (
        "Define variables and units before computing. Preserve requested units, rounding, and "
        "multi-part answer order."
    ),
    "general_math": (
        "Identify the mathematical structure, solve directly, and keep the final answer concise."
    ),
}


def build_baseline3_system_prompt(category):
    guidance = CATEGORY_GUIDANCE.get(category, CATEGORY_GUIDANCE["general_math"])
    return (
        f"{BASELINE2_SYSTEM_PROMPT}\n"
        "Baseline 3 routing note:\n"
        f"- This problem has been assigned exactly one category: {category}.\n"
        "- Use the category guidance only to choose the solving style; do not mention or debate the category.\n\n"
        f"Category guidance:\n{guidance}\n"
    )


def build_baseline3_user_prompt(context):
    return build_baseline2_user_prompt(context)

def build_adaptive_rule_user_prompt(context):
    base_prompt = build_baseline3_user_prompt(context)

    category = context.metadata.get("category") or "general_math"
    derived_rules = context.metadata.get("derived_rules") or []

    guidance = get_rule_guidance_text(
        rule_names=derived_rules,
        category=category,
    )

    if not guidance:
        return base_prompt

    return (
        f"{guidance}\n\n"
        f"Use the subtype-specific guidance above only when relevant. "
        f"The final answer format requirements still take priority.\n\n"
        f"{base_prompt}"
    )

def _seed_system_prompt(title, guidance):
    return (
        f"{BASELINE2_SYSTEM_PROMPT}\n\n"
        f"Specialized strategy: {title}\n"
        f"{guidance.strip()}\n"
    )



LINEAR_ALGEBRA_STRUCTURED_VERIFY_GUIDANCE = """
You are solving a linear algebra category problem.

Before computing, identify the subtype:
rank/nullity, determinant, matrix property, vector projection, system of equations, linear programming, transformation/mapping, combinatorial matrix, or column dependence.

Rules:
- For MCQ, compute the mathematical result first, compare it to every option, then output only the matching letter in the final box.
- For rank/nullity/linear-dependence with a parameter, solving determinant = 0 is only a necessary condition. Verify every candidate against the exact rank/nullity/independence requirement.
- For optimization with linear constraints, do not assume variables are integers unless the problem explicitly says integer, whole number, or nonnegative integer.
- For systems or word problems with multiple [ANS] blanks, count the blanks and return exactly that many answers in order inside one box.
- For projection/vector problems, distinguish scalar projection, vector projection, unit direction, and raw vector components.
- If the problem is actually an x_list/y_list algorithm sequence, treat it as a list-option verification problem.
"""


LINEAR_ALGEBRA_MCQ_OPTION_VERIFIER_GUIDANCE = """
You are solving a linear algebra multiple-choice problem.

Use this method:
1. Identify the exact object being requested: scalar, vector, matrix, parameter value, determinant, rank, or option.
2. Solve independently.
3. Check each answer choice against the computed result.
4. Reject choices that are only necessary conditions but not sufficient.
5. If the computed result is numeric or symbolic, map it to the corresponding choice.
6. The final answer must be exactly one option letter inside \\boxed{}.

Never put the raw numeric/matrix result in the final box for MCQ.
"""


LINEAR_ALGEBRA_FREEFORM_MULTI_ANSWER_GUIDANCE = """
You are solving a free-form linear algebra or system problem.

Use this method:
1. Count the number of [ANS] placeholders.
2. Solve every requested part, not just the final numerical part.
3. Preserve the exact order of the blanks.
4. If asked for equations, include the equations as answers.
5. If asked for values, include the values with enough precision.
6. The final answer must contain exactly the same number of comma-separated entries as [ANS] placeholders.
"""


LINEAR_ALGEBRA_LP_SYSTEMS_GUIDANCE = """
You are solving a linear constraints or linear programming word problem.

Use this method:
1. Define variables.
2. Write all constraints and the objective.
3. Unless the problem explicitly says integer/whole-number, treat variables as continuous.
4. Find feasible-region vertices by intersecting constraint lines.
5. Evaluate the objective at each feasible vertex.
6. Return the requested variables and objective value in the requested order.
"""


DISCRETE_SUBTYPE_ROUTER_GUIDANCE = """
You are solving a discrete algorithm category problem.

Before computing, identify the subtype:
x_list/y_list sequence, named sequence, counting/DP, recurrence, modular/number theory, graph/grid walk, Boolean/logical calculus, set family/extremal, or numeric root mistag.

Rules:
- For MCQ, compute or verify the answer, compare against options, and output only the letter in the final box.
- For x_list/y_list problems, parse x_list, parse the candidate y_list options, and verify the chosen option term-by-term.
- Do not invent a recurrence or rely on vague named-sequence memory unless the problem gives a computable definition.
- For logical calculus or Boolean-looking 0/1 problems, do not assume ordinary arithmetic. Consider Boolean algebra: + can mean OR, · can mean AND, and 1+1 can equal 1.
- For numerical root problems, output as much precision as possible, even if the text says two decimal places.
"""


DISCRETE_SEQUENCE_OPTION_VERIFIER_GUIDANCE = """
You are solving a discrete x_list/y_list sequence multiple-choice problem.

Use this method:
1. Parse the input x_list.
2. Parse every answer choice as a candidate y_list.
3. If the problem gives an exact definition, compute y for each x.
4. If the definition is hard, compare candidates using only justified constraints from the problem.
5. Verify the selected option term-by-term.
6. Do not choose based on "closest", "probably", "known sequence", or unsupported memory.
7. The final answer must be exactly one option letter inside \\boxed{}.
"""


DISCRETE_BOOLEAN_LOGIC_GUIDANCE = """
You are solving a discrete logical calculus or Boolean algebra problem.

Use this method:
1. Do not assume ordinary arithmetic if the problem says logical calculus, Boolean algebra, logic, or uses 0/1 symbolic operations.
2. Consider Boolean interpretations:
   - 1 = True
   - 0 = False
   - + can mean OR
   - · or * can mean AND
3. Evaluate each option under the relevant logical calculus.
4. The final answer must be exactly one option letter inside \\boxed{}.
"""


DISCRETE_COUNTING_DP_GUIDANCE = """
You are solving a discrete counting or dynamic-programming problem.

Use this method:
1. Define the state clearly.
2. Give the base case.
3. Give the recurrence or case split.
4. Apply boundary conditions carefully.
5. Sum or select the requested quantity.
6. For MCQ, map the final numeric result to the correct option letter.
"""


DISCRETE_NUMBER_THEORY_GUIDANCE = """
You are solving a discrete number theory problem.

Use this method:
1. Apply the exact definition in the problem.
2. Factor numbers or reduce modulo only when relevant.
3. Avoid vague prime/modular heuristics.
4. For x_list/y_list number theory sequences, compute each listed input independently when possible.
5. For MCQ, map the result to the correct option letter.
"""


def build_linear_algebra_structured_verify_system_prompt():
    return _seed_system_prompt(
        "linear_algebra_v1_structured_verify",
        LINEAR_ALGEBRA_STRUCTURED_VERIFY_GUIDANCE,
    )


def build_linear_algebra_mcq_option_verifier_system_prompt():
    return _seed_system_prompt(
        "linear_algebra_mcq_option_verifier",
        LINEAR_ALGEBRA_MCQ_OPTION_VERIFIER_GUIDANCE,
    )


def build_linear_algebra_freeform_multi_answer_system_prompt():
    return _seed_system_prompt(
        "linear_algebra_freeform_multi_answer",
        LINEAR_ALGEBRA_FREEFORM_MULTI_ANSWER_GUIDANCE,
    )


def build_linear_algebra_lp_systems_system_prompt():
    return _seed_system_prompt(
        "linear_algebra_lp_systems",
        LINEAR_ALGEBRA_LP_SYSTEMS_GUIDANCE,
    )


def build_discrete_subtype_router_system_prompt():
    return _seed_system_prompt(
        "discrete_algorithm_v1_subtype_router",
        DISCRETE_SUBTYPE_ROUTER_GUIDANCE,
    )


def build_discrete_sequence_option_verifier_system_prompt():
    return _seed_system_prompt(
        "discrete_sequence_option_verifier",
        DISCRETE_SEQUENCE_OPTION_VERIFIER_GUIDANCE,
    )


def build_discrete_boolean_logic_system_prompt():
    return _seed_system_prompt(
        "discrete_boolean_logic_v1",
        DISCRETE_BOOLEAN_LOGIC_GUIDANCE,
    )


def build_discrete_counting_dp_system_prompt():
    return _seed_system_prompt(
        "discrete_counting_dp_v1",
        DISCRETE_COUNTING_DP_GUIDANCE,
    )


def build_discrete_number_theory_system_prompt():
    return _seed_system_prompt(
        "discrete_number_theory_v1",
        DISCRETE_NUMBER_THEORY_GUIDANCE,
    )