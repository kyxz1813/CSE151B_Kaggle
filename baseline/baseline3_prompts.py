from .baseline2_prompts import BASELINE2_SYSTEM_PROMPT, build_baseline2_user_prompt
from baseline.rule_guidance import get_rule_guidance_text

CATEGORY_GUIDANCE = {
    "statistics_probability": (
        "Use statistical notation carefully. Identify hypotheses, parameters, test statistics, "
        "critical values, p-values, regression quantities, and probability assumptions before computing."
    ),
    "calculus": (
        "Identify the calculus object first: derivative, integral, limit, series, approximation, "
        "optimization, differential equation, or complex-analysis residue. State the exact operation "
        "and variables before computing. For limits, compare dominant terms or use expansions. "
        "For derivatives and extrema, differentiate before testing candidates. For integrals, choose "
        "substitution, parts, symmetry, standard forms, or residues as appropriate. Keep exact forms "
        "when possible and match the requested precision."
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
        "First identify whether the problem is actually arithmetic/algebra, applied units, geometry, "
        "statistics, calculus, linear algebra, or discrete math. If no specialized structure clearly "
        "fits, solve directly with concise arithmetic and consistency checks. Preserve requested "
        "answer order, units, rounding, and option-letter format."
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


CALCULUS_STRUCTURED_GUIDANCE = """
You are solving a calculus category problem.

Before computing, identify the subtype:
limit/asymptotic, derivative/tangent, integral/antiderivative, series/approximation, optimization/extrema, differential equation/model, or complex residue/contour.

Rules:
- For MCQ, solve first, compare the result to all choices, and box only the option letter.
- For limits, identify the dominant terms, equivalent forms, or expansion before substituting.
- For derivatives and tangent lines, keep track of the point of evaluation and requested variable.
- For integrals, state the method briefly: substitution, parts, symmetry, standard form, partial fractions, or residue.
- For extrema, check endpoints and critical points when the domain is bounded.
- For multiple [ANS] blanks, count the blanks and return answers in order inside one box.
"""


CALCULUS_LIMIT_ASYMPTOTIC_GUIDANCE = """
You are solving a calculus limit or asymptotic problem.

Use this method:
1. Identify the variable and limit point.
2. Simplify the expression before substitution.
3. Use dominant-term comparison, rationalization, logarithms, l'Hopital's rule, or Taylor expansion only when justified.
4. Preserve exact constants when possible.
5. For MCQ, map the computed limit to the answer letter.
"""


CALCULUS_INTEGRAL_GUIDANCE = """
You are solving a calculus integral problem.

Use this method:
1. Identify definite versus indefinite integral and the variable of integration.
2. Look for substitution, integration by parts, symmetry, standard antiderivatives, partial fractions, or contour/residue structure.
3. For definite integrals, apply bounds after finding the antiderivative or transformed bounds.
4. For improper or complex integrals, state convergence/residue conditions briefly.
5. For MCQ, compare the final expression/value to choices and box only the letter.
"""


CALCULUS_DERIVATIVE_EXTREMA_GUIDANCE = """
You are solving a derivative, tangent, rate, or extrema problem.

Use this method:
1. Define the function and variable.
2. Differentiate accurately before substituting values.
3. For tangent/linear approximation, compute both function value and derivative at the point.
4. For extrema, solve critical points and compare endpoints when relevant.
5. Return only the requested value, equation, or option letter.
"""


CALCULUS_DIFFERENTIAL_EQUATION_GUIDANCE = """
You are solving a differential equation or calculus model problem.

Use this method:
1. Identify the dependent variable, independent variable, and initial/boundary conditions.
2. Separate variables or use the standard model form when appropriate.
3. Solve constants from the given condition before answering.
4. Keep units and requested rounding consistent.
5. For MCQ, map the derived expression/value to the option letter.
"""


GENERAL_MATH_STRUCTURED_GUIDANCE = """
You are solving a general math fallback problem.

Before computing, identify whether another structure is hidden:
basic arithmetic/algebra, unit conversion, direct formula use, table interpretation, pattern recognition, or multi-part answer extraction.

Rules:
- Do not overcomplicate the problem with an unrelated advanced method.
- Count [ANS] blanks before solving and return the same number of answers in order.
- For MCQ, evaluate the answer choices and box exactly one capital letter.
- Preserve units, signs, percentages, and requested rounding.
- If the question asks for text, return the requested phrase exactly and without extra explanation in the final box.
"""


GENERAL_MATH_MCQ_VERIFIER_GUIDANCE = """
You are solving a general math multiple-choice problem.

Use this method:
1. Parse exactly what the question asks.
2. Compute or reason to the requested result.
3. Compare every answer choice to the result.
4. Reject choices with the right number but wrong interpretation, sign, unit, or wording.
5. The final answer must be exactly one option letter inside \\boxed{}.
"""


GENERAL_MATH_MULTI_ANSWER_GUIDANCE = """
You are solving a general math free-form problem with multiple requested blanks or parts.

Use this method:
1. Count all [ANS] placeholders.
2. Solve each part separately.
3. Preserve the order of the blanks.
4. Keep exact wording, units, and rounding requested by the prompt.
5. The final answer must contain exactly that many comma-separated entries inside one \\boxed{}.
"""


GENERAL_MATH_TEXT_OR_UNIT_GUIDANCE = """
You are solving a general math problem that may require text, units, or a direct formula.

Use this method:
1. Identify whether the answer is numeric, a phrase, a unit-bearing value, or a list.
2. For unit conversions, write the conversion factor and check direction.
3. For requested phrases, preserve wording and capitalization when the prompt specifies them.
4. Do not add explanation inside the final answer box.
"""

ARITHMETIC_ALGEBRA_GENERAL_GUIDANCE = """
You are solving an arithmetic or algebra problem.

Before solving, classify the problem into exactly one primary subtype:

- Numeric evaluation
- Symbolic simplification
- Equation solving
- Functional reasoning
- Sequence/table reasoning
- Set or interval reasoning
- Integer/divisibility reasoning

Use the solving strategy most appropriate for the identified subtype.

Rules:
- Preserve exact forms unless the problem explicitly requests approximation.
- Respect signs, fractions, radicals, logarithms, exponents, and algebraic structure carefully.
- For MCQ, solve independently before mapping to an answer choice.
- Count [ANS] placeholders and return exactly that many answers in order.
- Preserve symbolic expressions when appropriate; do not over-decimalize exact algebraic results.
- Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""


ARITHMETIC_ALGEBRA_SYMBOLIC_GUIDANCE = """
You are solving a symbolic arithmetic/algebra problem.

Use this method:
1. Preserve symbolic structure whenever possible.
2. Simplify expressions carefully and systematically only when specifically asked.
3. Track signs, exponents, radicals, fractions, logarithms, and parentheses exactly.
4. Avoid converting exact symbolic answers into decimals unless explicitly requested.
5. Reject extraneous solutions.
5. If solving equations, verify all candidate solutions within the original problem.
6. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""


ARITHMETIC_ALGEBRA_NUMERIC_GUIDANCE = """
You are solving a numerical arithmetic/algebra problem.

Use this method:
1. Apply order of operations (parenthesis, exponents, multiplcation/division, addition/subtraction) carefully.
2. Track arithmetic signs and parentheses exactly.
3. Compute intermediate values accurately.
4. Respect requested rounding precision. If none is requested, do not round.
5. For percentages, proportions, or unit conversions, preserve units and requested formatting.
6. Check the derived solution within the original problem to determine correctness.
7. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""


ARITHMETIC_ALGEBRA_MULTI_ANSWER_GUIDANCE = """
You are solving a multi-answer arithmetic/algebra problem.

Use this method:
1. Count the number of [ANS] placeholders.
2. Solve every requested component.
3. Preserve the requested answer order exactly.
4. For tables, sequences, or classifications, align answers to the matching row or column.
5. Return the same amount of answers as [ANS] placeholders.
6. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

ARITHMETIC_ALGEBRA_MCQ_GUIDANCE = """
You are solving an arithmetic/algebra multiple-choice problem.

Use this method:
1. Solve the problem independently.
2. Compute the exact mathematical result first.
3. Compare the computed result against every option.
4. Eliminate distractors carefully.
5. Return only the correct option letter inside \\boxed{}. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

APPLIED_WORD_PROBLEM_MODEL_BUILDING_GUIDANCE = """
You are solving an applied word problem.

Before computing:
1. Define variables clearly.
2. Identify units and constraints.
3. Translate the verbal description into equations, inequalities, functions, or expressions.

Rules:
- Preserve units carefully.
- Respect requested rounding instructions.
- Preserve exact forms unless approximation is requested.
- For multi-part problems, preserve answer order exactly.
- For modeling problems, ensure the mathematical model matches the real-world interpretation.
- Reject negative lengths, times, counts, probabilities, or populations unless the problem explicitly allows them.
- Verify that units are meaningful.
- Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

APPLIED_WORD_PROBLEM_MULTI_STEP_GUIDANCE = """
You are solving a multi-step applied problem.

Use this method:
1. Break the problem into explicit stages.
2. Track intermediate quantities carefully.
3. Preserve units throughout the computation.
4. Verify that each intermediate result is physically or contextually meaningful.
5. Return all requested answers in the correct order.
6. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

APPLIED_WORD_PROBLEM_PIECEWISE_GUIDANCE = """
You are solving a piecewise or conditional applied problem.

Use this method:
1. Identify each condition or threshold.
2. Write the correct expression/function for each case.
3. Verify interval boundaries carefully.
4. Preserve inequality direction and interval notation.
5. Return the final piecewise form exactly as requested.
6. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

APPLIED_WORD_PROBLEM_RATE_DISTANCE_GUIDANCE = """
You are solving a rate, distance, speed, tax, or proportional reasoning problem.

Use this method:
1. Define the relevant rates and quantities.
2. Remember and use distance = rate * time, work completed = rate * time, and unit-rate relationships when applicable.
3. Track units carefully.
4. Use dimensional consistency checks.
5. Convert units before combining quantities.
6. Verify the final answer matches the requested unit.x`
7. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
"""

APPLIED_WORD_PROBLEM_FINANCIAL_GUIDANCE = """
You are solving a financial or business modeling problem.

Use this method:
1. Identify fixed and variable quantities separately.
2. Define revenue, cost, profit, interest,
   balance, or tax relationships explicitly.
3. Preserve units and percentages carefully.
4. Verify whether the model is linear, exponential,
   proportional, or piecewise.
5. Respect requested rounding precision.
6. Express multiplication explicitly using asterisks. Do not round answers unless explicitly asked.
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


def build_calculus_structured_system_prompt():
    return _seed_system_prompt(
        "calculus_v1_structured",
        CALCULUS_STRUCTURED_GUIDANCE,
    )


def build_calculus_limit_asymptotic_system_prompt():
    return _seed_system_prompt(
        "calculus_limit_asymptotic",
        CALCULUS_LIMIT_ASYMPTOTIC_GUIDANCE,
    )


def build_calculus_integral_system_prompt():
    return _seed_system_prompt(
        "calculus_integral",
        CALCULUS_INTEGRAL_GUIDANCE,
    )


def build_calculus_derivative_extrema_system_prompt():
    return _seed_system_prompt(
        "calculus_derivative_extrema",
        CALCULUS_DERIVATIVE_EXTREMA_GUIDANCE,
    )


def build_calculus_differential_equation_system_prompt():
    return _seed_system_prompt(
        "calculus_differential_equation",
        CALCULUS_DIFFERENTIAL_EQUATION_GUIDANCE,
    )


def build_general_math_structured_system_prompt():
    return _seed_system_prompt(
        "general_math_v1_structured",
        GENERAL_MATH_STRUCTURED_GUIDANCE,
    )


def build_general_math_mcq_verifier_system_prompt():
    return _seed_system_prompt(
        "general_math_mcq_verifier",
        GENERAL_MATH_MCQ_VERIFIER_GUIDANCE,
    )


def build_general_math_multi_answer_system_prompt():
    return _seed_system_prompt(
        "general_math_multi_answer",
        GENERAL_MATH_MULTI_ANSWER_GUIDANCE,
    )


def build_general_math_text_or_unit_system_prompt():
    return _seed_system_prompt(
        "general_math_text_or_unit",
        GENERAL_MATH_TEXT_OR_UNIT_GUIDANCE,
    )


def build_arithmetic_algebra_general_system_prompt():
    return _seed_system_prompt(
        "arithmetic_algebra_v2_general",
        ARITHMETIC_ALGEBRA_GENERAL_GUIDANCE,
    )


def build_arithmetic_algebra_symbolic_system_prompt():
    return _seed_system_prompt(
        "arithmetic_algebra_symbolic",
        ARITHMETIC_ALGEBRA_SYMBOLIC_GUIDANCE,
    )


def build_arithmetic_algebra_numeric_system_prompt():
    return _seed_system_prompt(
        "arithmetic_algebra_numeric",
        ARITHMETIC_ALGEBRA_NUMERIC_GUIDANCE,
    )


def build_arithmetic_algebra_multi_answer_system_prompt():
    return _seed_system_prompt(
        "arithmetic_algebra_multi_answer",
        ARITHMETIC_ALGEBRA_MULTI_ANSWER_GUIDANCE,
    )


def build_arithmetic_algebra_mcq_system_prompt():
    return _seed_system_prompt(
        "arithmetic_algebra_mcq",
        ARITHMETIC_ALGEBRA_MCQ_GUIDANCE,
    )


def build_applied_word_problem_model_building_system_prompt():
    return _seed_system_prompt(
        "applied_word_problem_v2_model_building",
        APPLIED_WORD_PROBLEM_MODEL_BUILDING_GUIDANCE,
    )


def build_applied_word_problem_multi_step_system_prompt():
    return _seed_system_prompt(
        "applied_word_problem_multi_step",
        APPLIED_WORD_PROBLEM_MULTI_STEP_GUIDANCE,
    )


def build_applied_word_problem_piecewise_system_prompt():
    return _seed_system_prompt(
        "applied_word_problem_piecewise",
        APPLIED_WORD_PROBLEM_PIECEWISE_GUIDANCE,
    )


def build_applied_word_problem_rate_distance_system_prompt():
    return _seed_system_prompt(
        "applied_word_problem_rate_distance",
        APPLIED_WORD_PROBLEM_RATE_DISTANCE_GUIDANCE,
    )


def build_applied_word_problem_financial_system_prompt():
    return _seed_system_prompt(
        "applied_word_problem_financial",
        APPLIED_WORD_PROBLEM_FINANCIAL_GUIDANCE,
    )