from .baseline2_prompts import BASELINE2_SYSTEM_PROMPT, build_baseline2_user_prompt


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
    return f"{BASELINE2_SYSTEM_PROMPT}\nCategory guidance:\n{guidance}\n"


def build_baseline3_user_prompt(context):
    return build_baseline2_user_prompt(context)
