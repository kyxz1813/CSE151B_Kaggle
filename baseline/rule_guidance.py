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


def register_default_ved_guidance():
    items = [
        RuleGuidance(
            rule_name="calculus_limit_asymptotic",
            category="calculus",
            title="Limit or asymptotic problem",
            guidance=(
                "Identify the limit variable and point first. Simplify before substitution, "
                "then use dominant terms, rationalization, l'Hopital's rule, or Taylor expansion "
                "only when justified."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="calculus_integral",
            category="calculus",
            title="Integral problem",
            guidance=(
                "Identify the variable and definite/indefinite form. Choose substitution, parts, "
                "symmetry, partial fractions, standard antiderivatives, or residues as appropriate. "
                "Apply bounds after transforming correctly."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="calculus_derivative_extrema",
            category="calculus",
            title="Derivative or extrema problem",
            guidance=(
                "Differentiate accurately before substituting. For tangent or approximation, compute "
                "both function value and derivative at the point. For extrema on bounded domains, "
                "compare critical points and endpoints."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="calculus_endpoint_extrema_check",
            category="calculus",
            title="Endpoint extrema check",
            guidance=(
                "For absolute extrema on a closed interval, evaluate every critical point and both "
                "endpoints before choosing the requested maximum or minimum."
            ),
            priority=3,
        ),
        RuleGuidance(
            rule_name="calculus_series_approximation",
            category="calculus",
            title="Series or approximation",
            guidance=(
                "Use the requested expansion center and degree. Keep enough terms to answer the "
                "question and avoid evaluating at the wrong point."
            ),
            priority=15,
        ),
        RuleGuidance(
            rule_name="calculus_differential_equation",
            category="calculus",
            title="Differential equation",
            guidance=(
                "Identify variables and initial conditions. Solve the general form, then determine "
                "constants before answering with the requested units or value."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="calculus_initial_condition_check",
            category="calculus",
            title="Initial condition check",
            guidance=(
                "After solving the general form, use every given initial or boundary condition to "
                "determine constants, then substitute back once to verify."
            ),
            priority=3,
        ),
        RuleGuidance(
            rule_name="calculus_complex_residue",
            category="calculus",
            title="Complex residue or contour",
            guidance=(
                "Find the relevant poles and residues carefully. Check whether the contour includes "
                "each pole before summing residues."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="calculus_mcq_option_mapping",
            category="calculus",
            title="MCQ option mapping",
            guidance=(
                "Because this is multiple-choice, compare the calculus result to every answer "
                "choice and box only the final option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_mcq_option_mapping",
            category="general_math",
            title="MCQ option mapping",
            guidance=(
                "Compute the requested value or statement, compare every choice, and box only the "
                "single option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_multi_answer",
            category="general_math",
            title="Multi-answer fallback",
            guidance=(
                "Count all [ANS] placeholders before solving. Return exactly that many comma-separated "
                "answers in the same order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_text_answer",
            category="general_math",
            title="Text answer",
            guidance=(
                "If the prompt requests a phrase or word, preserve the requested wording and do not "
                "add explanation inside the final answer box."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="general_math_unit_or_conversion",
            category="general_math",
            title="Unit or conversion check",
            guidance=(
                "Track units and conversion direction. Include the requested units or rounding only "
                "when the final answer format asks for them."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="general_math_percentage_or_rate",
            category="general_math",
            title="Percentage or rate check",
            guidance=(
                "Verify whether the prompt asks for the percent, decimal rate, amount of change, "
                "or final amount. Keep the requested form in the final answer."
            ),
            priority=4,
        ),
        RuleGuidance(
            rule_name="general_math_direct_formula",
            category="general_math",
            title="Direct formula use",
            guidance=(
                "Identify the formula variables, substitute values carefully, and return only the "
                "requested unknown."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="general_math_ordered_answer",
            category="general_math",
            title="Ordered answer check",
            guidance=(
                "When an ordered list, ordered pair, or multiple blanks are requested, keep answers "
                "in the exact order requested and use one comma-separated final box."
            ),
            priority=4,
        ),
        RuleGuidance(
            rule_name="general_math_arithmetic_simplification",
            category="general_math",
            title="Arithmetic simplification check",
            guidance=(
                "Recompute the final arithmetic, simplify fractions when possible, and apply rounding "
                "only if the prompt requests it."
            ),
            priority=4,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items

def register_default_remaining_category_guidance():
    items = [
        RuleGuidance(
            rule_name="arithmetic_algebra_numeric_evaluation",
            category="arithmetic_algebra",
            title="Numeric evaluation",
            guidance=(
                "Compute carefully with order of operations, signs, fractions, exponents, and radicals. "
                "Keep exact form when useful and apply rounding only if requested."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_symbolic_manipulation",
            category="arithmetic_algebra",
            title="Symbolic manipulation",
            guidance=(
                "Simplify, factor, expand, reduce, or rewrite algebraically step by step. Preserve "
                "equivalent exact expressions and avoid unnecessary decimal approximations."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_equation_solving",
            category="arithmetic_algebra",
            title="Equation solving",
            guidance=(
                "Solve the equation systematically. Check for extraneous roots when squaring, using logs, "
                "or manipulating rational expressions. Return all requested solutions."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_function_or_relation",
            category="arithmetic_algebra",
            title="Function or relation",
            guidance=(
                "Track the function definition, input value, inverse/composition order, and requested "
                "output. Substitute only after identifying the correct expression."
            ),
            priority=15,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_discrete_integer",
            category="arithmetic_algebra",
            title="Integer/divisibility reasoning",
            guidance=(
                "Use integer constraints, divisibility, factorization, parity, gcd/lcm, or remainders "
                "explicitly. Check that the final answer satisfies the stated integer conditions."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_conversion_representation",
            category="arithmetic_algebra",
            title="Conversion or representation",
            guidance=(
                "Track the requested representation: fraction, decimal, percent, logarithmic form, or unit. "
                "Convert in the correct direction and preserve requested precision."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_interval_or_set",
            category="arithmetic_algebra",
            title="Interval or set operation",
            guidance=(
                "For unions, intersections, intervals, and sets, determine membership conditions first, "
                "then express the final answer in the requested notation."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_mcq",
            category="arithmetic_algebra",
            title="MCQ option mapping",
            guidance=(
                "Because this is multiple-choice, compute the result, compare it to every option, and box "
                "only the final option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_multi_answer",
            category="arithmetic_algebra",
            title="Multi-answer arithmetic/algebra",
            guidance=(
                "Count all [ANS] placeholders before solving. Return exactly that many comma-separated "
                "answers in the same order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_table_or_sequence",
            category="arithmetic_algebra",
            title="Table or sequence",
            guidance=(
                "Infer the rule from the given table or sequence only after checking multiple entries. "
                "Preserve order and output all requested values."
            ),
            priority=10,
        ),

        RuleGuidance(
            rule_name="applied_word_problem_rate_ratio_model",
            category="applied_word_problem",
            title="Rate or ratio model",
            guidance=(
                "Define variables and units. Identify whether the rate is per unit, percent, speed, tax, "
                "cost, or density. Multiply/divide in the direction implied by the units."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_linear_modeling",
            category="applied_word_problem",
            title="Linear modeling",
            guidance=(
                "Build the linear model explicitly: define variables, slope/rate, intercept/fixed cost, "
                "and the requested unknown. Check units before finalizing."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_exponential_modeling",
            category="applied_word_problem",
            title="Exponential modeling",
            guidance=(
                "Identify initial value, growth/decay factor, time units, and whether the model is "
                "continuous or discrete. Solve constants before evaluating."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_geometry_application",
            category="applied_word_problem",
            title="Geometry application",
            guidance=(
                "Translate the story into the correct geometric formula. Track radius/diameter, area, "
                "perimeter, volume, and unit conversions carefully."
            ),
            priority=15,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_function_interpretation",
            category="applied_word_problem",
            title="Function interpretation",
            guidance=(
                "Identify what the input and output represent. When interpreting a formula, answer in "
                "context rather than only computing a number."
            ),
            priority=15,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_piecewise_case",
            category="applied_word_problem",
            title="Piecewise or case problem",
            guidance=(
                "Determine which case applies before computing. Check thresholds such as at least, more "
                "than, inclusive, discount, or otherwise."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_table_schedule_reasoning",
            category="applied_word_problem",
            title="Table or schedule reasoning",
            guidance=(
                "Read the table/schedule in order. Track starting value, each transaction/change, and the "
                "final requested quantity."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_quantity_tracking",
            category="applied_word_problem",
            title="Quantity tracking",
            guidance=(
                "Track the quantity step by step through each change. Do not skip intermediate updates, "
                "especially purchases, returns, payments, remaining balances, or trips."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_unit_conversion_application",
            category="applied_word_problem",
            title="Unit conversion application",
            guidance=(
                "Convert units before combining quantities. Check whether the final answer should be in "
                "hours/minutes, miles/feet, Celsius/Fahrenheit, or another requested unit."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_inequality_constraint",
            category="applied_word_problem",
            title="Inequality or constraint",
            guidance=(
                "Translate words like at least, at most, between, maximum, minimum, inclusive, and support "
                "into inequalities before solving."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_multi_step",
            category="applied_word_problem",
            title="Multi-step applied problem",
            guidance=(
                "Solve each part in order. If later parts depend on earlier results, carry forward exact "
                "values when possible before rounding."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_multi_answer",
            category="applied_word_problem",
            title="Multi-answer applied problem",
            guidance=(
                "Count all [ANS] placeholders and return exactly that many answers in the same order, "
                "with units/rounding only if requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_mcq",
            category="applied_word_problem",
            title="MCQ option mapping",
            guidance=(
                "For multiple-choice applied problems, solve in context, compare to each option, and box "
                "only the final option letter."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="geometry_trig_angle_conversion",
            category="geometry_trig",
            title="Angle conversion",
            guidance=(
                "Use 180 degrees = pi radians. Track whether the requested final answer is in degrees "
                "or radians and simplify exact multiples of pi."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_arc_length_sector",
            category="geometry_trig",
            title="Arc length or sector area",
            guidance=(
                "Use radians for arc/sector formulas. Arc length is s = r theta and sector area is "
                "A = 1/2 r^2 theta."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_trig_equation",
            category="geometry_trig",
            title="Trigonometric equation",
            guidance=(
                "Find the reference angle, determine all valid quadrants in the requested interval, and "
                "return every solution required."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_inverse_trig",
            category="geometry_trig",
            title="Inverse trig",
            guidance=(
                "Use the principal-value range for inverse trig functions. Check quadrant restrictions "
                "before finalizing."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_quadrant_sign",
            category="geometry_trig",
            title="Quadrant/sign check",
            guidance=(
                "Use the quadrant to determine signs of sine, cosine, tangent, and coordinate values. "
                "Do not choose a value with the wrong sign."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="geometry_trig_coordinate_point",
            category="geometry_trig",
            title="Coordinate point trig",
            guidance=(
                "Relate x, y, and r using x^2 + y^2 = r^2. Match signs to the quadrant and compute trig "
                "ratios from coordinates."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_right_triangle",
            category="geometry_trig",
            title="Right triangle",
            guidance=(
                "Identify opposite, adjacent, hypotenuse, and the relevant angle. Use the correct trig "
                "ratio or Pythagorean theorem."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_law_of_sines_cosines",
            category="geometry_trig",
            title="Law of sines/cosines",
            guidance=(
                "Choose law of sines or cosines based on the given sides/angles. Watch for ambiguous SSA "
                "cases and compare possible triangles."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="geometry_trig_circle_geometry",
            category="geometry_trig",
            title="Circle geometry",
            guidance=(
                "Track radius, diameter, chord, tangent, arc, sector, and angle relationships. Convert "
                "degrees to radians when formulas require it."
            ),
            priority=10,
        ),
        RuleGuidance(
            rule_name="geometry_trig_multi_answer_order",
            category="geometry_trig",
            title="Multi-answer geometry/trig",
            guidance=(
                "Count all [ANS] placeholders and preserve the requested order. Put all answers in one "
                "comma-separated final box."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_mcq_option_mapping",
            category="geometry_trig",
            title="MCQ option mapping",
            guidance=(
                "For multiple-choice geometry/trig problems, compute the geometric/trig result, compare "
                "to every option, and box only the option letter."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="statistics_probability_hypothesis_test",
            category="statistics_probability",
            title="Hypothesis test",
            guidance=(
                "Identify null/alternative hypotheses, test statistic, sampling distribution, p-value or "
                "critical region, and rejection decision. Keep the requested form in the final answer."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_type_i_type_ii_power",
            category="statistics_probability",
            title="Type I/II error or power",
            guidance=(
                "Translate Type I, Type II, and power carefully. Type II is failing to reject under the "
                "alternative. Compute the non-rejection probability under the true alternative."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="statistics_probability_confidence_interval",
            category="statistics_probability",
            title="Confidence interval",
            guidance=(
                "Compute estimate ± critical value times standard error. Use z or t according to the "
                "problem conditions and preserve requested rounding."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_regression_correlation",
            category="statistics_probability",
            title="Regression or correlation",
            guidance=(
                "Identify slope, intercept, residual, prediction, correlation, or R-squared. Interpret "
                "slope/intercept in context when requested."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_probability_counting",
            category="statistics_probability",
            title="Probability counting",
            guidance=(
                "Define the sample space and favorable outcomes. Check independence, replacement, "
                "conditional probability, complement, and at least/at most wording."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_distribution",
            category="statistics_probability",
            title="Distribution problem",
            guidance=(
                "Identify the distribution and parameters. Standardize if normal, use the correct mass "
                "or density formula otherwise, and keep probabilities in [0,1]."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_expected_value_variance",
            category="statistics_probability",
            title="Expected value or variance",
            guidance=(
                "Use E[X], Var(X), linearity of expectation, and variance rules carefully. Distinguish "
                "standard deviation from variance."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_sampling_distribution",
            category="statistics_probability",
            title="Sampling distribution",
            guidance=(
                "Use the sample mean/proportion distribution with the correct standard error. Apply CLT "
                "only when justified by sample size or assumptions."
            ),
            priority=8,
        ),
        RuleGuidance(
            rule_name="statistics_probability_mcq_option_mapping",
            category="statistics_probability",
            title="MCQ option mapping",
            guidance=(
                "For multiple-choice statistics/probability problems, compute the result, compare to the "
                "choices, and box only the option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_multi_answer",
            category="statistics_probability",
            title="Multi-answer statistics/probability",
            guidance=(
                "Count all [ANS] placeholders and return exactly that many comma-separated answers in "
                "the requested order."
            ),
            priority=1,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items


def register_all_default_guidance():
    items = []
    items.extend(register_default_linear_discrete_guidance())
    items.extend(register_default_ved_guidance())
    items.extend(register_default_remaining_category_guidance())
    return items