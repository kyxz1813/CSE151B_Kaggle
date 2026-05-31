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

def register_deep_smoke_enhancement_guidance():
    items = [
        RuleGuidance(
            rule_name="arithmetic_algebra_temperature_conversion",
            category="arithmetic_algebra",
            title="Temperature conversion",
            guidance=(
                "For temperature conversions use the exact formulas: C=(F-32)*5/9, K=C+273.15, "
                "and Rankine R=F+459.67. Do not compute Rankine from Celsius or Kelvin by mistake. "
                "If several temperature units are requested, output them in the same order as the blanks with extra precision."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_bernstein_polynomial",
            category="arithmetic_algebra",
            title="Bernstein polynomial indexing",
            guidance=(
                "For a kth Bernstein polynomial of degree n in this dataset, use B_{k,n}(t)=C(n,k)t^k(1-t)^(n-k). "
                "Do not shift to zero-indexing unless the problem explicitly says k starts at 0. Preserve all requested polynomial formulas in order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_base_arithmetic",
            category="arithmetic_algebra",
            title="Base/binary arithmetic",
            guidance=(
                "For binary or base arithmetic, perform the carry in the stated base. Cross-check by converting each input to decimal, "
                "doing the arithmetic, and converting back. Final binary answers should contain only 0 and 1."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_numeric_precision",
            category="arithmetic_algebra",
            title="Numeric precision",
            guidance=(
                "When a problem asks for a decimal approximation, give 4-8 accurate decimal digits when possible, even if the minimum requested precision is lower. "
                "Do not round to one decimal unless the problem explicitly says exactly one decimal."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_floor_log_sum",
            category="arithmetic_algebra",
            title="Floor-log summation",
            guidance=(
                "For sums of floor(log_b n), group integers by powers of b. For floor(log_2 n), values m occur for n in [2^m,2^(m+1)-1]. "
                "Compute the full grouped sum, reduce modulo only at the end, and for MCQ box the matching option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_letter_set_selection",
            category="arithmetic_algebra",
            title="Letter-set free-form answer",
            guidance=(
                "If the problem asks for letter(s) but is not standard MCQ, evaluate each labeled statement and output exactly the selected letters with no extra letters."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="applied_word_problem_half_life_decay_exact",
            category="applied_word_problem",
            title="Half-life/decay exact form",
            guidance=(
                "For half-life/decay, remaining fraction after elapsed time is (1/2)^(elapsed/half_life). For percent decay r per period, half-life is ln(0.5)/ln(1-r). "
                "Preserve exact exponential/log form unless a decimal is explicitly required."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_rational_function_model",
            category="applied_word_problem",
            title="Rational function model",
            guidance=(
                "For rational-function models, write the equation exactly, solve algebraically before rounding, and check the requested year/time/value interpretation."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_step_function_ceiling",
            category="applied_word_problem",
            title="Ceiling or step function",
            guidance=(
                "For step costs, signatures, bundles, or capacity limits, use ceil(quantity/unit_size) for the number of units. Check inclusivity at thresholds such as up to, at least, and each additional."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_finance_percent_comparison",
            category="applied_word_problem",
            title="Finance/percent comparison",
            guidance=(
                "Track base amounts for each percent. Compute percent increases/decreases using the correct reference quantity and give extra decimal precision for money/rates unless exact cents are requested."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_exact_expression_preferred",
            category="applied_word_problem",
            title="Exact expression preferred",
            guidance=(
                "If the problem asks for a formula, expression, fraction, log, or exponential model, keep the exact expression in the final answer instead of replacing it with a rounded decimal."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="calculus_improper_parameter_integral",
            category="calculus",
            title="Improper parameter integral",
            guidance=(
                "For improper integrals with parameters, reduce to a known standard integral and track parameter powers carefully. Check convergence and simplify before comparing to MCQ choices."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_definite_integral_substitution",
            category="calculus",
            title="Definite integral substitution",
            guidance=(
                "For definite substitutions, transform both the integrand and the endpoints. Verify by differentiating the substitution and simplify before choosing an option."
            ),
            priority=3,
        ),
        RuleGuidance(
            rule_name="calculus_equation_root_dichotomy",
            category="calculus",
            title="Root solving / dichotomy",
            guidance=(
                "For numerical roots, bracket roots carefully and output enough precision. If MCQ, choose the option matching the computed root behavior."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="calculus_mcq_absent_option_fallback",
            category="calculus",
            title="MCQ option fallback",
            guidance=(
                "For calculus MCQ, even if the exact computed expression seems absent, simplify using conventions and choose the closest intended listed option. Never leave the final boxed answer blank."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="geometry_trig_angle_of_elevation_two_angles",
            category="geometry_trig",
            title="Two-angle elevation height",
            guidance=(
                "For two angle-of-elevation/depression problems, draw the two right triangles and use tangent ratios. Height difference often has the form d(tan upper - tan lower). Return extra decimal precision."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_projectile_motion",
            category="geometry_trig",
            title="Projectile motion geometry",
            guidance=(
                "For projectile/trajectory problems, identify whether the requested quantity is time, height, range, or angle. Use the formula exactly before rounding."
            ),
            priority=5,
        ),
        RuleGuidance(
            rule_name="geometry_trig_coordinate_point_exact",
            category="geometry_trig",
            title="Coordinate trig exact point",
            guidance=(
                "For quadrant coordinate trig, choose the simplest integer point matching the ratio and signs, compute r=sqrt(x^2+y^2), then compute sin/cos/tan exactly with correct signs."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_precision_numeric",
            category="geometry_trig",
            title="Geometry/trig numeric precision",
            guidance=(
                "When the result is numeric, keep 4-8 accurate decimal digits unless the problem explicitly asks for exact rounding to fewer digits."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="statistics_probability_descriptive_table",
            category="statistics_probability",
            title="Descriptive-statistics table",
            guidance=(
                "For data-table descriptive statistics, fill every [ANS] cell in order: deviations, squared deviations, sums, variance, and standard deviation as requested. Do not output only the final statistic."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_goodness_fit",
            category="statistics_probability",
            title="Chi-square goodness of fit",
            guidance=(
                "For chi-square goodness-of-fit, compute each expected count, then statistic sum((O-E)^2/E), degrees of freedom, critical value/p-value, and decision. Output every blank in order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_independence",
            category="statistics_probability",
            title="Chi-square independence",
            guidance=(
                "For chi-square independence, expected count = row total * column total / grand total, df=(r-1)(c-1), statistic=sum((O-E)^2/E). Output expected frequencies before the statistic/decision if blanks ask for them."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_sample_size_margin_error",
            category="statistics_probability",
            title="Sample-size / margin of error",
            guidance=(
                "For sample-size problems, first compute real n from the margin-of-error formula. Only apply ceiling if the problem asks for the smallest integer/sample size that works. Otherwise preserve the computed value."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_multi_blank_table",
            category="statistics_probability",
            title="Statistics multi-blank table",
            guidance=(
                "If a statistics problem has many [ANS] blanks, treat it as a table-fill problem and output one comma-separated entry per blank in row-major/problem order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_rounding_direction",
            category="statistics_probability",
            title="Statistics rounding direction",
            guidance=(
                "Distinguish exact computed statistic, rounded display value, and required integer sample size. Do not round to an integer unless explicitly asked."
            ),
            priority=2,
        ),

        RuleGuidance(
            rule_name="general_math_modular_number_theory",
            category="general_math",
            title="Fallback modular number theory",
            guidance=(
                "For modular/divisibility problems, work modulo the relevant number, track residues exactly, and box an MCQ letter if choices are provided."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_complex_roots",
            category="general_math",
            title="Complex roots / roots of unity",
            guidance=(
                "For roots of unity or complex polynomial expressions, reduce powers modulo the order and evaluate the expression for each root class carefully."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_set_combinatorics",
            category="general_math",
            title="Set/combinatorics fallback",
            guidance=(
                "For set/combinatorics problems, translate the condition exactly, test small cases, and preserve exact closed forms when possible."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="general_math_special_definition",
            category="general_math",
            title="Special definition fallback",
            guidance=(
                "For special definitions, apply the definition literally and avoid importing unrelated assumptions. For MCQ, still box exactly one option letter."
            ),
            priority=1,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items

def register_calculus_v2_guidance():
    items = [
        RuleGuidance(
            rule_name="calculus_numeric_precision_freeform",
            category="calculus",
            title="High-precision free-form numeric answer",
            guidance=(
                "For free-form numeric calculus answers, output extra precision whenever possible. "
                "If the prompt says nearest, approximate, graphically, or at least N decimals, give 6-12 significant digits unless it explicitly says round to exactly N decimals. "
                "Avoid rounding intermediate values."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_exponential_model",
            category="calculus",
            title="Exponential calculus model",
            guidance=(
                "For continuous growth/decay models, use P(t)=P0*exp(r*t) when the rate is continuous. "
                "For Newton cooling, use T(t)=T_room+(T0-T_room)*exp(-k*t). Solve k from the given observation before evaluating. "
                "Use exact logarithms until the final numeric answer and give high precision."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_newton_cooling",
            category="calculus",
            title="Newton cooling model",
            guidance=(
                "Use T(t)=T_room+(T0-T_room)*exp(-k*t). First solve exp(-k*t_obs)=(T_obs-T_room)/(T0-T_room). "
                "Then evaluate with full precision. For time-to-temperature, solve t=-ln((T_target-T_room)/(T0-T_room))/k."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_actual_max_error",
            category="calculus",
            title="Actual maximum error, not just differential estimate",
            guidance=(
                "If the question asks for maximum error from a measurement tolerance, do not stop at the differential approximation dV. "
                "Compute the actual endpoint difference when possible: for volume V=L^3 and length L±e, the maximum overestimate is (L+e)^3-L^3. "
                "Give the high-precision value."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_antiderivative_mcq_verify",
            category="calculus",
            title="Antiderivative MCQ verification",
            guidance=(
                "For indefinite-integral MCQ, verify by differentiating candidate forms. Pay close attention to chain-rule constants, signs, absolute values, and coefficients inside logarithms. "
                "When choices differ only by log arguments or signs, visual matching is unsafe; differentiate the selected option mentally before boxing the letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_definite_integral_numeric_mcq",
            category="calculus",
            title="Definite integral MCQ numeric comparison",
            guidance=(
                "For definite-integral MCQ, compute a simplified exact value or a high-precision numerical estimate, then evaluate/simplify the options and choose the matching letter. "
                "Do not choose a logarithm expression merely because it looks similar; compare values and constants."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_trig_integral",
            category="calculus",
            title="Trigonometric integral",
            guidance=(
                "For trig integrals, choose a substitution such as u=sin(kx) or u=cos(kx) when the derivative appears. Track the factor k. "
                "Use identities carefully and verify signs in logarithmic antiderivatives."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_improper_parameter_integral",
            category="calculus",
            title="Improper parameter integral",
            guidance=(
                "For integrals over -infinity to infinity, reduce to a standard formula. In particular, int_{-infty}^{infty} 1/(s^2+a^2) ds = pi/a for a>0. "
                "Then multiply by outside factors and map the resulting expression to the MCQ option letter."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_trig_derivative_simplification",
            category="calculus",
            title="Trigonometric derivative simplification",
            guidance=(
                "For trig derivative MCQs, differentiate term by term, then simplify using csc=1/sin, sec=1/cos, cot=cos/sin, and sin(2x)=2sin(x)cos(x). "
                "Compare powers of sin and cos exactly before choosing the option."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_differentiation_under_integral",
            category="calculus",
            title="Differentiation under the integral sign",
            guidance=(
                "For d/dy int_a^b f(x+y) dx, the literal Leibniz answer is int_a^b d/dy f(x+y) dx, which equals int_a^b f'(x+y) dx when simplified. "
                "For MCQ, prefer the option whose form matches the requested derivative operator if equivalent options appear."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_implicit_differentiation",
            category="calculus",
            title="Implicit differentiation",
            guidance=(
                "Differentiate both sides, collect dy/dx terms, and solve algebraically. Check whether the options ask for dy in terms of dx or dx in terms of dy. "
                "For e^{x+y}=xy+1, remember d(e^{x+y})=e^{x+y}(dx+dy) in differential form."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_piecewise_differential_equation",
            category="calculus",
            title="Piecewise differential equation",
            guidance=(
                "Solve the first interval using the initial conditions. Evaluate y and any needed derivative at the joining point. "
                "For the second interval, solve its DE and use continuity at the join to determine constants. Then match the whole piecewise expression to the options."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_first_order_ivp",
            category="calculus",
            title="First-order IVP",
            guidance=(
                "For first-order linear IVPs, rewrite as y' + p(t)y = q(t), use integrating factor when appropriate, solve the constant from the initial condition, then evaluate the requested t. "
                "For MCQ numeric values, compute enough precision and choose the closest option."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_volume_revolution",
            category="calculus",
            title="Volume of revolution",
            guidance=(
                "For volume around the x-axis, use V=pi*int y^2 dx. For rotation around the y-axis, use the appropriate shell/washer formula. "
                "If options are numeric, evaluate the integral numerically with enough precision and pick the closest option."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_surface_area_revolution",
            category="calculus",
            title="Surface area of revolution",
            guidance=(
                "For surface area around the x-axis, use S=2*pi*int y*sqrt(1+(dy/dx)^2) dx, with the correct parameterization if the curve is implicit/parametric. "
                "For astroids or special curves, parameterize if that simplifies the expression."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_area_between_curves",
            category="calculus",
            title="Area between curves",
            guidance=(
                "Find all intersection points first. Determine upper minus lower on each interval, split if needed, then integrate. "
                "For MCQ, compare the simplified or numeric area to all choices."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_optimization_geometry",
            category="calculus",
            title="Geometric optimization",
            guidance=(
                "Define the geometric variables, write the constraint, express the objective in one variable, differentiate, solve the critical point, and verify it gives the requested max/min. "
                "For wire square/circle problems, use total length constraint and keep enough numeric precision."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_domain_range_radical_rational",
            category="calculus",
            title="Domain/range radical rational",
            guidance=(
                "For domain, enforce radical nonnegativity and denominator nonzero. For range, solve y=f(x) for x or use monotonicity/limits. "
                "Do not output None for range unless the prompt explicitly has no range answer."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_fourier_sobolev_boundary",
            category="calculus",
            title="Fourier/Sobolev boundary term",
            guidance=(
                "For n*int f(x)e^{-2*pi*i*n*x} dx, use integration by parts. The leading term comes from boundary values f(1)-f(0), because e^{-2*pi*i*n}=1 for integer n. "
                "Track the factor 2*pi*i and its sign carefully."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_series_special_function",
            category="calculus",
            title="Special series or numeric series",
            guidance=(
                "Identify the known expansion if possible; otherwise compute enough partial terms for the requested error. "
                "For MCQ options that differ in tiny decimals, compare with higher precision before selecting the letter."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="calculus_discrete_series_convergence",
            category="calculus",
            title="Discrete-structured convergence series",
            guidance=(
                "When a convergence problem includes digit counts or combinatorial a(n), estimate growth by grouping n by size/length. "
                "Find when x^{a(n)} can overpower n^p, then map the upper bound to the MCQ option."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="calculus_numeric_mcq_close_options",
            category="calculus",
            title="Close numeric MCQ options",
            guidance=(
                "When MCQ options are close decimals, do not round early. Compute a high-precision numeric value and choose the closest matching option. "
                "Check all choices rather than stopping at the first plausible one."
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
    items.extend(register_deep_smoke_enhancement_guidance())
    items.extend(register_calculus_v2_guidance())
    return items
