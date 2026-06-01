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




def register_arithmetic_v2_guidance():
    items = [
        RuleGuidance(
            rule_name="arithmetic_algebra_temperature_conversion",
            category="arithmetic_algebra",
            title="Temperature conversion canonical precision",
            guidance=(
                "For Fahrenheit temperature conversions use exact formulas: C=(F-32)*5/9, "
                "K=C+273.15, and Rankine R=F+459.67. Output Celsius and Kelvin with 12-15 "
                "significant digits when possible, and Rankine with the exact decimal from F+459.67. "
                "Do not round C/K to 4 decimals unless the prompt explicitly says exactly 4 decimals."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_bernstein_polynomial",
            category="arithmetic_algebra",
            title="Bernstein polynomial canonical syntax",
            guidance=(
                "For Bernstein polynomial answers, use exactly the form C(n,k)*t^k*(1-t)^(n-k). "
                "In this dataset, '1st' means k=1, not k=0. Use explicit * signs. Do not write "
                "implicit multiplication like 3t. Do not omit exponent 1: write t^1 and (1-t)^1 when the exponent is 1."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_base_arithmetic",
            category="arithmetic_algebra",
            title="Binary/base arithmetic direct mode",
            guidance=(
                "For binary/base arithmetic, avoid long carry narration. Convert each addend to decimal, add, "
                "convert the result back to the requested base, and finish. Final answers must contain only digits "
                "valid in the base. For binary, final entries must contain only 0 and 1."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_numeric_precision",
            category="arithmetic_algebra",
            title="Numeric precision discipline",
            guidance=(
                "If the prompt says 'at least N decimal places', 'accurate to', 'correct to', 'graphically', "
                "or asks for a decimal approximation, give 4-8 decimal places or 8-15 significant digits when possible. "
                "Do not interpret 'at least 1 decimal place' as 'round to exactly 1 decimal place'."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_exponential_log_solve",
            category="arithmetic_algebra",
            title="Exponential/log solve precision",
            guidance=(
                "For equations like p=a*b^q, solve q=ln(p/a)/ln(b). Do not stop at a one-decimal graphical estimate. "
                "Return at least 4 decimal places when possible, e.g. 2.2892 rather than 2.3."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_formula_then_evaluate",
            category="arithmetic_algebra",
            title="Formula blanks before numeric evaluation",
            guidance=(
                "If the problem first asks for formula pieces like A=[ANS] and B=[ANS], keep those answers symbolic. "
                "Do not substitute the later numerical values into formula blanks. Only substitute numbers for the later evaluation blank. "
                "Example: if A and B contain variables S,T,W, answer A and B with expressions in S,T,W, then give the numerical decimal for R."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_letter_set_selection",
            category="arithmetic_algebra",
            title="Free-form letter-set formatting",
            guidance=(
                "When the final answer is a set/list of letters in a free-form problem, concatenate uppercase letters with no commas or spaces, "
                "unless the prompt explicitly requests commas. Use BCEG, not B,C,E,G."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_ordered_pair_answer",
            category="arithmetic_algebra",
            title="Ordered pair formatting",
            guidance=(
                "If the requested answer is a point, coordinate, or ordered pair, preserve tuple parentheses and order. "
                "Use (x,y), not y,x and not x,y without parentheses."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_canonical_symbolic_syntax",
            category="arithmetic_algebra",
            title="Canonical symbolic syntax",
            guidance=(
                "For symbolic final answers, use explicit multiplication with *, powers with ^, and parentheses around compound factors. "
                "Prefer e^(...) or exp(...) over informal exponent notation. Avoid implicit multiplication like 6e^{16x} or 3t."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_scientific_expression_style",
            category="arithmetic_algebra",
            title="Exponential expression style",
            guidance=(
                "For expressions involving exponentials, use canonical plain-text syntax such as 6*e^(16*x) or 6*exp(16*x). "
                "If a separate coefficient/value is requested, keep it as a separate comma-separated answer."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_multi_answer",
            category="arithmetic_algebra",
            title="Multi-answer blank audit",
            guidance=(
                "Before solving, count every [ANS] blank and label each blank as formula, number, letter-list, pair, interval, or text. "
                "Final answer must contain exactly one entry per blank in the same order. Do not collapse formula blanks into substituted numbers."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_symbolic_manipulation",
            category="arithmetic_algebra",
            title="Symbolic manipulation canonical form",
            guidance=(
                "Preserve exact symbolic structure. Use explicit * for products, ^ for powers, and parentheses around sums/products. "
                "Do not decimalize symbolic answers unless a decimal is explicitly requested."
            ),
            priority=3,
        ),
        RuleGuidance(
            rule_name="arithmetic_algebra_conversion_representation",
            category="arithmetic_algebra",
            title="Conversion representation",
            guidance=(
                "Track requested representation exactly: fraction, decimal, percent, base notation, exponential notation, or unit. "
                "For decimal conversion tasks, output extra precision unless exact rounding is specified."
            ),
            priority=3,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items

def register_statistics_v2_guidance():
    items = [
        RuleGuidance(
            rule_name="statistics_probability_descriptive_table",
            category="statistics_probability",
            title="Descriptive-statistics table",
            guidance=(
                "Fill every blank in order. For a standard-deviation table, output each deviation, each squared deviation, "
                "the sum of squared deviations, the sample variance using N-1 when shown, and the standard deviation. "
                "Use plain integers for integer cells when exact, and give the final statistic with 10-15 significant digits."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_stat_table_high_precision",
            category="statistics_probability",
            title="High-precision statistics table",
            guidance=(
                "For table-based descriptive statistics, do not over-round final computed values. Keep exact integer entries as integers, "
                "but give final variance/standard-deviation values with high precision."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_goodness_fit",
            category="statistics_probability",
            title="Chi-square goodness-of-fit",
            guidance=(
                "Compute expected counts, then chi-square statistic=sum((O-E)^2/E), df=k-1, critical value or p-value as requested. "
                "For final support-the-claim questions, output only YES or NO unless another blank explicitly asks for reject/fail-to-reject."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_independence",
            category="statistics_probability",
            title="Chi-square independence",
            guidance=(
                "Expected frequency = row total * column total / grand total. Output expected frequencies in row-major table order, "
                "then the chi-square statistic, then the critical value, then one final YES/NO decision. Do not add Reject H0 as an extra final entry."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_decision_contract",
            category="statistics_probability",
            title="Chi-square decision answer contract",
            guidance=(
                "When the final wording asks 'Is there sufficient data to support the claim?', the final decision blank expects YES or NO. "
                "Do not include Reject H0, fail to reject, null hypothesis text, or explanatory phrases in the boxed answer unless explicitly requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_chi_square_assumption_letters",
            category="statistics_probability",
            title="Chi-square assumption letters",
            guidance=(
                "For assumption checks: Assumption 1 is all expected frequencies >= 1. Assumption 2 is at most 20 percent of expected frequencies < 5. "
                "Compute expected frequencies, decide which assumptions hold, then output only the embedded option letters in order."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_sample_size_margin_error",
            category="statistics_probability",
            title="Sample size / margin of error",
            guidance=(
                "Use the standard margin-of-error formula. For one mean, n=(z*sigma/E)^2. For equal-size difference of means, "
                "n=z^2*(sigma1^2+sigma2^2)/E^2. In this dataset, output the raw computed n with high precision unless the problem explicitly says integer, whole number, smallest integer, or round up."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_sample_size_raw_value",
            category="statistics_probability",
            title="Raw sample-size value",
            guidance=(
                "Do not automatically ceil sample-size calculations. If the answer blank is n=[ANS] and no exact integer rounding instruction appears, "
                "return the raw decimal value with 10-15 significant digits."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_type_i_type_ii_power",
            category="statistics_probability",
            title="Type I/II error or power",
            guidance=(
                "For Type II error, first find the non-rejection interval under H0 using the correct alpha/tails, then compute the probability of that interval under the true alternative mean. "
                "Output beta with 10-15 significant digits."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_type_ii_error_beta",
            category="statistics_probability",
            title="Type II error beta precision",
            guidance=(
                "For two-sided z tests, lower=mu0-zcrit*sigma/sqrt(n), upper=mu0+zcrit*sigma/sqrt(n). "
                "Beta=P(lower <= Xbar <= upper | mu=mu1). Keep high precision in the final probability."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_probability_threshold_log",
            category="statistics_probability",
            title="Probability threshold via logarithms",
            guidance=(
                "For p=a^n or p=(1-c)^n threshold problems, solve with n=ln(target)/ln(base). "
                "If the question asks for 'fewer than [ANS]' or a threshold value, output the raw threshold with high precision unless integer rounding is explicitly requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_embedded_letter_choices",
            category="statistics_probability",
            title="Embedded letter-choice blanks",
            guidance=(
                "If each [ANS] blank has its own A/B/C/D choices inside a free-form problem, solve each mini-question separately and output only the letters in order, comma-separated. "
                "Do not include explanations or option text in the final answer."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_f_critical_embedded_choices",
            category="statistics_probability",
            title="F critical embedded choices",
            guidance=(
                "For F-curve right-tail critical values, use the given numerator and denominator degrees of freedom in order. "
                "Compare the computed/table critical value against the embedded choices and output only the correct letter for each part."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="statistics_probability_uppercase_categorical",
            category="statistics_probability",
            title="Uppercase categorical answers",
            guidance=(
                "Use uppercase canonical categorical words in final answers: YES, NO, INCREASING, DECREASING. "
                "Do not output lowercase variants."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_high_precision_numeric",
            category="statistics_probability",
            title="High-precision statistics numeric answer",
            guidance=(
                "For probabilities, critical values, test statistics, sample sizes, regression values, and standard deviations, "
                "give 10-15 significant digits when possible unless exact rounding is explicitly requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_long_regression_vector",
            category="statistics_probability",
            title="Long regression vector task",
            guidance=(
                "For long x=c(...), y=c(...) regression tasks, avoid narrating element counting. Use compact computation steps and prioritize producing the final ordered list of requested statistics. "
                "Count all blanks first and keep high precision."
            ),
            priority=3,
        ),
        RuleGuidance(
            rule_name="statistics_probability_multi_answer",
            category="statistics_probability",
            title="Multi-answer statistics/probability",
            guidance=(
                "Count answer blanks, but handle known statistics decision templates carefully: if duplicated final placeholders only represent a decision sentence, output the single requested YES/NO decision rather than adding Reject H0 as another answer."
            ),
            priority=1,
        ),

        RuleGuidance(
            rule_name="statistics_probability_representative_choice_letters",
            category="statistics_probability",
            title="Representative/non-representative answer codes",
            guidance=(
                "For representative/non-representative prompts with embedded A/B choices, output only the A/B letter for each blank. "
                "Do not output REPRESENTATIVE or NON-REPRESENTATIVE in the final box."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_measurement_scale_abbreviation",
            category="statistics_probability",
            title="Measurement-scale abbreviations",
            guidance=(
                "For Nominal/Ordinal/Interval/Ratio classification blanks, output compact abbreviations: "
                "N for Nominal, O for Ordinal, I for Interval, R for Ratio. Do not spell out the words in the final box."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_true_false_abbreviation",
            category="statistics_probability",
            title="True/False abbreviations",
            guidance=(
                "For True/False statement prompts, output T or F for each blank. "
                "Do not output YES/NO or TRUE/FALSE in the final box."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_coded_categorical_answer",
            category="statistics_probability",
            title="Coded categorical answer format",
            guidance=(
                "When a statistics problem asks for categorical classifications with embedded choices or conventional codes, "
                "return the compact code letters only, comma-separated, in the same order as the blanks."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="statistics_probability_two_mean_equal_sample_size",
            category="statistics_probability",
            title="Two-mean equal-sample-size formula",
            guidance=(
                "For estimating the difference between two means with independent equal-size samples, use the prompt notation carefully. "
                "If the problem gives variances sigma_1^2 and sigma_2^2, use those variances directly in the numerator; do not square them again. "
                "Use the margin-of-error formula for a two-mean difference and return the raw n unless integer rounding is explicitly requested."
            ),
            priority=1,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items

def register_applied_word_problem_v2_guidance():
    items = [
        RuleGuidance(
            rule_name="applied_word_problem_half_life_decay_exact",
            category="applied_word_problem",
            title="Half-life / decay exact form",
            guidance=(
                "For half-life and decay problems, preserve exact expressions unless a decimal approximation is explicitly requested. "
                "For fraction remaining after years, use (1/2)^[(target_year-start_year)/half_life]. "
                "For half-life from a daily percent decay r, use [ln(0.5)]/[ln(1-r)]."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_half_life_fraction_remaining",
            category="applied_word_problem",
            title="Fraction remaining exact power",
            guidance=(
                "If the question asks what fraction remains after a time interval, output the exact power expression, not a decimal. "
                "Example: (1/2)^[(1999-1963)/31]."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_half_life_percent_decay_log",
            category="applied_word_problem",
            title="Percent decay half-life log form",
            guidance=(
                "For an element that decays by r percent each day, solve (1-r)^t=0.5 and output [ln(0.5)]/[ln(1-r)] unless a decimal is explicitly requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_finance_percent_comparison",
            category="applied_word_problem",
            title="Finance / paycheck percent comparison",
            guidance=(
                "Set up the exact ratio equation. Do not round money answers to cents unless the prompt explicitly asks for cents or dollars-and-cents. "
                "Give 10-15 significant digits for raw paycheck, salary, and total values."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_high_precision_money",
            category="applied_word_problem",
            title="High-precision money answer",
            guidance=(
                "For salary/paycheck/monthly income word problems, return the raw decimal value with high precision unless the prompt explicitly requests nearest cent or nearest dollar."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_mixture_exact_expression",
            category="applied_word_problem",
            title="Mixture exact ratio",
            guidance=(
                "For mixture/concentration problems, set solute equation exactly and prefer exact ratio expressions when no rounding is requested. "
                "Do not replace an exact expression with a short decimal."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_real_world_invertibility",
            category="applied_word_problem",
            title="Real-world invertibility",
            guidance=(
                "For real-world invertibility questions, decide whether each input gives a unique output and whether the output determines the input. "
                "Volume of water vs kg is invertible; accumulated rainfall during the storm is treated as invertible if cumulative rainfall increases; postage cost by weight is not invertible because it is stepwise. "
                "Use lowercase yes/no in the final answer."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_step_function_ceiling",
            category="applied_word_problem",
            title="Step-function / ceiling convention",
            guidance=(
                "For step-function cost problems, distinguish the formula blank from the separate rounded up/down blank. "
                "If the formula blank is followed by a separate 'rounded (up/down)' blank, put the simple proportional expression in the formula blank and put up in the direction blank."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_printing_signature_formula_convention",
            category="applied_word_problem",
            title="Printing signature formula convention",
            guidance=(
                "For printing-signature problems like 16 pages per signature and $0.16 per signature, the dataset often expects C(p)=0.16*p/16 in the formula blank, then up in the separate rounding blank. "
                "Do not put ceil(...) in the formula blank when there is a separate up/down answer blank."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_counting_mcq",
            category="applied_word_problem",
            title="Applied counting MCQ",
            guidance=(
                "For digit/counting MCQs, derive a compact counting formula and compare to answer choices. "
                "Always finish with a single boxed option letter even if the reasoning is long."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_modular_crypto_mcq",
            category="applied_word_problem",
            title="Modular crypto MCQ",
            guidance=(
                "For cryptography/modular arithmetic MCQs, compute the modular operation carefully and compare the numeric plaintext/ciphertext to the options. "
                "Output only the option letter in the final box."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="applied_word_problem_exact_or_high_precision",
            category="applied_word_problem",
            title="Exact or high-precision applied answer",
            guidance=(
                "If no rounding instruction appears, avoid short rounded decimals. Prefer exact expressions or high-precision decimal values according to the requested answer type."
            ),
            priority=1,
        ),
    ]

    for item in items:
        register_rule_guidance(item)

    return items

def register_geometry_trig_v2_guidance():
    items = [
        RuleGuidance(
            rule_name="geometry_trig_trig_equation",
            category="geometry_trig",
            title="Trig equation answer form",
            guidance=(
                "For trig equations, match the blank structure exactly. If the problem gives theta=[ANS]+[ANS] n, "
                "the first blank should be a principal value and the second blank should be the period such as pi or 2*pi."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_general_solution_exact_period",
            category="geometry_trig",
            title="General trig solution exact period",
            guidance=(
                "For tan(theta)=a with theta=[ANS]+[ANS]n, use atan(a), pi. "
                "Do not replace atan(a) or pi with rounded decimals when exact symbolic entries are accepted."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_arc_length_sector",
            category="geometry_trig",
            title="Arc length and sector formulas",
            guidance=(
                "Use s=r*theta with theta in radians. Convert degrees by theta=degrees*pi/180. "
                "If the answer blank asks for a numeric radius in feet/meters and exact form is not explicitly required, output a high-precision decimal."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_arc_radius_decimal_preferred",
            category="geometry_trig",
            title="Arc radius decimal answer",
            guidance=(
                "For arc length radius problems with degree angle, r=s/(theta*pi/180). "
                "Even if an expression with pi is mathematically equivalent, prefer a high-precision decimal when the blank is [ANS] feet/meters."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_coordinate_point",
            category="geometry_trig",
            title="Coordinate point trig",
            guidance=(
                "For quadrant point problems, choose a simple integer point that has the correct tangent/signs. "
                "Format coordinate points with no spaces, e.g. (-1,-3)."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_coordinate_point_decimal_value",
            category="geometry_trig",
            title="Coordinate point plus decimal trig value",
            guidance=(
                "If the problem asks for one valid point and then a trig value, output the point exactly as a coordinate pair and use a high-precision decimal for the trig value unless exact form is explicitly requested."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_exact_sqrt_plain_text",
            category="geometry_trig",
            title="Plain sqrt exact form",
            guidance=(
                "When the prompt says exact form and says to type sqrt, use plain text expressions like -2*sqrt(14)/9. "
                "Avoid LaTeX \\frac or \\dfrac in the final answer."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_angle_of_elevation_two_angles",
            category="geometry_trig",
            title="Two-angle elevation height",
            guidance=(
                "For two angles of elevation from the same horizontal distance d, height difference = d*(tan(top_angle)-tan(bottom_angle)). "
                "Use degrees when the angles are given in degrees, and keep at least 4 decimal places unless the prompt explicitly requires fewer."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_pythagorean_equation_canonical",
            category="geometry_trig",
            title="Canonical Pythagorean equation",
            guidance=(
                "For wire/tree Pythagorean equations where x is the wire/hypotenuse and the height is x-4, write the equation as "
                "13^2 + (x-4)^2 = x^2. Preserve this visual/canonical order in the final answer."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_bearing_vector_contract",
            category="geometry_trig",
            title="Bearing vector answer contract",
            guidance=(
                "For bearing problems, decompose into north/east components and then output exactly four entries: "
                "distance, first direction letter, angle, second direction letter. If the two travel legs are perpendicular, the distance may simplify to sqrt(a^2+b^2)."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_direct_numeric_trig_values",
            category="geometry_trig",
            title="Direct numeric trig values",
            guidance=(
                "For sin(0.6), cos(0.6), tan(0.6), treat the input as radians unless degrees are explicitly stated. "
                "Compute each value directly from the original input, not from rounded intermediate sin/cos values. Give 10-15 significant digits."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_high_precision_decimal_preferred",
            category="geometry_trig",
            title="High-precision geometry decimal",
            guidance=(
                "For geometry/trig numeric answers, use at least 4 decimal places unless the problem explicitly says to round to a smaller precision. "
                "If wording says 'if needed', keep extra precision."
            ),
            priority=1,
        ),
        RuleGuidance(
            rule_name="geometry_trig_mcq_long_reasoning",
            category="geometry_trig",
            title="Geometry MCQ finalization",
            guidance=(
                "For geometry/trig MCQs, always end with exactly one boxed option letter. If time is running out, choose the best matching option and finalize."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="geometry_trig_parallel_triangle_area_mcq",
            category="geometry_trig",
            title="Parallel triangle area MCQ",
            guidance=(
                "For DE parallel AB with intersecting cevians and given areas, use similarity and area-ratio relationships compactly, compare with answer choices, and finalize with one letter."
            ),
            priority=2,
        ),
        RuleGuidance(
            rule_name="geometry_trig_fourier_series_mcq",
            category="geometry_trig",
            title="Fourier series MCQ",
            guidance=(
                "For periodic extension Fourier series questions, derive the coefficient/form expression and compare symbolically to choices. "
                "Be careful about sign: (e^(2*pi*alpha)+1)/(e^(2*pi*alpha)-1) is different from its reciprocal/sign-flipped variants."
            ),
            priority=2,
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
    items.extend(register_arithmetic_v2_guidance())
    items.extend(register_statistics_v2_guidance())
    items.extend(register_applied_word_problem_v2_guidance())
    items.extend(register_geometry_trig_v2_guidance())
    return items
