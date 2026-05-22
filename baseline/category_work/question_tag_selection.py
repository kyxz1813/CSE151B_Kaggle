import pandas as pd


def get_primary_category_ids(
    jsonl_path: str,
    target_category: str
) -> list:
    """
    Fast pandas-based JSONL reader that returns all IDs
    where primary_category == target_category.
    """

    # Read JSONL file
    df = pd.read_json(jsonl_path, lines=True)

    # Filter rows and extract IDs
    matching_ids = (
        df.loc[df["primary_category"] == target_category, "id"]
        .dropna()
        .tolist()
    )

    return matching_ids

def get_questions_by_ids(
    target_jsonl: str,
    ids: list
) -> list[dict]:
    """
    Returns rows from target_jsonl where id is in ids,
    including the question field.
    """

    df = pd.read_json(target_jsonl, lines=True)

    filtered = df.loc[
        df["id"].isin(ids),
        ["id", "question"]
    ]

    return filtered.to_dict(orient="records")


if __name__ == "__main__":
    tag_jsonl_file = "results/baseline3_rule_annotations/public_linear_discrete_rules.jsonl"
    questions_file = "data/public.jsonl"

    # Category to search for
    category = "applied_word_problem"

    ids = get_primary_category_ids(tag_jsonl_file, category)
    questions = get_questions_by_ids(questions_file, ids)


    for item in questions:
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")

    print(len(questions))