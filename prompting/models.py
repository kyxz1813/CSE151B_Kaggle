from dataclasses import dataclass, field


@dataclass(slots=True)
class Problem:
    id: int
    question: str
    options: list | None = None
    answer: object = None
    tags: set = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    @property
    def is_mcq(self):
        return bool(self.options)

    @property
    def answer_format(self):
        return "mcq" if self.is_mcq else "free_form"


@dataclass(slots=True)
class PromptContext:
    problem: Problem
    strategy_name: str
    route_name: str
    tags: set = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PromptSpec:
    name: str
    system_prompt: str
    user_prompt: str
    few_shot_messages: list = field(default_factory=list)
    generation_hints: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shot_messages)
        messages.append({"role": "user", "content": self.user_prompt})
        return messages


def problem_from_record(record):
    metadata = dict(record.get("metadata") or {})
    for key in (
        "qwen_categories",
        "primary_category",
        "category_tag_raw_output",
        "category_tag_prompt",
    ):
        if key in record:
            metadata[key] = record[key]

    return Problem(
        id=int(record["id"]),
        question=str(record["question"]),
        options=record.get("options"),
        answer=record.get("answer"),
        tags=set(record.get("tags") or []),
        metadata=metadata,
    )
