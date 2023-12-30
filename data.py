import json
from dataclasses import asdict, dataclass, is_dataclass
from glob import glob
from typing import Any

from dacite import from_dict


Logprobs = list[dict[str, float]]

@dataclass
class AnswerPrediction:
    pred_correct: bool | None
    prob_correct: float | None = None

    # For text generation models:
    model_output: str | None = None
    logprobs: Logprobs | None = None


@dataclass
class Answer:
    text: str
    correct: bool
    prediction: AnswerPrediction | None = None


@dataclass
class Item:
    question: str
    answers: list[Answer]
    multiple: bool


@dataclass
class Metadata:
    dataset: str
    split: str
    extra: dict[str, Any]


@dataclass
class Article:
    text: str
    items: list[Item]
    metadata: Metadata

    @classmethod
    def from_json(cls, data: str) -> "Article":
        return from_dict(data_class=cls, data=json.loads(data))

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def load_dataset(dataset: str) -> dict[str, list[Article]]:
    """Load all splits of a dataset from JSONL files.

    Args:
        dataset: The name of the dataset to load (e.g. "dw").
    """
    splits = {}
    for path in glob(f"data/{dataset}/*.jsonl"):
        split = path.split("/")[-1].split(".")[0]
        with open(path, encoding="utf-8") as infile:
            splits[split] = [Article.from_json(line) for line in infile]
    return splits


class DataclassEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)
