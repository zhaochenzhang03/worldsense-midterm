import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorldSenseExample:
    uid: str
    video_id: str

    video_duration: str
    duration_group: str

    domain: str
    sub_category: str
    audio_class: list[str]
    video_caption: str

    task_id: str
    task_domain: str
    task_type: str

    question: str
    candidates: list[str]
    answer: str


def load_worldsense(json_path: str | Path) -> list[WorldSenseExample]:
    """
    Load the original WorldSense JSON and flatten each
    video's task0/task1/... entries into individual QA examples.
    """

    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []

    for video_key, video_data in data.items():

        video_id = video_data["video_id"]

        # Find task0, task1, task2, ...
        task_keys = [
            key
            for key in video_data.keys()
            if key.startswith("task")
            and isinstance(video_data[key], dict)
        ]

        for task_id in task_keys:

            task = video_data[task_id]

            example = WorldSenseExample(
                uid=f"{video_id}::{task_id}",
                video_id=video_id,

                video_duration=video_data.get(
                    "video_duration", ""
                ),
                duration_group=video_data.get(
                    "duration", ""
                ),

                domain=video_data.get(
                    "domain", ""
                ),
                sub_category=video_data.get(
                    "sub_category", ""
                ),
                audio_class=video_data.get(
                    "audio_class", []
                ),
                video_caption=video_data.get(
                    "video_caption", ""
                ),

                task_id=task_id,
                task_domain=task.get(
                    "task_domain", ""
                ),
                task_type=task.get(
                    "task_type", ""
                ),

                question=task.get(
                    "question", ""
                ),
                candidates=task.get(
                    "candidates", []
                ),
                answer=task.get(
                    "answer", ""
                ),
            )

            examples.append(example)

    return examples