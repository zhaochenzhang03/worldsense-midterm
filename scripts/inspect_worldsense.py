from collections import Counter

from src.worldsense_data import load_worldsense


JSON_PATH = "data/worldsense/worldsense_qa.json"


def main():

    examples = load_worldsense(JSON_PATH)

    print("=" * 70)
    print("WorldSense Dataset")
    print("=" * 70)

    print(f"Total QA examples: {len(examples)}")
    print(
        f"Unique videos: "
        f"{len(set(x.video_id for x in examples))}"
    )

    print("\nDomains:")
    print(Counter(x.domain for x in examples))

    print("\nTask domains:")
    print(Counter(x.task_domain for x in examples))

    print("\nTask types:")
    print(Counter(x.task_type for x in examples))

    print("\nAudio classes:")
    audio_counter = Counter()

    for example in examples:
        for audio_class in example.audio_class:
            audio_counter[audio_class] += 1

    print(audio_counter)

    print("\nFirst example:")
    x = examples[0]

    print("UID:", x.uid)
    print("Video:", x.video_id)
    print("Domain:", x.domain)
    print("Audio:", x.audio_class)
    print("Task:", x.task_type)
    print("Question:", x.question)

    for candidate in x.candidates:
        print(candidate)

    print("Answer:", x.answer)


if __name__ == "__main__":
    main()