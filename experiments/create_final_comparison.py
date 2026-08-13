from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "outputs"
    / "day5_final"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_comparison"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    noisy = Image.open(
        INPUT_DIR / "noisy.png"
    ).convert("L")

    restored = Image.open(
        INPUT_DIR / "restored.png"
    ).convert("L")

    ground_truth = Image.open(
        INPUT_DIR / "ground_truth.png"
    ).convert("L")

    # Resize noisy image for visual comparison
    noisy = noisy.resize(
        restored.size
    )

    images = [
        ("Noisy Input", noisy),
        ("ResidualSR Restored", restored),
        ("Ground Truth", ground_truth),
    ]

    width, height = restored.size

    label_height = 40

    comparison = Image.new(
        "L",
        (
            width * 3,
            height + label_height
        ),
        color=255
    )

    draw = ImageDraw.Draw(
        comparison
    )

    for i, (label, image) in enumerate(images):

        x = i * width

        comparison.paste(
            image,
            (x, label_height)
        )

        draw.text(
            (x + 10, 10),
            label,
            fill=0
        )

    output_path = (
        OUTPUT_DIR
        / "final_comparison.png"
    )

    comparison.save(
        output_path
    )

    print(
        "Final comparison saved:"
    )

    print(output_path)


if __name__ == "__main__":
    main()