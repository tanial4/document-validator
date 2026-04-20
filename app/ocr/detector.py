import argparse

from app.ocr.ocr import process
from app.ocr.visualization import visualize, print_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--output", default="result.png")
    args = parser.parse_args()

    output, image_bgr = process(args.image, args.debug)

    print_report(output)
    visualize(image_bgr, output, args.output)


if __name__ == "__main__":
    main()