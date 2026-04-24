import argparse
import json
import cv2
from app.engine.image import load_image
from app.visualizer import print_report, visualize_matplotlib
from models import Config
from app.engine.engine import PaddleOCRAdapter
from pipeline import run



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output",    default="output/result.png")
    parser.add_argument("--no-viz",    action="store_true")
    args = parser.parse_args()

    config = Config(confidence_threshold=args.threshold)
    engine = PaddleOCRAdapter(config)
    image = load_image(args.image, config.max_image_side)

    if image is None:
        print(f"Could not load image: {args.image}")
        return

    pipeline_output, agent_result = run(image, engine, config, args.debug)

    if pipeline_output:
        print(f"MRZ en pipeline_output: {pipeline_output.mrz}")
        print_report(pipeline_output, agent_result)

    if pipeline_output is None:
        print(json.dumps(agent_result, ensure_ascii=False, indent=2))
        return

    print_report(pipeline_output, agent_result)

    if not args.no_viz:
        visualize_matplotlib(image, pipeline_output, args.output)

    print("\n── ALL OCR LINES (unclassified) ──")
    for line in pipeline_output.raw_lines:
        print(f"[{line.confidence:.2f}] {line.text}")

    print("\n── JSON OUTPUT ──────────────────────────────────────────")
    print(json.dumps(agent_result, ensure_ascii=False, indent=2))

    


if __name__ == "__main__":
    main()