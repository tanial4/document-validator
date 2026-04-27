import argparse
import json
import cv2
from app.visualizer import print_report, visualize_matplotlib
from models import Config
from app.engine.engine   import PaddleOCRAdapter
from app.backends        import OllamaBackend
from pipeline            import run



BACKENDS = {
    "ollama": lambda cfg: OllamaBackend(cfg.ollama_url, cfg.ollama_model),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output",    default="output/result.png")
    parser.add_argument("--no-viz",    action="store_true")
    parser.add_argument("--backend",   choices=["ollama", "claude", "openai"],
                        default="ollama")
    args = parser.parse_args()

    config  = Config(confidence_threshold=args.threshold)
    backend = BACKENDS[args.backend](config)
    engine  = PaddleOCRAdapter(config)
    image   = cv2.imread(args.image)

    if image is None:
        print(f"Could not load image: {args.image}")
        return

    pipeline_output, agent_result = run(image, engine, config, backend, args.debug)

    if pipeline_output is None:
        print(json.dumps(agent_result, ensure_ascii=False, indent=2))
        return

    print_report(pipeline_output, agent_result)

    if not args.no_viz:
        visualize_matplotlib(image, pipeline_output, args.output)

    print("\n── JSON OUTPUT ──────────────────────────────────────────")
    print(json.dumps(agent_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()