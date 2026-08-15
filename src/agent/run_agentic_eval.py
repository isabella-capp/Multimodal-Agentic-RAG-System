import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from agent.log import setup_logging
from agent.evaluation import AgenticEvaluator, EvalConfig, build_agent
from agent.trace import format_trace
from vlm.dataset import load_dataset


def _trace_one(agent, dataset, logger, max_try: int = 25) -> None:
    """Print one full tool-calling iteration (tool call → evidence → answer)."""
    for it in dataset[:max_try]:
        if not os.path.exists(it["image_path"]):
            continue
        run = agent.run(it["image_path"], it["question"], capture_messages=True)
        if run.tool_called:
            logger.info("Full tool-calling iteration (Q: %s | GT: %s):\n%s",
                        it["question"], it["answer"], format_trace(run.raw))
            return


def main():
    config = EvalConfig.from_cli()
    logger = setup_logging(verbose=config.verbose)
    os.makedirs(os.path.dirname(config.output) or ".", exist_ok=True)

    agent = build_agent(config, logger)
    if config.debug_samples:
        dataset = load_dataset(json_path=config.json_path, base_folder=config.base_folder)
        _trace_one(agent, dataset, logger)
    AgenticEvaluator(config, agent, logger).run()


if __name__ == "__main__":
    main()
