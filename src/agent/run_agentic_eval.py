import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from agent.log import setup_logging
from agent.evaluation import AgenticEvaluator, EvalConfig, build_agent


def main():
    config = EvalConfig.from_cli()
    logger = setup_logging(verbose=config.verbose)
    os.makedirs(os.path.dirname(config.output) or ".", exist_ok=True)

    agent = build_agent(config, logger)
    AgenticEvaluator(config, agent, logger).run()


if __name__ == "__main__":
    main()
