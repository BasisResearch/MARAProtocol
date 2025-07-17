import logging

logging.basicConfig(level=logging.INFO)
import grpc
import argparse
import os
import pandas as pd
import json

from generated.mara import mara_evaluation_controller_pb2 as controller_pb2
from generated.mara import mara_evaluation_controller_pb2_grpc as controller_grpc
import time
from .evaluation_controller import EvaluationController


def get_environment_ids():
    """
    Get list of environment names based on the `mcqs` folder, considering all the json files.
    """
    logging.info(os.getcwd())
    environment_ids = []
    for file in os.listdir("./python_examples/autumnbench/mcqs"):
        if file.endswith(".json"):
            environment_ids.append(file.split(".")[0])
    return environment_ids


def run_multi_environment_evaluation():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Run MARA evaluation and save results to CSV.")
    parser.add_argument("--output-csv",
                        type=str,
                        required=True,
                        help="Path to the output CSV file.")
    parser.add_argument("--model",
                        type=str,
                        default="openai/gpt-4o",
                        help="Model to use for evaluation.")
    parser.add_argument("--llm-provider",
                        type=str,
                        default="openrouter",
                        help="LLM provider to use for evaluation.")
    parser.add_argument("--agent",
                        type=str,
                        default="autumn_llm_interactive_agent_v1",
                        help="Agent to use for evaluation.")
    args = parser.parse_args()
    output_csv_path = args.output_csv

    # Connect to evaluation controller
    is_connected = False
    while not is_connected:
        try:
            controller = EvaluationController()
            logging.info(
                "Initializing controller with Autumn Defect Detection environment"
            )
            init_response = controller.initialize(
                controller_pb2.ControllerInitializeRequest(
                    environment_ids=get_environment_ids(),
                    transitions=
                    [],  # Normally, transition need to be specified, but if we delegate that logic to evaluation controller, we can leave it empty
                    agent_id=args.agent,
                    config={
                        "llm_provider": args.llm_provider,
                        "llm_model": args.model,
                        "agent": args.agent
                    }))

            is_connected = True
        except Exception as e:
            logging.error(f"Error connecting to evaluation controller: {e}")
            time.sleep(1)

    # Initialize controller with only text adventure

    controller_id = init_response.controller_id
    logging.info(
        f"Initialized controller: {init_response.message}, controller_id: {controller_id}"
    )

    # Run evaluation
    logging.info("Starting evaluation")
    run_response = stub.RunEvaluation(
        controller_pb2.RunEvaluationRequest(controller_id=controller_id,
                                            reset=True,
                                            max_transitions=100,
                                            timeout=300.0))

    # Print results
    logging.info("Evaluation complete, printing results")
    print("\n=== Evaluation Results ===")
    print(f"Status: {run_response.message}")
    print(f"Aggregate Reward: {run_response.aggregate_reward}")
    print("\nEnvironment Rewards:")
    # Convert gRPC map to a standard dict for easier processing/serialization
    env_rewards_dict = {
        env_id: reward
        for env_id, reward in run_response.environment_rewards.items()
    }
    for env_id, reward in env_rewards_dict.items():
        print(f"  {env_id}: {reward}")

    print("\nEnvironment Sequence:")
    for i, env_id in enumerate(run_response.environments_visited):
        print(f"  {i+1}. {env_id}")

    print(f"\nEvaluation Complete: {run_response.evaluation_complete}")

    # Write environment rewards dict results to CSV
    df = pd.DataFrame(env_rewards_dict.items(),
                      columns=['Environment', 'Reward'])
    df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_multi_environment_evaluation()
