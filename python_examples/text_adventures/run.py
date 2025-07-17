import logging

logging.basicConfig(level=logging.INFO)
import grpc
import argparse

from generated.mara import mara_evaluation_controller_pb2 as controller_pb2
from generated.mara import mara_evaluation_controller_pb2_grpc as controller_grpc


def run_multi_environment_evaluation():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Connect to evaluation controller
    logging.info("Connecting to evaluation controller")
    channel = grpc.insecure_channel('localhost:50059')
    stub = controller_grpc.MARAEvaluationControllerStub(channel)

    # Initialize controller with only text adventure
    logging.info("Initializing controller with text adventure environment")
    init_response = stub.Initialize(
        controller_pb2.ControllerInitializeRequest(
            environment_ids=["text_adventure_v1"],  # Only text adventure
            transitions=[],  # No transitions needed for single environment
            agent_id="text_adventure_agent_v1",
            config={}))

    controller_id = init_response.controller_id
    logging.info(
        f"Initialized controller: {init_response.message}, controller_id: {controller_id}"
    )

    # Run evaluation
    logging.info("Starting evaluation")
    run_response = stub.RunEvaluation(
        controller_pb2.RunEvaluationRequest(controller_id=controller_id,
                                            reset=True,
                                            max_transitions=1,
                                            timeout=300.0))

    # Print results
    logging.info("Evaluation complete, printing results")
    print("\n=== Evaluation Results ===")
    print(f"Status: {run_response.message}")
    print(f"Aggregate Reward: {run_response.aggregate_reward}")
    print("\nEnvironment Rewards:")
    for env_id, reward in run_response.environment_rewards.items():
        print(f"  {env_id}: {reward}")

    print("\nEnvironment Sequence:")
    for i, env_id in enumerate(run_response.environments_visited):
        print(f"  {i+1}. {env_id}")

    print(f"\nEvaluation Complete: {run_response.evaluation_complete}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_multi_environment_evaluation()
