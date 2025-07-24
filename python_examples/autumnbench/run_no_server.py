import logging
logging.basicConfig(level=logging.INFO)
import os
import pandas as pd
import hydra
import json
from omegaconf import DictConfig

from generated.mara import mara_evaluation_controller_pb2 as controller_pb2
from generated.mara import mara_evaluation_controller_pb2_grpc as controller_grpc
import time

from python_examples.autumnbench.evaluation_controller import EvaluationControllerNoServer


def get_environment_ids(data_dir: str, task_name: str):
    """
    Get list of environment names based on the `prompts` folder, considering all the json files.
    """
    logging.info(os.getcwd())
    environment_ids = []
    for file in os.listdir(f"{data_dir}/prompts/"):
        if file.endswith(f"{task_name}.json"):
            with open(f"{data_dir}/prompts/{file}", "r") as f:
                data = json.load(f)
                task_type = data["type"]
                match task_type:
                    case "masked_frame_prediction":
                        environment_ids.append(f"{data['program']}_mfp")
                    case "change_detection":
                        environment_ids.append(f"{data['program']}_cd")
                    case "planning":
                        environment_ids.append(f"{data['program']}_planning")
                    case _:
                        raise ValueError(f"Unknown task type: {task_type}")
    logging.info(f"Environment IDs: {environment_ids}") 
    environment_ids = list(set(environment_ids))
    return environment_ids

def run_multi_environment_evaluation(cfg: DictConfig):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    output_csv_path = cfg.output_csv
    logging.info(f"Running evaluation for {cfg} config")
    # Connect to evaluation controller
    logging.info("Initializing controller with Autumn environments")
    environment_ids = get_environment_ids(cfg.data_dir, cfg.task_name) if not cfg.envs else cfg.envs
    logging.info(f"Running evaluation for {environment_ids}")

    evaluation_controller = EvaluationControllerNoServer()
    init_success, init_message = evaluation_controller.initialize(
        environment_ids=environment_ids, 
        transitions=[],  # Normally, transition need to be specified, but if we delegate that logic to evaluation controller, we can leave it empty
        agent_id=cfg.agent,
        config={
            "llm_provider": cfg.llm_provider,
            "llm_model": cfg.model,
            "agent": cfg.agent,
            "logging_path": cfg.logging_path,
            "max_history_length": cfg.max_history_length,
            "stack_frames": cfg.stack_frames,
            "skip_frames": cfg.skip_frames,
            "per_env_max_steps": cfg.per_env_max_steps,
            "render_mode": cfg.render_mode,
            "seed": cfg.seed,
            "data_dir": cfg.data_dir,
            "task_name": cfg.task_name,
            "use_scratchpad": cfg.use_scratchpad,
            "instruction_type": cfg.instruction_type,
            "hint": cfg.hint
        }
    )

    logging.info(f"Initialized controller: {init_message}")
    
    # Run evaluation
    logging.info("Starting evaluation")
    run_response = evaluation_controller.run_evaluation(
        reset=True,
        max_transitions=100
    )
    
    # Print results
    logging.info("Evaluation complete, printing results")
    print("\n=== Evaluation Results ===")
    print(f"Status: {run_response["message"]}")
    print(f"Aggregate Reward: {run_response["aggregate_reward"]}")
    print("\nEnvironment Rewards:")
    # Convert gRPC map to a standard dict for easier processing/serialization
    env_rewards_dict = {env_id: reward for env_id, reward in run_response["environment_rewards"].items()}
    for env_id, reward in env_rewards_dict.items():
        print(f"  {env_id}: {reward}")
    
    print("\nEnvironment Sequence:")
    for i, env_id in enumerate(run_response["environments_visited"]):
        print(f"  {i+1}. {env_id}")
    
    print(f"\nEvaluation Complete: {run_response["evaluation_complete"]}")

    # Write environment rewards dict results to CSV
    df = pd.DataFrame(env_rewards_dict.items(), columns=['Environment', 'Reward'])
    df.to_csv(output_csv_path, index=False)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    run_multi_environment_evaluation(cfg)


if __name__ == '__main__':
    main()
