import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
import grpc
from concurrent import futures
import time

from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_environment_service_pb2 as env_service_pb2
from generated.mara import mara_environment_service_pb2_grpc as env_grpc
from generated.mara import mara_agent_pb2 as agent_pb2
from generated.mara import mara_agent_pb2_grpc as agent_grpc
from generated.mara import mara_evaluation_controller_pb2 as controller_pb2
from generated.mara import mara_evaluation_controller_pb2_grpc as controller_grpc
from .environment_interfaces import MARACompositeAutumnChangeDetectionServicer, MARACompositeAutumnPlanningServicer
from .environment_interfaces_mfp import MARACompositeAutumnMFPServicer
from .agent import MARARandomAgentServicer
from .llm_agent import ReactLLMAgentServicer, ReactLLMAgent2, SummaryReactLLMAgent, ReactVLMAgent, UnifiedReactAgent
from .simple_wm_agent import SimpleWMAgentServicer
from generated.mara import mara_environment_service_pb2 as env_service_pb2
from generated.mara import mara_environment_service_pb2_grpc as env_grpc

from concurrent import futures


class EvaluationControllerNoServer:

    def __init__(self) -> None:
        self.environments: Dict[str, Optional[Any]] = {}
        self.transitions: Dict[str, Dict[str, str]] = {}
        self.transition_messages: Dict[Any, Any] = {
        }  # Used both as nested dict and tuple keys
        self.configs: Dict[str, Dict[str, str]] = {}
        self.current_environment: Optional[str] = None
        self.environment_sequence: List[str] = []
        self.environment_rewards: Dict[str, float] = {}
        self.aggregate_reward: float = 0.0
        self.evaluation_complete: bool = False
        self.environment_ids: List[str] = []

    def initialize(self, environment_ids: List[str], transitions: List[Any],
                   agent_id: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        print("Initializing controller")
        print(f"Environment IDs: {environment_ids}")
        self.environment_ids = environment_ids
        self.environments = {
            **{
                f"{eid}_interactive_{config['task_name']}": None
                for eid in environment_ids
            },
        }
        self.transitions = {
            **{
                f"{environment_ids[eid]}_interactive_{config['task_name']}": {
                    "default":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "quit":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "finish":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "fail":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                }
                for eid in range(len(environment_ids) - 1)
            },
        }

        self.transition_messages = {
            **{
                f"{environment_ids[eid]}_interactive_{config['task_name']}": {
                    "default":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "quit":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "finish":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                    "fail":
                    f"{environment_ids[eid + 1]}_interactive_{config['task_name']}",
                }
                for eid in range(len(environment_ids) - 1)
            },
        }

        self.agents = {
            "autumn_random_interactive_agent_v1": MARARandomAgentServicer,
            "autumn_llm_interactive_agent_v1": ReactLLMAgentServicer,
            "autumn_llm_interactive_agent_v2": ReactLLMAgent2,
            "autumn_llm_summary_interactive_agent_v1": SummaryReactLLMAgent,
            "autumn_llm_image_interactive_agent_v1": ReactVLMAgent,
            "autumn_llm_unified_interactive_agent_v1": UnifiedReactAgent,
            "autumn_simple_wm_agent": SimpleWMAgentServicer,
        }

        # Initialize rewards for environments that exist
        for env_id in self.environments:
            self.environment_rewards[env_id] = 0.0

        # Store available transitions (only those involving available environments)
        for transition in transitions:
            from_env = transition.from_environment_id
            to_env = transition.to_environment_id

            # Skip transitions for non-existent environments
            if from_env not in self.environments or to_env not in self.environments:
                logger.warning(
                    f"Skipping transition {from_env} -> {to_env} (environment not available)"
                )
                continue

            condition = transition.transition_condition
            message = transition.transition_message

            if from_env not in self.transitions:
                self.transitions[from_env] = {}

            self.transitions[from_env][condition] = to_env
            self.transition_messages[(from_env, to_env)] = message

        # Reset state - use the first available environment
        available_envs = list(self.environments.keys())
        self.current_environment = available_envs[0] if available_envs else None
        self.environment_sequence = []
        self.aggregate_reward = 0.0
        self.evaluation_complete = False
        self.config = config
        return True, f"Initialized controller with available environments: {list(self.environments.keys())}, transitions: {self.transitions.keys()}, agent: {agent_id}"

    def run_evaluation(self,
                       reset: bool = False,
                       max_transitions: int = 10) -> Dict[str, Any]:
        """Run full evaluation across multiple environments"""
        logger.info(
            f"Starting evaluation with reset={reset}, max_transitions={max_transitions}"
        )

        if reset:
            self.environment_sequence = []
            self.environment_rewards = {
                env_id: 0.0
                for env_id in self.environments
            }
            self.aggregate_reward = 0.0
            self.evaluation_complete = False
            first_env_id = next(iter(self.environments.keys()))
            self.current_environment = first_env_id
            logger.info(
                f"Reset evaluation state, starting with environment: {self.current_environment}"
            )

        transitions_made = 0

        logger.info(
            f"Current environment before loop: {self.current_environment}")
        logger.info(
            f"Loop conditions: transitions_made={transitions_made}, max_transitions={max_transitions}, evaluation_complete={self.evaluation_complete}"
        )

        if transitions_made < max_transitions and not self.evaluation_complete:
            logger.info("Entering main evaluation loop")
        else:
            logger.warning(
                "Loop conditions not met, skipping main evaluation loop")

        while transitions_made < max_transitions and not self.evaluation_complete:
            # Run current environment
            if not self.current_environment:
                logger.warning(
                    "No current environment, marking evaluation as complete")
                self.evaluation_complete = True
                break

            print(f"Current environment: {self.current_environment}")
            env_id = self.current_environment
            logger.info(f"Adding environment to sequence: {env_id}")
            self.environment_sequence.append(env_id)

            logger.info(
                f"Running environment: {env_id}, agent: {self.config.get('agent', 'autumn_llm_interactive_agent_v1')}"
            )
            reward, terminal_condition, final_state = self.run_environment(
                env_id, self.agents[self.config.get(
                    "agent", "autumn_llm_interactive_agent_v1")])
            logger.info(
                f"Environment run complete. Reward: {reward}, Terminal condition: {terminal_condition}"
            )

            # Store reward
            self.environment_rewards[env_id] += reward

            # Check for transition
            next_env = None
            if env_id in self.transitions and terminal_condition in self.transitions[
                    env_id]:
                next_env = self.transitions[env_id][terminal_condition]
                transition_message = self.transition_messages.get(
                    (env_id, next_env), "")
                logger.info(
                    f"Transitioning: {env_id} -> {next_env} | {transition_message}"
                )

            if next_env:
                self.current_environment = next_env
                transitions_made += 1
                logger.info(
                    f"Transitioned to new environment: {next_env}, transitions made: {transitions_made}"
                )
            else:
                # No valid transition found, evaluation is complete
                if terminal_condition == "error":
                    logger.error(f"Environment reported error: {final_state}")
                else:
                    logger.info(
                        f"No transition found for condition: {terminal_condition}"
                    )

                self.current_environment = None
                self.evaluation_complete = True
                logger.info(
                    "No more transitions available, marking evaluation as complete"
                )

        logger.info(
            f"Evaluation loop finished. Transitions made: {transitions_made}, evaluation complete: {self.evaluation_complete}"
        )
        logger.info(f"Environment sequence: {self.environment_sequence}")

        # Calculate aggregate reward (R_C)j
        self.aggregate_reward = self.calculate_aggregate_reward(
            self.environment_rewards)
        logger.info(f"Aggregate reward: {self.aggregate_reward}")

        return {
            "success": True,
            "message": "Evaluation complete"
            if self.evaluation_complete else "Evaluation in progress",
            "aggregate_reward": self.aggregate_reward,
            "environment_rewards": self.environment_rewards,
            "environments_visited": self.environment_sequence,
            "evaluation_complete": self.evaluation_complete
        }

    def run_environment(self,
                        env_id: str,
                        agent_class,
                        max_steps: int = 301) -> Tuple[float, str, str]:
        """Run a single environment episode"""
        logger.info(f"Attempting to run environment: {env_id}")
        print(f"Environment map: {self.environments}"
              )  # Print the environments map
        print(f"Looking up endpoint for {env_id}")
        observation_text = ""

        try:
            # Create environment stub
            env_stub = None
            if "_mfp" in env_id:
                env_stub = MARACompositeAutumnMFPServicer()
            elif "_cd" in env_id:
                env_stub = MARACompositeAutumnChangeDetectionServicer()
            elif "_planning" in env_id:
                env_stub = MARACompositeAutumnPlanningServicer()
            else:
                logger.error(f"No environment stub found for {env_id}")
                return 0.0, "error", f"No environment stub found for {env_id}"

            # try:
            #     logger.info("Initializing environment")
            # import pdb; pdb.set_trace();
            env_init = env_stub.Initialize(
                env_service_pb2.InitializeRequest(
                    env_type=env_pb2.REACTIVE,
                    config=self.configs.get(
                        env_id, {
                            "env_name":
                            "_".join(env_id.split("_")[:-2]),
                            "per_env_max_steps":
                            str(self.config.get("per_env_max_steps", 25)),
                            "stack_frames":
                            str(self.config.get("stack_frames", 0)),
                            "skip_frames":
                            str(self.config.get("skip_frames", False)),
                            "render_mode":
                            self.config.get("render_mode", "text"),
                            "logging_path":
                            self.config.get("logging_path", "./logs"),
                            "seed":
                            str(self.config.get("seed", 0)),
                            "data_dir":
                            self.config.get("data_dir", "./data")
                        })), None)
            logger.info(f"Environment initialized: {env_init.message}")
            # except grpc.RpcError as e:
            #     logger.error(f"Failed to initialize environment: {e}")
            #     return 0.0, "error", f"Initialization failed: {str(e)}"

            logger.info("Resetting environment")
            env_reset = env_stub.Reset(env_service_pb2.ResetRequest(), None)
            logger.info("Environment reset successfully")

            agent_stub = agent_class()
            # Initialize agent
            logger.info(f"Initializing agent with config: {self.config}")
            agent_init = agent_stub.Initialize(
                agent_pb2.AgentInitializeRequest(
                    config={
                        "llm_provider":
                        self.config.get("llm_provider", "openai"),
                        "llm_model":
                        self.config.get("llm_model", "openai/gpt-4o"),
                        "env_name":
                        "_".join(env_id.split("_")[:-2]),
                        "logging_path":
                        self.config.get("logging_path", "./logs"),
                        "max_history_length":
                        str(self.config.get("max_history_length", -1)),
                        "use_scratchpad":
                        str(self.config.get("use_scratchpad", False)),
                        "instruction_type":
                        self.config.get("instruction_type", "react"),
                        "hint":
                        str(self.config.get("hint", False)),
                        "task_name":
                        self.config.get("task_name", "mfp"),
                        "stack_frames":
                        str(self.config.get("stack_frames", 0)),
                        "skip_frames":
                        str(self.config.get("skip_frames", False)),
                        "render_mode":
                        self.config.get("render_mode", "text"),
                        "seed":
                        str(self.config.get("seed", 0)),
                        "data_dir":
                        self.config.get("data_dir", "./data"),
                        "use_oracle_interpreter_seed":
                        str(self.config.get("use_oracle_interpreter_seed")),
                    }), None)
            logger.info(
                f"\n\n[Controller] Agent initialized: {agent_init.message}, id: {agent_init.agent_id}"
            )

            # Reset agent
            agent_reset = agent_stub.Reset(
                agent_pb2.AgentResetRequest(
                    initial_observation=env_reset.initial_observation), None)
            logger.info("Agent reset successfully")

            # Run episode loop
            steps = 0
            total_reward = 0.0
            is_terminal = False
            current_observation = env_reset.initial_observation

            logger.info(f"Starting episode loop (max_steps={max_steps})")
            logger.info(
                f"Initial observation: {current_observation.text_data}")
            while not is_terminal and steps < max_steps:
                # Log step information
                logger.info(f"Step {steps + 1}/{max_steps}")
                # Query action space
                try:
                    space_query = env_service_pb2.SpaceQueryRequest()
                    space_query.reactive_query.SetInParent()
                    space_response = env_stub.QuerySpaces(space_query, None)
                    # logger.info(f"Action space: {space_response.reactive_response.action_space}")
                except grpc.RpcError as e:
                    logger.error(f"Failed to query action space: {e}")
                    break
                # Get agent action
                try:
                    act_response = agent_stub.Act(
                        agent_pb2.ActRequest(
                            observation=current_observation,
                            reactive_action_space=space_response.
                            reactive_response.action_space), None)
                    action_text = act_response.action.text_data if hasattr(
                        act_response.action, 'text_data') else "unknown"
                    logger.info(f"Agent action: {action_text}")
                except grpc.RpcError as e:
                    logger.error(f"Failed to get agent action: {e}")
                    break

                # Take environment step
                try:
                    step_response = env_stub.Step(
                        env_service_pb2.StepRequest(
                            action=act_response.action), None)
                    reward = step_response.reward
                    is_terminal = step_response.is_terminal
                    total_reward += reward
                    logger.info(
                        f"Step result: reward={reward}, terminal={is_terminal}"
                    )
                except grpc.RpcError as e:
                    logger.error(f"Failed to take environment step: {e}")
                    break

                # Store previous observation for feedback
                prev_observation = current_observation
                current_observation = step_response.observation
                logger.info(f"Observation: {current_observation.text_data}")

                # Provide feedback to agent
                try:
                    agent_stub.Feedback(
                        agent_pb2.FeedbackRequest(
                            previous_observation=prev_observation,
                            action=act_response.action,
                            current_observation=current_observation,
                            reward=reward,
                            is_terminal=is_terminal,
                            info=step_response.info), None)
                except grpc.RpcError as e:
                    logger.error(f"Failed to provide feedback to agent: {e}")

                steps += 1

                # Determine terminal condition
                if is_terminal:
                    terminal_condition = "default"
                    if "terminal_condition" in step_response.info:
                        terminal_condition = step_response.info[
                            "terminal_condition"]
                    else:
                        observation_text = current_observation.text_data.lower(
                        ) if hasattr(current_observation, 'text_data') else ""
                        if "quit" in observation_text:
                            terminal_condition = "quit"
                        elif "finish" in observation_text or "congratulations" in observation_text:
                            terminal_condition = "finish"
                        elif "fail" in observation_text:
                            terminal_condition = "fail"
                        else:
                            logger.warning(
                                f"Unknown terminal condition: {observation_text}"
                            )
                            # raise ValueError(f"Unknown terminal condition: {observation_text}")

            # Episode complete
            logger.info(
                f"Episode complete: steps={steps}, total_reward={total_reward}"
            )

            # Notify agent of episode completion
            try:
                agent_stub.EndEpisode(
                    agent_pb2.EndEpisodeRequest(total_reward=total_reward,
                                                num_steps=steps,
                                                success=total_reward > 0),
                    None)
                logger.info("Agent notified of episode completion")
            except grpc.RpcError as e:
                logger.error(f"Failed to notify agent of episode end: {e}")

            # Close environment
            try:
                env_stub.Close(env_service_pb2.CloseRequest(), None)
                logger.info("Environment closed successfully")
            except grpc.RpcError as e:
                logger.error(f"Failed to close environment: {e}")

            return total_reward, "default", observation_text

        except Exception as e:
            logger.exception(
                f"Unexpected error running environment {env_id}: {e}")
            return 0.0, "error", f"Unexpected error: {str(e)}"

    def get_state(self) -> Dict[str, Any]:
        """Get current evaluation state"""
        return {
            "current_environment_id": self.current_environment,
            "rewards_so_far": self.environment_rewards,
            "environment_sequence": self.environment_sequence,
            "aggregate_reward": self.aggregate_reward,
            "evaluation_complete": self.evaluation_complete
        }

    def transition(self, from_env: str, to_env: str,
                   transition_data: Any) -> Tuple[bool, str, str]:
        """Manual environment transition"""
        if from_env != self.current_environment:
            return False, "Specified from_environment does not match current environment", ""

        if to_env not in self.environments:
            return False, f"Specified to_environment {to_env} not found", ""

        # Perform transition (T function in the paper)
        self.current_environment = to_env
        message = self.transition_messages.get((from_env, to_env), "")

        return True, "Transition successful", message

    def calculate_aggregate_reward(
            self, environment_rewards: Dict[str, float]) -> float:
        """Calculate aggregate reward (R_C function in the paper)"""
        # Simple implementation: sum of rewards
        # A more complex implementation could use weights or other transformations
        return sum(environment_rewards.values())

    def get_transition_message(self, from_env: str, to_env: str) -> str:
        """Get transition message (O_C function in the paper)"""
        return self.transition_messages.get((from_env, to_env), "")


class EvaluationController:

    def __init__(self) -> None:
        self.environments: Dict[str,
                                str] = {}  # Map of environment_id to endpoint
        self.transitions: Dict[str, Dict[str, str]] = {
        }  # Map of from_env -> {condition: to_env}
        self.transition_messages: Dict[Any, Any] = {
        }  # Messages to show during transitions
        self.configs: Dict[str, Dict[str, str]] = {
        }  # Configuration for each environment
        self.current_environment: Optional[str] = None
        self.environment_sequence: List[str] = []
        self.environment_rewards: Dict[str, float] = {}
        self.aggregate_reward: float = 0.0
        self.evaluation_complete: bool = False
        self.environment_ids: List[str] = []

    def initialize(self, environment_ids: List[str], transitions: List[Any],
                   agent_id: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        print("Initializing controller")
        print(f"Environment IDs: {environment_ids}")
        self.environment_ids = environment_ids
        # Only use the text adventure environment on its actual port
        self.environments = {
            **{
                f"{eid}_interactive_mfp": f"localhost:{port}"
                for eid, port in zip(
                    environment_ids,
                    range(50050, 50050 + len(environment_ids) * 3, 3))
            },
            # **{f"{eid}_interactive_cd": f"localhost:{port}" for eid, port in zip(environment_ids, range(50051, 50051 + len(environment_ids)*3, 3))},
        }
        print(f"Environments: {environment_ids}")
        self.transitions = {
            **{
                f"{environment_ids[eid]}_interactive_mfp": {
                    "default": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "quit": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "finish": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "fail": f"{environment_ids[eid + 1]}_interactive_mfp",
                }
                for eid in range(len(environment_ids) - 1)
            },
        }

        self.transition_messages = {
            **{
                f"{environment_ids[eid]}_interactive_mfp": {
                    "default": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "quit": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "finish": f"{environment_ids[eid + 1]}_interactive_mfp",
                    "fail": f"{environment_ids[eid + 1]}_interactive_mfp",
                }
                for eid in range(len(environment_ids) - 1)
            },
        }

        # self.transitions = {
        #     **{f"{eid}_interactive_mfp": {
        #         "default": f"{eid}_interactive_cd",
        #         "quit": f"{eid}_interactive_cd",
        #         "finish": f"{eid}_interactive_cd",
        #         "fail": f"{eid}_interactive_cd",
        #     } for eid in environment_ids},
        #     **{f"{environment_ids[eid]}_interactive_cd": {
        #         "default": f"{environment_ids[eid + 1]}_interactive_mfp",
        #         "quit": f"{environment_ids[eid + 1]}_interactive_mfp",
        #         "finish": f"{environment_ids[eid + 1]}_interactive_mfp",
        #         "fail": f"{environment_ids[eid + 1]}_interactive_mfp",
        #     } for eid in range(len(environment_ids) - 1)},
        # }
        # self.transition_messages = {
        #     **{f"{eid}_interactive_mfp": {
        #         "default": f"{eid}_interactive_cd",
        #         "quit": f"{eid}_interactive_cd",
        #         "finish": f"{eid}_interactive_cd",
        #         "fail": f"{eid}_interactive_cd",
        #     } for eid in environment_ids},
        #     **{f"{environment_ids[eid]}_interactive_cd": {
        #         "default": f"{environment_ids[eid]}_interactive_mfp",
        #         "quit": f"{environment_ids[eid]}_interactive_mfp",
        #         "finish": f"{environment_ids[eid]}_interactive_mfp",
        #         "fail": f"{environment_ids[eid]}_interactive_mfp",
        #     } for eid in range(len(environment_ids) - 1)},
        # }
        self.agents = {
            "autumn_random_interactive_agent_v1": "localhost:50253",
            "autumn_llm_interactive_agent_v1": "localhost:50254"
        }

        # Initialize rewards for environments that exist
        for env_id in self.environments:
            self.environment_rewards[env_id] = 0.0

        # Store available transitions (only those involving available environments)
        for transition in transitions:
            from_env = transition.from_environment_id
            to_env = transition.to_environment_id

            # Skip transitions for non-existent environments
            if from_env not in self.environments or to_env not in self.environments:
                logger.warning(
                    f"Skipping transition {from_env} -> {to_env} (environment not available)"
                )
                continue

            condition = transition.transition_condition
            message = transition.transition_message

            if from_env not in self.transitions:
                self.transitions[from_env] = {}

            self.transitions[from_env][condition] = to_env
            self.transition_messages[(from_env, to_env)] = message

        # Reset state - use the first available environment
        available_envs = list(self.environments.keys())
        self.current_environment = available_envs[0] if available_envs else None
        self.environment_sequence = []
        self.aggregate_reward = 0.0
        self.evaluation_complete = False
        self.config = config
        return True, f"Initialized controller with available environments: {list(self.environments.keys())}, transitions: {self.transitions.keys()}, agent: {agent_id}"

    def run_evaluation(self,
                       reset: bool = False,
                       max_transitions: int = 10) -> Dict[str, Any]:
        """Run full evaluation across multiple environments"""
        logger.info(
            f"Starting evaluation with reset={reset}, max_transitions={max_transitions}"
        )

        if reset:
            self.environment_sequence = []
            self.environment_rewards = {
                env_id: 0.0
                for env_id in self.environments
            }
            self.aggregate_reward = 0.0
            self.evaluation_complete = False
            first_env_id = next(iter(self.environments.keys()))
            self.current_environment = first_env_id
            logger.info(
                f"Reset evaluation state, starting with environment: {self.current_environment}"
            )

        transitions_made = 0

        logger.info(
            f"Current environment before loop: {self.current_environment}")
        logger.info(
            f"Loop conditions: transitions_made={transitions_made}, max_transitions={max_transitions}, evaluation_complete={self.evaluation_complete}"
        )

        if transitions_made < max_transitions and not self.evaluation_complete:
            logger.info("Entering main evaluation loop")
        else:
            logger.warning(
                "Loop conditions not met, skipping main evaluation loop")

        while transitions_made < max_transitions and not self.evaluation_complete:
            # Run current environment
            if not self.current_environment:
                logger.warning(
                    "No current environment, marking evaluation as complete")
                self.evaluation_complete = True
                break

            print(f"Current environment: {self.current_environment}")
            env_id = self.current_environment
            logger.info(f"Adding environment to sequence: {env_id}")
            self.environment_sequence.append(env_id)

            logger.info(
                f"Running environment: {env_id}, agent: {self.config.get('agent', 'autumn_llm_interactive_agent_v1')}"
            )
            reward, terminal_condition, final_state = self.run_environment(
                env_id, self.agents[self.config.get(
                    "agent", "autumn_llm_interactive_agent_v1")])
            logger.info(
                f"Environment run complete. Reward: {reward}, Terminal condition: {terminal_condition}"
            )

            # Store reward
            self.environment_rewards[env_id] += reward

            # Check for transition
            next_env = None
            if env_id in self.transitions and terminal_condition in self.transitions[
                    env_id]:
                next_env = self.transitions[env_id][terminal_condition]
                transition_message = self.transition_messages.get(
                    (env_id, next_env), "")
                logger.info(
                    f"Transitioning: {env_id} -> {next_env} | {transition_message}"
                )

            if next_env:
                self.current_environment = next_env
                transitions_made += 1
                logger.info(
                    f"Transitioned to new environment: {next_env}, transitions made: {transitions_made}"
                )
            else:
                # No valid transition found, evaluation is complete
                if terminal_condition == "error":
                    logger.error(f"Environment reported error: {final_state}")
                else:
                    logger.info(
                        f"No transition found for condition: {terminal_condition}"
                    )

                self.current_environment = None
                self.evaluation_complete = True
                logger.info(
                    "No more transitions available, marking evaluation as complete"
                )

        logger.info(
            f"Evaluation loop finished. Transitions made: {transitions_made}, evaluation complete: {self.evaluation_complete}"
        )
        logger.info(f"Environment sequence: {self.environment_sequence}")

        # Calculate aggregate reward (R_C)j
        self.aggregate_reward = self.calculate_aggregate_reward(
            self.environment_rewards)
        logger.info(f"Aggregate reward: {self.aggregate_reward}")

        return {
            "success": True,
            "message": "Evaluation complete"
            if self.evaluation_complete else "Evaluation in progress",
            "aggregate_reward": self.aggregate_reward,
            "environment_rewards": self.environment_rewards,
            "environments_visited": self.environment_sequence,
            "evaluation_complete": self.evaluation_complete
        }

    def run_environment(self,
                        env_id: str,
                        agent_endpoint: str,
                        max_steps: int = 201) -> Tuple[float, str, str]:
        """Run a single environment episode"""
        logger.info(f"Attempting to run environment: {env_id}")
        print(f"Environment map: {self.environments}"
              )  # Print the environments map
        print(f"Looking up endpoint for {env_id}")
        observation_text = ""

        try:
            # Connect to environment
            env_endpoint = self.environments.get(env_id)
            print(f"Found endpoint: {env_endpoint}")  # Print found endpoint

            # Create environment stub
            env_stub = None
            if "_mfp" in env_id:
                env_stub = MARACompositeAutumnMFPServicer()
            elif "_cd" in env_id:
                env_stub = MARACompositeAutumnChangeDetectionServicer()
            elif "_planning" in env_id:
                env_stub = MARACompositeAutumnPlanningServicer()
            else:
                logger.error(f"No environment stub found for {env_id}")
                return 0.0, "error", f"No environment stub found for {env_id}"

            server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            env_grpc.add_MARAEnvironmentServicer_to_server(env_stub, server)
            port = self.environments[env_id].split(":")[1]
            server.add_insecure_port(f'[::]:{port}')
            server.start()
            print(f"Environment server started on port {port}")
            server_is_running = False
            while not server_is_running:
                try:
                    logger.info(f"Connecting to environment at {env_endpoint}")
                    time.sleep(1)
                    env_channel = grpc.insecure_channel(env_endpoint)
                    env_stub = env_grpc.MARAEnvironmentStub(env_channel)
                    server_is_running = True
                except Exception as e:
                    print("Waiting for server to start, retrying...")
                    time.sleep(1)

            if not env_endpoint:
                logger.error(
                    f"No endpoint configured for environment {env_id}")
                return 0.0, "error", f"No endpoint for {env_id}"

            try:
                logger.info("Initializing environment")
                env_init = env_stub.Initialize(
                    env_service_pb2.InitializeRequest(
                        env_type=env_pb2.REACTIVE,
                        config=self.configs.get(
                            env_id, {
                                "env_name": "_".join(env_id.split("_")[:-2]),
                                "per_env_max_steps": "100"
                            })),
                    timeout=10)
                logger.info(f"Environment initialized: {env_init.message}")
            except grpc.RpcError as e:
                logger.error(f"Failed to initialize environment: {e}")
                return 0.0, "error", f"Initialization failed: {str(e)}"

            # Reset environment
            try:
                logger.info("Resetting environment")
                env_reset = env_stub.Reset(env_service_pb2.ResetRequest())
                logger.info("Environment reset successfully")
            except grpc.RpcError as e:
                logger.error(f"Failed to reset environment: {e}")
                return 0.0, "error", f"Reset failed: {str(e)}"

            # Connect to agent
            try:
                logger.info(f"Connecting to agent at {agent_endpoint}")
                agent_channel = grpc.insecure_channel(agent_endpoint)
                agent_stub = agent_grpc.MARAAgentStub(agent_channel)
                # Initialize agent
                logger.info(
                    f"Initializing agent with config herhehre: {self.config}")
                agent_init = agent_stub.Initialize(
                    agent_pb2.AgentInitializeRequest(
                        config={
                            "llm_provider":
                            self.config.get("llm_provider", "openai"),
                            "llm_model":
                            self.config.get("llm_model", "openai/gpt-4o")
                        }))
                logger.info(
                    f"\n\n[Controller] Agent initialized: {agent_init.message}, id: {agent_init.agent_id}"
                )

                # Reset agent
                agent_reset = agent_stub.Reset(
                    agent_pb2.AgentResetRequest(
                        initial_observation=env_reset.initial_observation))
                logger.info("Agent reset successfully")
            except grpc.RpcError as e:
                logger.error(f"Failed to initialize or reset agent: {e}")
                return 0.0, "error", f"Agent initialization failed: {str(e)}"

            # Run episode loop
            steps = 0
            total_reward = 0.0
            is_terminal = False
            current_observation = env_reset.initial_observation

            logger.info(f"Starting episode loop (max_steps={max_steps})")
            logger.info(
                f"Initial observation: {current_observation.text_data}")
            while not is_terminal and steps < max_steps:
                # Log step information
                logger.info(f"Step {steps + 1}/{max_steps}")
                # Query action space
                try:
                    space_query = env_service_pb2.SpaceQueryRequest()
                    space_query.reactive_query.SetInParent()
                    space_response = env_stub.QuerySpaces(space_query)
                except grpc.RpcError as e:
                    logger.error(f"Failed to query action space: {e}")
                    break

                # Get agent action
                try:
                    act_response = agent_stub.Act(
                        agent_pb2.ActRequest(
                            observation=current_observation,
                            reactive_action_space=space_response.
                            reactive_response.action_space))
                    action_text = act_response.action.text_data if hasattr(
                        act_response.action, 'text_data') else "unknown"
                    logger.info(f"Agent action: {action_text}")
                except grpc.RpcError as e:
                    logger.error(f"Failed to get agent action: {e}")
                    break

                # Take environment step
                try:
                    step_response = env_stub.Step(
                        env_service_pb2.StepRequest(
                            action=act_response.action))
                    reward = step_response.reward
                    is_terminal = step_response.is_terminal
                    total_reward += reward
                    logger.info(
                        f"Step result: reward={reward}, terminal={is_terminal}"
                    )
                except grpc.RpcError as e:
                    logger.error(f"Failed to take environment step: {e}")
                    break

                # Store previous observation for feedback
                prev_observation = current_observation
                current_observation = step_response.observation
                logger.info(f"Observation: {current_observation.text_data}")

                # Provide feedback to agent
                try:
                    agent_stub.Feedback(
                        agent_pb2.FeedbackRequest(
                            previous_observation=prev_observation,
                            action=act_response.action,
                            current_observation=current_observation,
                            reward=reward,
                            is_terminal=is_terminal,
                            info=step_response.info))
                except grpc.RpcError as e:
                    logger.error(f"Failed to provide feedback to agent: {e}")

                steps += 1

                # Determine terminal condition
                if is_terminal:
                    terminal_condition = "default"
                    if "terminal_condition" in step_response.info:
                        terminal_condition = step_response.info[
                            "terminal_condition"]
                    else:
                        observation_text = current_observation.text_data.lower(
                        ) if hasattr(current_observation, 'text_data') else ""
                        if "quit" in observation_text:
                            terminal_condition = "quit"
                        elif "finish" in observation_text or "congratulations" in observation_text:
                            terminal_condition = "finish"
                        elif "fail" in observation_text:
                            terminal_condition = "fail"
                        else:
                            logger.warning(
                                f"Unknown terminal condition: {observation_text}"
                            )
                            # raise ValueError(f"Unknown terminal condition: {observation_text}")

            # Episode complete
            logger.info(
                f"Episode complete: steps={steps}, total_reward={total_reward}"
            )

            # Notify agent of episode completion
            try:
                agent_stub.EndEpisode(
                    agent_pb2.EndEpisodeRequest(total_reward=total_reward,
                                                num_steps=steps,
                                                success=total_reward > 0))
                logger.info("Agent notified of episode completion")
            except grpc.RpcError as e:
                logger.error(f"Failed to notify agent of episode end: {e}")

            # Close environment
            try:
                env_stub.Close(env_service_pb2.CloseRequest())
                logger.info("Environment closed successfully")
            except grpc.RpcError as e:
                logger.error(f"Failed to close environment: {e}")

            return total_reward, "default", observation_text

        except Exception as e:
            logger.exception(
                f"Unexpected error running environment {env_id}: {e}")
            return 0.0, "error", f"Unexpected error: {str(e)}"

    def get_state(self) -> Dict[str, Any]:
        """Get current evaluation state"""
        return {
            "current_environment_id": self.current_environment,
            "rewards_so_far": self.environment_rewards,
            "environment_sequence": self.environment_sequence,
            "aggregate_reward": self.aggregate_reward,
            "evaluation_complete": self.evaluation_complete
        }

    def transition(self, from_env: str, to_env: str,
                   transition_data: Any) -> Tuple[bool, str, str]:
        """Manual environment transition"""
        if from_env != self.current_environment:
            return False, "Specified from_environment does not match current environment", ""

        if to_env not in self.environments:
            return False, f"Specified to_environment {to_env} not found", ""

        # Perform transition (T function in the paper)
        self.current_environment = to_env
        message = self.transition_messages.get((from_env, to_env), "")

        return True, "Transition successful", message

    def calculate_aggregate_reward(
            self, environment_rewards: Dict[str, float]) -> float:
        """Calculate aggregate reward (R_C function in the paper)"""
        # Simple implementation: sum of rewards
        # A more complex implementation could use weights or other transformations
        return sum(environment_rewards.values())

    def get_transition_message(self, from_env: str, to_env: str) -> str:
        """Get transition message (O_C function in the paper)"""
        return self.transition_messages.get((from_env, to_env), "")


class MARAEvaluationControllerServicer(
        controller_grpc.MARAEvaluationControllerServicer):

    def __init__(self) -> None:
        self.controllers = {}  # Map of controller_id to EvaluationController
        self.next_controller_id = 1

    def Initialize(self, request: Any, context: Any) -> Any:
        controller = EvaluationController()
        success, message = controller.initialize(request.environment_ids,
                                                 request.transitions,
                                                 request.agent_id,
                                                 request.config)

        controller_id = f"controller_{self.next_controller_id}"
        self.next_controller_id += 1
        self.controllers[controller_id] = controller
        logger.info(f"[Controller] Initialized controller {controller_id}")

        return controller_pb2.ControllerInitializeResponse(
            success=success, controller_id=controller_id, message=message)

    def RunEvaluation(self, request: Any, context: Any) -> Any:
        if request.controller_id not in self.controllers:
            return controller_pb2.RunEvaluationResponse(
                success=False,
                message=f"Controller {request.controller_id} not found")

        controller = self.controllers[request.controller_id]
        logger.info(
            f"[Controller] Running evaluation for controller {request.controller_id}"
        )
        result = controller.run_evaluation(
            reset=request.reset, max_transitions=request.max_transitions)

        return controller_pb2.RunEvaluationResponse(
            success=result["success"],
            message=result["message"],
            aggregate_reward=result["aggregate_reward"],
            environment_rewards=result["environment_rewards"],
            environments_visited=result["environments_visited"],
            evaluation_complete=result["evaluation_complete"])

    def GetEvaluationState(self, request: Any, context: Any) -> Any:
        if request.controller_id not in self.controllers:
            # Return empty state with error flag
            return controller_pb2.EvaluationStateResponse(
                current_environment_id="", evaluation_complete=True)

        controller = self.controllers[request.controller_id]
        state = controller.get_state()

        return controller_pb2.EvaluationStateResponse(
            current_environment_id=state["current_environment_id"],
            rewards_so_far=state["rewards_so_far"],
            environment_sequence=state["environment_sequence"],
            aggregate_reward=state["aggregate_reward"],
            evaluation_complete=state["evaluation_complete"])

    def TransitionEnvironment(self, request: Any, context: Any) -> Any:
        if request.controller_id not in self.controllers:
            return controller_pb2.TransitionResponse(
                success=False,
                message=f"Controller {request.controller_id} not found")

        controller = self.controllers[request.controller_id]
        success, message, transition_message = controller.transition(
            request.from_environment_id, request.to_environment_id,
            request.transition_data)

        return controller_pb2.TransitionResponse(
            success=success,
            message=message,
            transition_message=transition_message)

    def CalculateRewards(self, request: Any, context: Any) -> Any:
        if request.controller_id not in self.controllers:
            return controller_pb2.CalculateRewardsResponse(
                aggregate_reward=0.0)

        controller = self.controllers[request.controller_id]
        aggregate = controller.calculate_aggregate_reward(
            request.environment_rewards)

        return controller_pb2.CalculateRewardsResponse(
            aggregate_reward=aggregate)


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controller_grpc.add_MARAEvaluationControllerServicer_to_server(
        MARAEvaluationControllerServicer(), server)

    port = 51059
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Evaluation Controller server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
