import logging
import random
import grpc
import numpy as np
import json
from typing import Any, Dict, List, Union, Optional
from abc import abstractmethod
import yaml
from collections import deque
import itertools

# Assume protobuf-generated files are accessible in the Python path.
from generated.mara import mara_agent_pb2 as agent_pb2
from generated.mara import mara_agent_pb2_grpc as agent_grpc
from generated.mara import mara_environment_pb2 as env_pb2
from .concrete_envs import InteractiveEnvironment
from .interpreter_module import Interpreter
from .autumnstdlib import autumnstdlib
from .env_utils import load_yaml_to_dict
from .concrete_envs import convert_to_grid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

Observation = np.ndarray[str]
Action = str


class SimpleWMAgent:
    """
    Implements the core logic for the SimpleWMAgent.
    This version includes an explicit phase attribute to switch from an
    'interaction' phase to an 'evaluation' phase.
    """

    def __init__(self, config: dict):
        """Initializes the agent with configuration and sets up the initial state."""
        self.config = config
        self.task_name = self.config.get("task_name", "mfp")
        self.num_samples_mc = int(self.config.get("num_samples_mc", 1))
        self.history = {}
        self.phase = 'interaction'  # Initialize phase
        self.reset(None)

    def reset(self, initial_observation: env_pb2.Observation):
        """Resets the agent's state for a new episode."""
        self.history = {"observations": [], "actions": []}
        self.phase = 'interaction'  # Reset phase to interactive for each new episode
        logger.info(f"Agent reset. Phase set to '{self.phase}'.")
        return True

    def _sample_random_action(self,
                              available_actions: list[env_pb2.Action],
                              exclude_giveup: bool = False,
                              exclude_found_fault: bool = False,
                              exclude_submit: bool = False) -> env_pb2.Action:
        """Takes a random action. Used for the interactive phase of all tasks."""
        new_available_actions = []
        for action in available_actions:
            if exclude_giveup and action.text_data == "quit":
                continue
            if exclude_found_fault and action.text_data == "I found the fault!":
                continue
            if exclude_submit and action.text_data == "submit":
                continue
            new_available_actions.append(action)
        available_actions = new_available_actions
        
        random_action = random.choice(available_actions)
        action_str = random_action.text_data

        # Handle specific action formats like 'click' with coordinates
        if 'click' in action_str and len(action_str.split(' ')) == 3:
            try:
                _, x_range, y_range = action_str.split(' ')
                x_min, x_max = map(int, x_range.strip('[]').split('-'))
                y_min, y_max = map(int, y_range.strip('[]').split('-'))
                return env_pb2.Action(
                    text_data=
                    f"click {random.randint(x_min, x_max)} {random.randint(y_min, y_max)}"
                )
            except (ValueError, IndexError):
                return random_action  # Fallback if parsing fails
        return random_action
    
    def _expand_actions(self, available_actions: List[env_pb2.Action],
                        exclude_quit: bool = False,
                        exclude_submit: bool = False
                        ) -> List[env_pb2.Action]:
        """Expend the actions that contain click with range to all possible
        actions."""
        expanded_actions = []
        for action in available_actions:
            if exclude_quit and action.text_data == "quit":
                continue
            if exclude_submit and action.text_data == "submit":
                continue
            action_str = action.text_data
            if 'click' in action_str and len(action_str.split(' ')) == 3:
                try:
                    _, x_range, y_range = action_str.split(' ')
                    x_min, x_max = map(int, x_range.strip('[]').split('-'))
                    y_min, y_max = map(int, y_range.strip('[]').split('-'))
                    for x in range(x_min, x_max + 1):
                        for y in range(y_min, y_max + 1):
                            expanded_actions.append(
                                env_pb2.Action(text_data=f"click {x} {y}"))
                except (ValueError, IndexError):
                    expanded_actions.append(action)
            else:
                expanded_actions.append(action)
        return expanded_actions

    def _act_mfp_eval(
            self, obs: Dict[str, Any],
            available_actions: list[env_pb2.Action]) -> env_pb2.Action:
        """Selects an MFP choice using the specified Monte Carlo strategy."""
        if len(available_actions) == 1:
            # Not at the stage of making a choice yet, save the action and
            # store the observation
            self.history["observations"].append(obs['obs_state'])
            if obs['video_location'] != 0:
                self.history["actions"].append(obs['action_took'])
            return self._sample_random_action(available_actions)
        else:
            # Draw N samples from our world model
            # Keep the ones that are consistent with the real observations
            # Take the most common choice among the consistent samples
            # If no consistent samples, fallback to random action
            self.history["observations"].append(obs['obs_state'])
            self.history["actions"].append(obs['action_took'])
            final_states = []
            for _ in range(self.num_samples_mc):
                final_state = self._simulate_for_consistent_final_state(
                    self.history["observations"], self.history["actions"])
                final_states.append(final_state)
            choice = self._make_mfp_choice(final_states, obs['obs_state'],
                                           obs['options'])
            return env_pb2.Action(text_data=f"choose_option_{choice}")

    def _act_cd_eval(
            self, obs: Observation,
            available_actions: list[env_pb2.Action]) -> env_pb2.Action:
        """Acts in the CD evaluation phase.
        A simpler solver for defect detection. It compars the received 
        observation with the observation predicted from its world model, and 
        declare defect is here if they differ.
        TODO: Take the potential stochasticity into account and compute the
        surprisal score of the current observation.

        If the agent has found a defect, it will report it.
        Otherwise, it will act randomly, excluding the 'give up' action.
        """
        # If it's the first observation, then skip the check
        if self.history["actions"]:
            # If the agent has already taken an action, check for defects
            # based on the previous observation and action.
            found_defect = self._check_for_defect(obs, self.history["actions"])
        else:
            found_defect = False

        # If the agent thinks it has found a defect and the `found fault` action
        # is available, which is only available after a few interactions, return
        # the action to report it.
        if found_defect:
            if any(action.text_data == "I found the fault!"
                   for action in available_actions):
                return env_pb2.Action(text_data="I found the fault!")

        # Otherwise, act randomly, excluding the 'give up' action.
        action = self._sample_random_action(available_actions,
                                            exclude_giveup=True,
                                            exclude_found_fault=True)
        self.history["actions"].append(action.text_data)
        return action

    @abstractmethod
    def _act_planning_eval(
            self, goal: np.ndarray[str],
            mask: np.ndarray[int],
            render: np.ndarray[str],
            available_actions: list[env_pb2.Action]) -> env_pb2.Action:
        raise NotImplementedError(
            "This method should be implemented in subclasses for planning "
            "tasks."
        )

    def act(self, observation: env_pb2.Observation,
            available_actions: list[env_pb2.Action]) -> env_pb2.Action:
        """Determines the agent's action based on its current phase."""
        logging.info(f"available_action: {available_actions}")
        # self.history['observations'].append(observation)

        # Check for phase transition trigger
        if self.phase == 'interaction' and (
                "The interaction phase is now over." in observation.text_data
                or "Interactive environment ended" in observation.text_data):
            self.phase = 'evaluation'
            logger.info(f"Phase transition triggered in agent. New phase: "
                        f"'{self.phase}'.")
            # Choose the default action to enter the evaluation phase
            if self.task_name == 'mfp':
                start = observation.text_data.find('{')
                observation.text_data = observation.text_data[start:]
            else:
                return self._sample_random_action(available_actions)

        # Dispatch action based on the current phase
        if self.phase == 'interaction':
            action = self._sample_random_action(available_actions)

        elif self.phase == 'evaluation':
            if self.task_name == 'mfp':
                obs_dict = parse_text_obs_to_dict(observation.text_data)
                action = self._act_mfp_eval(obs_dict, available_actions)
            elif self.task_name == 'dd':
                # If the option `The fault is here!` is available, it means
                # the agent has found the previous observation to be a defect.
                # So it would choose `The fault is here!` action.
                fault_is_here_available = any(
                            action.text_data == "The fault is here!"
                        for action in available_actions)
                if fault_is_here_available:
                    return env_pb2.Action(text_data="The fault is here!")

                obs_str = observation.text_data.strip('"\'').replace(
                    '\\n', '\n')
                obs_arr = _str_grid_to_ndarray(obs_str)
                action = self._act_cd_eval(obs_arr, available_actions)
            elif self.task_name == 'planning':
                data_dict = json.loads(observation.text_data)
                goal, mask, render = data_dict["goal"],\
                                data_dict["mask"], data_dict["render"]
                goal, mask, render = np.array(goal), np.array(mask), \
                                        _str_grid_to_ndarray(render)
                action = self._act_planning_eval(goal, mask, render, 
                                                 available_actions)
            else:
                logger.warning(
                    f"Unknown task '{self.task_name}' for evaluation phase. Acting randomly."
                )
                action = self._sample_random_action(available_actions)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        # self.history['actions'].append(action)
        return action

    # --- Placeholder Methods for World Model ---
    @abstractmethod
    def _simulate_for_consistent_final_state(
            self, observations: List[Observation],
            actions: List[Action]) -> Observation:
        """**Placeholder**: Simulates a trajectory using the world model."""
        raise NotImplementedError

    @abstractmethod
    def _check_for_defect(self, next_obs: Observation, 
                          action_history: List[Action],
                          num_samples: int = 10) -> bool:
        """**Placeholder**: Checks if the current observation indicates a defect."""
        raise NotImplementedError

    def _make_mfp_choice(self, sampled_trajectories: List[Observation],
                         observations: Observation,
                         choices: List[Observation]) -> int:
        num_matching = [0] * len(choices)
        masked_area = observations == 'transparent'
        for sampled_obs in sampled_trajectories:
            for i, choice_content in enumerate(choices):
                match = sampled_obs[masked_area] == choice_content.flatten()
                if match.all():
                    num_matching[i] += 1
        return np.argmax(num_matching)

    def _check_consistency(self, sampled_trajectory, real_observations):
        """**Placeholder**: Checks if a sampled trajectory is consistent with reality."""
        return sampled_trajectory == real_observations

    def _are_obs_content_equal(self, obs_object, choice_content):
        """**Placeholder**: Compares a final state with an MFP choice."""
        return False


class OracleAutumnSynthAgent(SimpleWMAgent):

    def __init__(self, config: dict):
        """Initializes the agent with configuration and sets up the initial 
        state. This implements an AutumnSynth agent with a GT world model.
        """
        super().__init__(config)
        self.wm = Interpreter()
        data_dir = config.get("data_dir", "data")
        env_name = config.get("env_name")
        self.prog = open(f"{data_dir}/programs/{env_name}.sexp", "r").read()
        task = config.get("task_name")
        self.experiment_seed = int(config.get("seed"))
        random.seed(self.experiment_seed)
        if str(config.get("use_oracle_interpreter_seed")).lower() == 'true':
            if task == "mfp":
                self.interpreter_seed = yaml.safe_load(
                open(f"{data_dir}/tasks/{env_name}_mfp.yaml")
                )["seed"]
            else:
                self.interpreter_seed = int(config.get("seed"))
        else:
            self.interpreter_seed = None
        logger.debug(f"Interpreter seed: {self.interpreter_seed}")
        self.colors: Dict[int, str] = load_yaml_to_dict(
            f"{data_dir}/color_dict.yaml")
        self.colors_str_to_int = {v: k for k, v in self.colors.items()}

        # For planning
        self.current_plan: list[str] = []          # cache the open‑loop plan
        self.max_depth   = int(config.get("planning_horizon", 6))
        self.max_branch  = int(config.get("planning_branching", 6))

    def _simulate_from_start_given_actions(self, actions: List[Action], 
                    interpreter_seed: Optional[int] = None,
                    reference_observations: Optional[List[Observation]] = None,
                    num_of_retries: int = 1,
                    log_actions: bool = True
                    ) -> List[Observation]:
        """Simulates a trajectory using the world model given a list of actions.
        If interpreter_seed is None, will use a random seed.
        if reference_observations is not None, we will only return observations
        that match the reference observations."""
        if interpreter_seed is None: 
            draw_random_interpreter_seed = True
        else:
            draw_random_interpreter_seed = False

        for _ in range(num_of_retries):
            if draw_random_interpreter_seed:
                # This is seeded by experiment_seed
                interpreter_seed = random.randint(0, 2**31 - 1)
            matching_reference_obs = True
            self.wm = Interpreter()
            self.wm.run_script(self.prog, autumnstdlib, "", interpreter_seed)
            # To deal with: https://github.com/BasisResearch/AutumnBenchmark/issues/7
            obs_json = self.wm.render_all()

            sampled_observations = []
            for i, action in enumerate(actions):
                # Get the next observation from the world model
                if log_actions:
                    logging.info(f"Action: {i}: {action}, ")
                # Take the action in the world model
                if action == "left":
                    self.wm.left()
                elif action == "right":
                    self.wm.right()
                elif action == "up":
                    self.wm.up()
                elif action == "down":
                    self.wm.down()
                elif action in ["noop", "noop"]:
                    pass
                elif "click" in action:
                    _, x, y = action.split()
                    self.wm.click(int(x), int(y))
                else:
                    raise ValueError(
                        f"Unrecognized action: {action}. "
                        "Please check the action format."
                    )
                self.wm.step()

                # Get observation from the world model
                obs_json = self.wm.render_all()
                if isinstance(obs_json, str):
                    try:
                        obs_json = json.loads(obs_json)
                    except:
                        print(f"Warning: Could not parse observation JSON: "
                            f"{obs_json}"
                        )
                        continue
                grid_size = obs_json.get("GRID_SIZE", 16)
                if hasattr(self.wm, 'get_background') and \
                    self.wm.get_background() in self.colors_str_to_int:
                    # Use the background color from the interpreter if available
                    background_color = self.colors_str_to_int[
                        self.wm.get_background()]
                else:
                    # Default background color if not in colors
                    background_color = 1
                obs_grid = convert_to_grid(obs_json,
                                            background_color,
                                            self.colors_str_to_int,
                                            grid_size,
                                            use_color_str=True)
                obs_arr = np.array(obs_grid, dtype=str)
                sampled_observations.append(obs_arr)

                if reference_observations:
                    # If we have reference observations, check if the current
                    # observation matches the reference one
                    target_obs = reference_observations[i + 1]
                    mask = target_obs != 'transparent'
                    if not np.array_equal(obs_arr[mask], target_obs[mask]):
                        logging.info(
                            f"Observation {i+1} does not match reference. "
                            f"Restarting sequence."
                        )
                        matching_reference_obs = False
                        break
            # If we reach here, all observations match the reference ones
            if matching_reference_obs:
                return sampled_observations
        logging.warning(
            f"Failed to simulate a consistent trajectory after {num_of_retries}"
             " retries."
        )
        return sampled_observations

    def _simulate_for_consistent_final_state(
            self,
            observations: List[Observation],
            actions: List[Action]) -> Observation:
        """Simulates a trajectory using the world model and returns the final
        state if it matches the real observations.
        This method will keep trying until it finds a consistent trajectory.
        """
        return self._simulate_from_start_given_actions(
            actions, interpreter_seed=self.interpreter_seed,
            reference_observations=observations)[-1]

    def _check_for_defect(self, next_obs: Observation, 
                          action_history: List[Action],
                          num_samples: int = 1) -> bool:
        """Checks if the current observation indicates a defect.
        It is if it's different from the prediction of the world model"""
        possible_frames: List[Observation] = []
        for _ in range(num_samples):
            # TODO: could also add exisitng observations as reference
            possible_frame = self._simulate_from_start_given_actions(
                action_history,
                self.interpreter_seed)
            possible_frames.append(possible_frame[-1])
        found_defect = not any(
            np.array_equal(next_obs, pred_obs)
            for pred_obs in possible_frames
        )
        return found_defect
    
    def _act_planning_eval(
        self,
        goal: np.ndarray[str],
        mask: np.ndarray[int],
        render: np.ndarray[str],
        available_actions: list[env_pb2.Action],
    ) -> env_pb2.Action:
        """
        One‑step interface required by the MARA planning evaluation protocol.

        • When the cached open‑loop plan ``self.current_plan`` still has steps
          left, just pop the next action and return it.

        • Otherwise, perform a bounded breadth‑first search (see helper above)
          from the start state using the ground‑truth interpreter as the
          transition model, store the resulting sequence, and execute the first
          step immediately.

        • If no plan can be found within the given limits, fall back to a
          uniformly random legal action (excluding 'quit').
        """
        # ---------- if reached the goal, return submit action -----------
        mask_bool = mask.astype(bool)
        if np.array_equal(render[mask_bool], goal[mask_bool]):
            logger.info("Goal reached! Submitting the solution.")
            return env_pb2.Action(text_data="submit")

        # ---------- fast path: still executing a previously‑found plan ----------
        if self.current_plan:
            next_step = self.current_plan.pop(0)
            return env_pb2.Action(text_data=next_step)
        

        # ---------- build the discrete action set we are willing to search -----
        all_actions = self._expand_actions(available_actions,
                                        exclude_quit=True,
                                        exclude_submit=True)

        # ---------- find a new plan (open‑loop) --------------------------------
        all_action_strs = [a.text_data for a in all_actions]
        plan = self._plan_to_goal(goal, mask_bool, render, 
                                  all_action_strs)
        logging.info(f"Found a plan: {plan}")

        if plan:
            self.current_plan = plan                # cache for future calls
            next_step = self.current_plan.pop(0)
            return env_pb2.Action(text_data=next_step)

        # ---------- graceful degradation: no plan found ------------------------
        logger.warning("Planning failed – falling back to random action.")
        # give up
        return env_pb2.Action(text_data="quit")

    # helper ----------------------------------------------------------------
    def _plan_to_goal(
        self,
        goal: np.ndarray[str],
        mask: np.ndarray[bool],
        render: np.ndarray[str],
        actions: list[Action],
    ) -> list[str] | None:
        """
        Breadth‑first search from the *initial* environment state.
        Because the world model can only simulate from t=0, every node holds the
        **entire prefix** that has to be replayed each time we want to expand it.
        This is expensive but completely general and safe for the small Autumn
        planning instances (typical branching ≤ 30 and depth ≤ 8).

        Returns the first (shortest) sequence whose final frame matches the goal
        inside the masked region, or ``None`` if nothing is found within
        ``self.max_depth``.
        """
        goal_slice = goal[mask]
        visited: dict[bytes, int] = {}           # state‑signature → best depth
        queue: deque[list[str]] = deque([[a] for a in actions])

        # A signature is just a byte representation of the *masked* cells; this
        # is tiny (≤ 256 bytes) and fast to hash / compare
        def _sig(state: np.ndarray[str]) -> bytes:
            return b" ".join(state.astype("S"))

        initial_sig = _sig(render)
        visited[initial_sig] = 0
        num_visited = 1

        while queue:
            logging.info(f"num_visited: {num_visited}, queue size: {len(queue)}")
            num_visited += 1
            prefix = queue.popleft()
            depth  = len(prefix)
            if depth > self.max_depth:
                continue

            # Re‑simulate from scratch (world model limitation)
            if depth == 0:
                final_state = render                # root node – already known
            else:
                sim_obs = self._simulate_from_start_given_actions(
                    prefix,
                    interpreter_seed=self.interpreter_seed,
                    log_actions=False,
                )
                if not sim_obs:
                    # world model failed to give us a frame; skip
                    continue
                final_state = sim_obs[-1]

            # Goal test on the **masked** region
            if np.array_equal(final_state[mask], goal_slice):
                return prefix                       # found a plan 🎉

            if depth == self.max_depth:
                continue                            # cannot expand further

            sig = _sig(final_state)
            if sig in visited and visited[sig] <= depth:
                continue                            # already reached sooner
            visited[sig] = depth

            # Expand – cap branching factor so search never explodes
            non_click_actions = actions[:4] + actions[-1:]
            click_actions = actions[4:-1]
            random.shuffle(click_actions)  # shuffle to avoid bias
            for a in itertools.islice(non_click_actions + click_actions, 
                                      self.max_branch):
                queue.append(prefix + [a])

        return None
    # --------------------------------------------------------------------

class SimpleWMAgentServicer(agent_grpc.MARAAgentServicer):
    """gRPC Servicer for the SimpleWMAgent."""

    def __init__(self):
        self.agent = None

    def Initialize(self, request, context):
        self.agent = OracleAutumnSynthAgent(request.config)
        logger.info(f"SimpleWMAgent initialized with config: {request.config}")
        return agent_pb2.AgentInitializeResponse(success=True,
                                                 agent_id="simple_wm_agent")

    def Reset(self, request, context):
        if not self.agent:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Agent not initialized.")
        self.agent.reset(request.initial_observation)
        return agent_pb2.AgentResetResponse(success=True)

    def Act(self, request, context):
        if not self.agent:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Agent not initialized.")
        available_actions = list(
            request.reactive_action_space.available_actions)
        if not available_actions:
            return agent_pb2.ActResponse(action=env_pb2.Action(
                text_data="noop"))
        action = self.agent.act(request.observation, available_actions)
        logger.info(f"SimpleWMAgent action: {action.text_data}")
        return agent_pb2.ActResponse(action=action)

    def Feedback(self, request, context):
        return agent_pb2.FeedbackResponse(acknowledged=True)

    def EndEpisode(self, request, context):
        return agent_pb2.EndEpisodeResponse(acknowledged=True)

    def Close(self, request, context):
        self.agent = None
        return agent_pb2.AgentCloseResponse(success=True)


def _str_grid_to_ndarray(grid_txt: str) -> np.ndarray:
    """
    Helper: turn the whitespace-separated, newline-delimited grid text
    into a 2-D NumPy array of dtype=str.
    """
    rows: List[List[str]] = [
        row.split() for row in grid_txt.strip().splitlines()
    ]
    arr = np.array(rows, dtype=str)
    return arr


def parse_text_obs_to_dict(obs_text: str) -> Dict[str, Any]:
    """
    Parse the JSON-encoded observation string coming from the environment.

    Parameters
    ----------
    obs_text : str
        JSON string like:
        '{"video_location": 1, "render": "...", "action_took": "...", ...}'

    Returns
    -------
    Dict[str, Any]
        A Python dict with:
        - all original scalar fields intact,
        - "state": np.ndarray[str] holding the render grid,
        - "options" converted to List[np.ndarray[str]] if present.
          (Original "render" key is removed.)
    """
    obs: Dict[str, Union[str, int, bool, List[str]]] = json.loads(obs_text)

    # Convert main render block → ndarray and store under "state"
    render_txt = obs.pop("render", None)
    if render_txt is not None:
        obs["obs_state"] = _str_grid_to_ndarray(render_txt)

    # If multiple option boards are supplied, convert each as well
    if "options" in obs and isinstance(obs["options"], list):
        obs["options"] = [_str_grid_to_ndarray(opt) for opt in obs["options"]]

    return obs
