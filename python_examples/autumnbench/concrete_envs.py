import os
import hashlib
import uuid
from typing import List, Tuple, Dict, Any, Union
import random

import json
import yaml
import time
import logging
import base64

from generated.mara import mara_environment_pb2 as env_pb2
from .interpreter_module import Interpreter
from .autumnstdlib import autumnstdlib
from .env_utils import (parse_grid, get_action_space_interactive,
                        interpreter_action_to_text, render_grid,
                        render_grid_matplotlib, render_string_grid_matplotlib,
                        load_yaml_to_dict, render_grid_to_matrix,
                        check_grid_same)

logger = logging.getLogger(__name__)
fh = logging.FileHandler("log_environment_interfaces.txt")
logger.addHandler(fh)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class InteractiveEnvironment:

    def __init__(self,
                 env_name,
                 parsed_in_id=None,
                 stack_frames=0,
                 skip_frames=False,
                 render_mode="text",
                 logging_path="./logs",
                 data_dir=CURR_DIR,
                 seed=0):
        self.prog = open(f"{data_dir}/programs/{env_name}.sexp", "r").read()
        self.is_terminal = False
        self.id = str(uuid.uuid4()) if parsed_in_id is None else parsed_in_id
        self.inited = False
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.render_mode = render_mode
        self.logging_path = logging_path
        self.seed = seed
        self.env_name = env_name

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, "", self.seed)
        self.inited = False
        self.time = 0
        if not os.path.exists(f"{self.logging_path}/{self.env_name}"):
            os.makedirs(f"{self.logging_path}/{self.env_name}")
        if not os.path.exists(
                f"{self.logging_path}/{self.env_name}/interactive"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/interactive")

    def get_action_space(self) -> List[env_pb2.Action]:
        _, grid_size = parse_grid(self.interpreter.render_all())
        return get_action_space_interactive(grid_size,
                                            self.time,
                                            truncated_action_space=True)

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        self.time += 1
        if action.text_data == "quit":
            self.is_terminal = True
            observation = self.get_observation()
            return observation, 0, self.is_terminal, {}
        else:
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}
            self.interpreter.step()
            observation = self.get_observation()
            if self.stack_frames > 0:
                if self.render_mode == "text":
                    stacked_frames = [observation.text_data]
                    for _ in range(self.stack_frames):
                        self.interpreter.step()
                        observation = self.get_observation()
                        stacked_frames.append(observation.text_data)
                    # if self.render_mode == "text":
                    if not self.skip_frames:
                        observation = "\n\nObservation: ".join(stacked_frames)
                        observation = env_pb2.Observation(
                            text_data=observation)
                    else:
                        observation = env_pb2.Observation(
                            text_data=
                            f"Skipped {self.stack_frames} frames. Current frame: {observation.text_data}"
                        )
                elif self.render_mode == "image":
                    stacked_frames = [observation.image_data]
                    for _ in range(self.stack_frames):
                        self.interpreter.step()
                        observation = self.get_observation()
                        stacked_frames.append(observation.image_data)
                    if not self.skip_frames:
                        observation = "\n\nObservation: ".join(stacked_frames)
                        observation = env_pb2.Observation(
                            image_data=observation.image_data)
                    else:
                        observation = env_pb2.Observation(
                            text_data=
                            f"Skipped {self.stack_frames} frames. Here is the current frame: ",
                            image_data=observation.image_data)
            logger.debug(f"Observation: {observation} for id: {self.id}")
            return observation, 0, self.is_terminal, {}

    def get_observation(self) -> env_pb2.Observation:
        if not self.inited:
            self.inited = True
            return env_pb2.Observation(
                text_data=
                "Welcome, you will now be playing an Autumn interactive environment."
                +
                "Remember what you see and how the environment works, as you will be queried about it later."
            )
        render_dict = json.loads(self.interpreter.render_all())
        render_img_str = render_grid_matplotlib(
            render_dict,
            output_path=
            f"{self.logging_path}/{self.env_name}/interactive/interactive_{self.time}.jpeg"
        )
        render_img_bytes = base64.b64decode(render_img_str)
        render_dict = render_grid(render_dict)
        if self.render_mode == "text":
            return env_pb2.Observation(text_data=json.dumps(render_dict))
        elif self.render_mode == "image":
            return env_pb2.Observation(text_data="Here is the current frame: ",
                                       image_data=render_img_bytes)
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def terminal(self):
        return self.is_terminal


class DefectDetectionEnvironment:

    def __init__(self, env_name):
        self.prog = open(f"{CURR_DIR}/modified_programs/{env_name}_wrong.sexp",
                         "r").read()
        self.event = open(
            f"{CURR_DIR}/modified_programs/{env_name}_condition.sexp",
            "r").read()
        self.is_terminal = False
        self.triggering_state = False
        self.trigger_start_time = None
        self.frames = []
        self.id = str(uuid.uuid4())
        self.inited = False

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, self.event)
        self.triggering_state = False
        self.trigger_start_time = None
        self.inited = False

    def get_action_space(self) -> List[env_pb2.Action]:
        _, grid_size = parse_grid(self.interpreter.render_all())
        action_space = get_action_space_interactive(grid_size)
        action_space.append(env_pb2.Action(text_data="Fault!"))
        action_space.append(env_pb2.Action(text_data="quit"))
        return action_space

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        logger.debug(f"Stepping with action: {action} for id: {self.id}")
        curr_state = self.triggering_state
        self.triggering_state |= self.interpreter.get_trigger_state()
        if curr_state != self.triggering_state:
            self.trigger_start_time = time.time()
        if action.text_data == "quit":
            self.is_terminal = True
            observation = self.get_observation()
            return observation, 0, self.is_terminal, {}
        if action.text_data == "Fault!":
            self.is_terminal = True
            observation = self.fault_observation()
            return observation, self.fault_reward(), self.is_terminal, {}
        else:
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}
            self.interpreter.step()
            interpreter_action_to_text(self.interpreter, action.text_data)
            self.triggering_state |= self.interpreter.get_trigger_state()
            observation = self.get_observation()
            logger.debug(f"Observation: {observation} for id: {self.id}")
            return observation, 0, self.is_terminal, {}

    def fault_observation(self) -> env_pb2.Observation:
        orig_observation = self.get_observation()
        text_data = json.loads(orig_observation.text_data)
        if self.triggering_state and self.trigger_start_time:
            return env_pb2.Observation(text_data=json.dumps(
                {
                    "render": text_data,
                    "fault": True,
                    "trigger_time": time.time() - self.trigger_start_time,
                    "detect_time": time.time() - self.trigger_start_time
                }))
        else:
            return env_pb2.Observation(text_data=json.dumps({
                "render": text_data,
                "fault": False,
                "trigger_time": 0,
                "detect_time": 0
            }))

    def fault_reward(self) -> float:
        if self.triggering_state and self.trigger_start_time:
            delta = time.time() - self.trigger_start_time
            return 1 / (1 + delta)
        return -1

    def get_observation(self) -> env_pb2.Observation:
        if not self.inited:
            self.inited = True
            return env_pb2.Observation(
                text_data=
                "Welcome, you will now be playing an Autumn defect detection environment."
                +
                "Remember what you see and how different the environment is from the normal Autumn environment."
                +
                "Once you have detected the fault, you have to click 'Fault!' to terminate the environment."
                +
                "You will be penalized if you click 'Fault!' before the fault is detected."
            )
        render_dict = json.loads(self.interpreter.render_all())
        render_dict = render_grid(render_dict)
        return env_pb2.Observation(text_data=json.dumps(render_dict))

    def terminal(self):
        return self.is_terminal


class DDSliderEnvironment:

    def __init__(self,
                 env_name,
                 render_mode="text",
                 stack_frames=0,
                 skip_frames=False,
                 logging_path="./logs",
                 data_dir=CURR_DIR,
                 seed=0):
        self.env_name = env_name
        self.data_dir = data_dir

        with open(f"{data_dir}/programs/{env_name}_wrong.sexp", "r") as f:
            self.prog = f.read()
        with open(f"{data_dir}/answers/{env_name}_condition.json", "r") as f:
            self.event = json.loads(f.read())["condition"]

        self.id = str(uuid.uuid4())
        self.render_mode = render_mode
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.logging_path = logging_path
        self.seed = seed
        self.last_interpreting_action = None
        self.reset()

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, self.event,
                                    self.seed)
        self.triggering_state = False
        self.trigger_start_time = None
        self.frames = []
        self.curr_frame = 0
        self.trigger_frame = -1
        self.inited = False
        self.time = 0
        self.is_terminal = False
        self.state = "interactive"
        self.inited = False
        if not os.path.exists(f"{self.logging_path}/{self.env_name}"):
            os.makedirs(f"{self.logging_path}/{self.env_name}")
        if not os.path.exists(f"{self.logging_path}/{self.env_name}/dd"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/dd")

    def get_action_space(self) -> List[env_pb2.Action]:
        if self.state == "interactive":
            _, grid_size = parse_grid(self.interpreter.render_all())
            action_space = get_action_space_interactive(grid_size,
                                                        time_step=int(
                                                            self.inited))
            if self.time > 2:
                action_space.append(
                    env_pb2.Action(text_data="I found the fault!"))
            action_space.append(env_pb2.Action(text_data="quit"))
            return action_space
        elif self.state == "fault":
            action_space = [
                env_pb2.Action(text_data=f"choose_frame_{i}")
                for i in range(len(self.frames))
            ]
            action_space.append(env_pb2.Action(text_data="The fault is here!"))
            action_space.append(env_pb2.Action(text_data="quit"))
            return action_space
        else:
            return []

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        logger.debug(f"Stepping with action: {action} for id: {self.id}")
        self.time += 1
        curr_state = self.triggering_state
        # Test if fault is triggered
        self.triggering_state |= ('true' in self.interpreter.evaluate_to_string(
        self.event))
        if curr_state != self.triggering_state:
            self.trigger_start_time = self.time

        if action.text_data == "quit":
            self.is_terminal = True
            self.last_interpreting_action = None
            observation = self.get_observation()
            reward = 0
        if action.text_data == "I found the fault!":
            self.state = "fault"
            self.curr_frame = 0
            self.last_interpreting_action = None
            # Give agent extra chance to choose the frame
            observation = self.get_observation()
            reward = 0
        if action.text_data == "The fault is here!":
            self.is_terminal = True
            if not self.triggering_state:
                self.triggering_state |= ('true' in self.interpreter.evaluate_to_string(
                                            self.event))
                self.trigger_start_time = self.time
            self.last_interpreting_action = None
            observation = self.fault_observation()
            self.curr_frame = self.time-1
            return observation, self.fault_reward(), self.is_terminal, {}
        if action.text_data.startswith("choose_frame_"):
            self.curr_frame = int(action.text_data.split("_")[-1])
            observation = self.get_observation()
            reward = 0
        else:
            last_interpreting_action = action.text_data
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}

            self.triggering_state |= ('true' in self.interpreter.evaluate_to_string(
                                        self.event))
            if curr_state != self.triggering_state:
                self.trigger_start_time = self.time + 1

            self.interpreter.step()
            observation = self.get_observation()
            logger.debug(f"Observation: {observation} for id: {self.id}")
            reward = 0
        self.frames.append(observation)
        return observation, reward, self.is_terminal, {}

    def fault_observation(self) -> env_pb2.Observation:
        orig_observation = self.get_observation()
        text_data = json.loads(orig_observation.text_data)
        if self.triggering_state and self.trigger_start_time:
            return env_pb2.Observation(text_data=json.dumps(
                {
                    "render": text_data,
                    "fault": True,
                    "offsets": self.trigger_frame - self.trigger_start_time
                }))
        else:
            return env_pb2.Observation(text_data=json.dumps({
                "render": text_data,
                "fault": False,
                "offsets": 0
            }))

    def fault_reward(self) -> float:
        if self.triggering_state and self.trigger_start_time:
            delta = self.curr_frame - self.trigger_start_time
            return 1 / (1 + delta)
        return -1

    def get_observation(self) -> env_pb2.Observation:
        if not self.inited:
            self.inited = True
            return env_pb2.Observation(
                text_data=
                "The interaction phase is now over. You will now interact with a defective version of the environment - where one of the dynamics rules has been changed. Your goal is to use you understanding of the environemnt from the interaction phase to detect the fault."
                +
                "Once you have detected the fault, you can either select 'The fault is here!' to select the current frame as the one where the defect appears and terminate the environment and get your score. You can click 'I found the fault!', you will be allowed to choose, from the frames that you have seen so far, the frame that contains the fault. After choosing the frame you can click 'The fault is here!' to terminate the environment and get your score."
                +
                "Then you will be asked to choose the frame in which the fault is located."
            )
        if self.state == "interactive":
            if self.render_mode == "text":
                render_dict = json.loads(self.interpreter.render_all())
                render_dict = render_grid(render_dict)
                return env_pb2.Observation(text_data=json.dumps(render_dict))
            elif self.render_mode == "image":
                render_dict = json.loads(self.interpreter.render_all())
                render_img_str = render_grid_matplotlib(
                    render_dict,
                    output_path=
                    f"{self.logging_path}/{self.env_name}/dd/dd_{self.time}.jpeg"
                )
                render_img_bytes = base64.b64decode(render_img_str)
                return env_pb2.Observation(
                    text_data="Here is the current frame: ",
                    image_data=render_img_bytes)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")
        elif self.state == "fault":
            if self.render_mode == "text":
                return env_pb2.Observation(text_data=json.dumps({
                    "render":
                    self.frames[self.curr_frame].text_data,
                    "fault":
                    True,
                    "offsets":
                    abs(self.trigger_frame -
                        self.curr_frame) if self.trigger_frame !=
                    -1 else 100_000
                }))
            elif self.render_mode == "image":
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "fault":
                        True,
                        "offsets":
                        abs(self.trigger_frame - self.curr_frame)
                        if self.trigger_frame != -1 else 100_000
                    }),
                    image_data=self.frames[self.curr_frame].image_data)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")
        else:
            if self.render_mode == "text":
                return env_pb2.Observation(text_data=json.dumps(
                    {
                        "render": self.frames[self.curr_frame].text_data,
                        "fault": False,
                        "offsets": 0
                    }))
            elif self.render_mode == "image":
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "fault": False,
                        "offsets": 0
                    }),
                    image_data=self.frames[self.curr_frame].image_data)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")

    def terminal(self):
        return self.is_terminal


class ActionPredictionEnvironment:

    def __init__(self,
                 env_name,
                 render_mode="text",
                 stack_frames=0,
                 skip_frames=False,
                 logging_path="./logs",
                 data_dir=CURR_DIR,
                 seed=0):
        self.env_name = env_name
        self.data_dir = data_dir
        self.prog = open(f"{data_dir}/programs/{env_name}.sexp", "r").read()
        with open(f"{data_dir}/prompts/{env_name}_planning.json", "r") as f:
            self.ap_dict = json.load(f)
        self.id = str(uuid.uuid4())
        self.render_mode = render_mode
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.logging_path = logging_path
        self.seed = seed
        self.reset()

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, "", self.seed)
        self.inited = False
        self.is_terminal = False

        self.color_dict = load_yaml_to_dict(f"{self.data_dir}/color_dict.yaml")
        self.inv_mask = self.ap_dict["mask"]  # Inv mask is for checking which region need to be the same color
        self.goal_state = self.ap_dict["goal"]
        self.goal_state = [[self.color_dict[cell] for cell in row]
                           for row in self.goal_state]
        if not os.path.exists(f"{self.logging_path}/{self.env_name}"):
            os.makedirs(f"{self.logging_path}/{self.env_name}")
        if not os.path.exists(f"{self.logging_path}/{self.env_name}/planning"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/planning")
        self.time = 0

    def get_action_space(self) -> List[env_pb2.Action]:
        _, grid_size = parse_grid(self.interpreter.render_all())
        action_space = get_action_space_interactive(grid_size)
        action_space.append(env_pb2.Action(text_data="quit"))
        action_space.append(env_pb2.Action(text_data="submit"))
        # action_space.append(env_pb2.Action(text_data="reset"))
        return action_space

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        logger.debug(f"Stepping with action: {action} for id: {self.id}")
        self.time += 1
        if action.text_data == "quit":
            self.is_terminal = True
            observation = self.get_quit_observation()
            return observation, 0, self.is_terminal, {}
        # if action.text_data == "reset":
        #     self.reset()
        #     observation = self.get_observation()
        #     return observation, 0, self.is_terminal, {}
        if action.text_data == "submit":
            self.is_terminal = True
            observation = self.action_prediction_observation()
            reward = 1 if observation.text_data == "You have successfully predicted the action." else -1
            return observation, reward, self.is_terminal, {}
        else:
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}
            self.interpreter.step()
            observation = self.get_observation()
            logger.debug(f"Observation: {observation} for id: {self.id}")
            return observation, 0, self.is_terminal, {}

    def get_quit_observation(self) -> env_pb2.Observation:
        return env_pb2.Observation(
            text_data="You quit the environment. No reward will be given.")

    def action_prediction_observation(self) -> env_pb2.Observation:
        render_dict = json.loads(self.interpreter.render_all())
        # render_dict = render_grid(render_dict)
        grid_matrix = render_grid_to_matrix(render_dict)
        if check_grid_same(grid_matrix, self.goal_state, self.inv_mask):
            return env_pb2.Observation(
                text_data="You have successfully predicted the action.")
        else:
            return env_pb2.Observation(
                text_data="You have failed to predict the action.")

    def get_observation(self) -> env_pb2.Observation:
        if not self.inited:
            self.inited = True
            return env_pb2.Observation(
                text_data=
                "The interaction phase is now over. Now you will try to solve a planning task in the environment you interacted with using your understanding of the dynamics of the environment."
                +
                "You will be given a goal state as well as a mask. Your aim is to interact in the environment such that the masked region (i.e. where mask is 1) of the grid matches the goal state."
                + "If you want to quit, click 'quit'." +
                "If you want to submit, click 'submit'." +
                "You will be penalized if you click 'submit' before the goal state is reached. Think carefully about the goal to be reached and your understanding of the environment before taking any actions as you might get stuck."
            )
        if self.render_mode == "text":
            render_dict = json.loads(self.interpreter.render_all())
            render_grid_matrix = render_grid(render_dict)
            return env_pb2.Observation(
                text_data=json.dumps({
                    "render": render_grid_matrix,
                    "goal": self.goal_state,
                    "mask": self.inv_mask
                }))
        elif self.render_mode == "image":
            render_dict = json.loads(self.interpreter.render_all())
            render_img_str = render_grid_matplotlib(
                render_dict,
                output_path=
                f"{self.logging_path}/{self.env_name}/planning/planning_{self.time}.jpeg"
            )

            color_name_lookup = {v: k for k, v in self.color_dict.items()}
            goal_color_grid = "\n".join(
                [" ".join(row) for row in self.goal_state])
            goal_img_str = render_string_grid_matplotlib(
                goal_color_grid,
                output_path=
                f"{self.logging_path}/{self.env_name}/planning/goal_state_{self.time}.jpeg"
            )

            # Create a JSON structure with both images
            image_data = {"grid": render_img_str, "goal_state": goal_img_str}
            image_json_str = json.dumps(image_data)

            return env_pb2.Observation(
                text_data=json.dumps({"mask": self.inv_mask}),
                image_data=image_json_str.encode('utf-8'))
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def terminal(self):
        return self.is_terminal


class MARAMCQEnvironment:

    def __init__(self,
                 env_name,
                 render_mode="text",
                 logging_path="./logs",
                 data_dir=CURR_DIR):
        self.env_name = env_name
        self.prog = open(f"{data_dir}/programs/{env_name}.sexp", "r").read()
        self.is_terminal = False
        self.is_finished = False
        self.render_mode = render_mode
        self.logging_path = logging_path
        self.data_dir = data_dir

    def reset(self) -> None:
        with open(f"{self.data_dir}/prompts/{self.env_name}_mfp.json", "r") as f:
            self.task_dict = json.load(f)
        
        with open(f"{self.data_dir}/answers/{self.env_name}_mfp.json", "r") as f:
            self.answer_answer_idx = json.load(f)["correct_idx"]
        self.is_terminal = False
        self.is_finished = False
        # self.data = json.load(open(f"{self.data_dir}/tasks/prompts/{self.env_name}_mcq.json"))
        self.current_time = 0
        self.current_question = 0
        self.options = None
        # Load and process colors
        self.colors: Dict[int, str] = load_yaml_to_dict(
            f"{self.data_dir}/color_dict.yaml")
        self.colors_str_to_int = {v: k for k, v in self.colors.items()}
        if not os.path.exists(f"{self.logging_path}/{self.env_name}"):
            os.makedirs(f"{self.logging_path}/{self.env_name}")
        if not os.path.exists(f"{self.logging_path}/{self.env_name}/mcq"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/mcq")
        self.parse_actions_and_update_task_dict()

    def parse_actions_and_update_task_dict(self) -> None:
        """Update the task_dict with masked_grid observations and gt answer 
        grid."""
        seed = int(self.task_dict.get("seed", 42))
        interpreter = Interpreter()
        interpreter.run_script(self.prog, autumnstdlib, "", seed)

        # Process observations
        actions_and_masks = self.task_dict["observations"]

        obss = [{"masked_grid": self.task_dict["observations"][0]["masked_grid"]}]

        # prev_obs = None
        for _, action_and_mask in enumerate(actions_and_masks[1:]):

            action = action_and_mask.get(
                "action", {"type": "noop"}
            ) 

            action_type = action.get("type", "")
            action_y = action.get("y", None)
            action_x = action.get("x", None)

            # Execute the action to get observation
            if action_type == "left":
                interpreter.left()
            elif action_type == "right":
                interpreter.right()
            elif action_type == "up":
                interpreter.up()
            elif action_type == "down":
                interpreter.down()
            elif action_type == "noop":
                pass
            elif action_type == "click" and action_y is not None and\
                action_x is not None:
                interpreter.click(action_x, action_y)
            else:
                # Skip unrecognized actions
                continue
            interpreter.step()

            processed_observation = {
                "action": action,
                "masked_grid": action_and_mask["masked_grid"],
            }
            # print(f"[green]Processed observation at time {t}: {processed_observation}[/green]")
            # print(f"mask {mask}")
            obss.append(processed_observation)

        self.task_dict["observations"] = obss
        self.task_dict["answer"] = {"correct_idx": self.answer_answer_idx}
        self.task_dict.pop("seed", None)

    def get_action_space(self) -> List[env_pb2.Action]:
        # If not finished
        actions = ["step"]
        if self.is_finished:
            options = self.get_options(self.current_question)
            actions.extend(["rewind"])
            actions.extend(
                ["choose_option_" + str(i) for i in range(len(options))])
        return [env_pb2.Action(text_data=act) for act in actions]

    def get_observation(self) -> env_pb2.Observation:
        # The observation consists of:
        # Current video frame, current video location, current grid (masked)
        # Choices if any
        convert_text_color = lambda x: '\n'.join([
            ' '.join([self.colors.get(cell, "black") for cell in row])
            for row in x
        ])

        with open(f"tmp.json", "w") as f:
            json.dump(self.task_dict, f)
        render = self.task_dict["observations"][self.current_time]["masked_grid"]
        color_grid = convert_text_color(render)

        if self.render_mode == "text":
            if self.is_finished:
                if self.task_dict["observations"][-1]["action"][
                        "type"] == "click":
                    action_took = self.task_dict["observations"][-1]["action"][
                        "type"] + " " + str(
                            self.task_dict["observations"][-1]["action"]["x"]
                        ) + " " + str(
                            self.task_dict["observations"][-1]["action"]["y"])
                else:
                    action_took = self.task_dict["observations"][-1]["action"][
                        "type"]
                options = self.get_options(self.current_question)
                options = [convert_text_color(option) for option in options]
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": self.current_time,
                        "render": color_grid,
                        "action_took": action_took,
                        "options": options,
                        "is_finished": self.is_finished,
                    }))
            else:
                if self.current_time == 0:
                    text_data =\
                    """The interaction phase is now over. You will now step through frames from a trajectory in this same environment you interacted with. Each frame is structured as a json object with the following fields:
\"render\": the grid observed,
\"video_location\": timestep at which the frame was observed,
\"action_took\": action taken at this timestep,
\"is_finished\": whether the episode is finished.
You will step through the trajectory one frame at a time. Towards the end of the trajectory, parts of the grid will be masked (where the masked locations are marked as `mask`) and you will be given a set of options to fill in the masked region at the final timestep. You need to choose option that fits the masked region at the final timestep.\n"""+\
                    json.dumps({
                        "video_location": self.current_time,
                        "render": color_grid,
                        # "action_took": action_took,
                        "is_finished": self.is_finished,})
                else:
                    if self.task_dict["observations"][
                            self.current_time]["action"]["type"] == "click":
                        click_x = self.task_dict["observations"][
                            self.current_time]["action"]["x"]
                        click_y = self.task_dict["observations"][
                            self.current_time]["action"]["y"]
                        action_took = f"click {click_x} {click_y}"
                    else:
                        action_took = self.task_dict["observations"][
                            self.current_time]["action"]["type"]
                    text_data = json.dumps({
                        "video_location": self.current_time,
                        "render": color_grid,
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    })
                return env_pb2.Observation(text_data=text_data)
        elif self.render_mode == "image":
            if self.is_finished:
                options = self.get_options(self.current_question)
                options = [convert_text_color(option) for option in options]
                options = [
                    render_string_grid_matplotlib(
                        option,
                        output_path=
                        f"{self.logging_path}/{self.env_name}/mcq/mcq_option_{i}.jpeg"
                    ) for i, option in enumerate(options)
                ]
                grid_image = render_string_grid_matplotlib(
                    color_grid,
                    output_path=
                    f"{self.logging_path}/{self.env_name}/mcq/mcq_render_{self.current_time}.jpeg"
                )

                # Create a JSON structure with all images
                image_data = {"options": options, "grid": grid_image}
                image_json_str = json.dumps(image_data)
                if self.task_dict["observations"][-1]["action"][
                        "type"] == "click":
                    action_took = self.task_dict["observations"][-1]["action"][
                        "type"] + " " + str(
                            self.task_dict["observations"][-1]["action"]["x"]
                        ) + " " + str(
                            self.task_dict["observations"][-1]["action"]["y"])
                else:
                    action_took = self.task_dict["observations"][-1]["action"][
                        "type"]
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": self.current_time,
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=image_json_str.encode('utf-8'))
            else:
                if self.task_dict["observations"][
                        self.current_time]["action"]["type"] == "click":
                    action_took = self.task_dict["observations"][
                        self.current_time]["action"]["type"] + " " + str(
                            self.task_dict["observations"][
                                self.current_time]["action"]["x"]) + " " + str(
                                    self.task_dict["observations"][
                                        self.current_time]["action"]["y"])
                else:
                    action_took = self.task_dict["observations"][
                        self.current_time]["action"]["type"]
                return env_pb2.Observation(
                    text_data=
                    """The interaction phase is now over. You will now step through frames from a trajectory in this same environment you interacted with. Each frame is structured as a json object with the following fields:
                                           \"video_location\": timestep at which the frame was observed,
                                           \"action_took\": action taken at this timestep,
                                           \"is_finished\": whether the episode is finished.
                                           You will step through the trajectory one frame at a time. Towards the end of the trajectory, you will be given masked states (where the masked locations are colored slategrey) and at the end of the trajectory, you will be given a set of options to fill in the masked region. You need to choose the correct option.\n"""
                    + json.dumps({
                        "video_location": self.current_time,
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=base64.b64decode(
                        render_string_grid_matplotlib(
                            color_grid,
                            output_path=
                            f"{self.logging_path}/{self.env_name}/mcq/mcq_render_{self.current_time}.jpeg"
                        ))
                ) if self.current_time == 0 else env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": self.current_time,
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=base64.b64decode(
                        render_string_grid_matplotlib(
                            color_grid,
                            output_path=
                            f"{self.logging_path}/{self.env_name}/mcq/mcq_render_{self.current_time}.jpeg"
                        )))

    def terminal(self):
        return self.is_terminal

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        if self.terminal():
            return self.get_observation(), 0, self.terminal(), {}
        if action.text_data == "rewind":
            self.current_time -= 1
            if self.current_time < 0:
                self.current_time = 0
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}
        elif action.text_data == "step":
            self.current_time = min(self.current_time + 1,
                                    len(self.task_dict["observations"]) - 1)
            if self.current_time >= len(self.task_dict["observations"]) - 1:
                self.is_finished = True
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}
        elif action.text_data.startswith("choose_option_"):
            self.is_terminal = True
            self.current_option = int(action.text_data.split("_")[-1])
            if self.current_option == self.correct_idx:
                observation = self.get_observation()
                return observation, 1, self.terminal(), {}
            else:
                observation = self.get_observation()
                return observation, 0, self.terminal(), {}
        elif action.text_data == "quit":
            self.is_terminal = True
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}
        else:
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}

    def get_options(self, question_idx: int) -> List[str]:
        if self.options is not None:
            return self.options
        all_options = self.task_dict["choices"]

        correct_idx = self.task_dict["answer"]["correct_idx"]

        self.options = all_options
        self.correct_idx = correct_idx
        return all_options


def create_rectangular_mask(rect: Dict[str, int],
                            grid_size: int) -> List[List[int]]:
    """Create a mask grid from rectangular coordinates."""
    x, y, width, height = rect["x"], rect["y"], rect["width"], rect["height"]
    mask = [[1 for _ in range(grid_size)] for _ in range(grid_size)]

    for i in range(y, y + height):
        for j in range(x, x + width):
            if 0 <= i < grid_size and 0 <= j < grid_size:
                mask[i][j] = 0

    return mask


def convert_to_grid(
        obs_json: Dict[str, Any],
        background_color: int,
        color_dict: Dict[str, int],
        grid_size: int,
        use_color_str: bool = False) -> List[List[Union[int, str]]]:
    """Convert the observation JSON to a grid."""
    grid_size = obs_json.get("GRID_SIZE", 0)

    if use_color_str:
        color_int_to_str = {v: k for k, v in color_dict.items()}
        grid = [[color_int_to_str[background_color] for _ in range(grid_size)]
                for _ in range(grid_size)]  # background is 1 by default
    else:
        grid = [[background_color for _ in range(grid_size)]
                for _ in range(grid_size)]  # background is 1 by default

    # Fill in objects from the observation data
    for obj_type, objects in obs_json.items():
        if obj_type == "GRID_SIZE":
            continue

        for obj in objects:
            x = obj["position"]["x"]
            y = obj["position"]["y"]
            color = obj["color"]

            if 0 <= x < grid_size and 0 <= y < grid_size:
                if color in color_dict:
                    if use_color_str:
                        grid[y][x] = color
                    else:
                        grid[y][x] = color_dict[color]
                else:
                    raise ValueError(
                        f"Color {color} not found in color_dict. Please add it to the color_dict.yaml file."
                    )

    return grid


def apply_mask(obs_grid: List[List[int]],
               mask: List[List[int]]) -> List[List[int]]:
    """Apply the mask to the observation grid."""
    if not mask:
        return obs_grid

    new_obs_grid = []
    for row in range(len(obs_grid)):
        new_row = []
        for col in range(len(obs_grid[row])):
            if mask[row][col] == 0:
                new_row.append(0)
            else:
                new_row.append(obs_grid[row][col])
        new_obs_grid.append(new_row)
    return new_obs_grid


if __name__ == "__main__":
    env = ActionPredictionEnvironment("space_invaders")
    print(env.get_action_space())
    print(env.get_observation())
    print(env.step(env.get_action_space()[0]))
