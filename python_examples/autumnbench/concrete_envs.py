import os
import hashlib
import uuid
from typing import List, Tuple, Dict, Any, Union
import random
import math

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
                 stack_frames=False,
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
        self.num_stack_frames = 1
        self.skip_frames = skip_frames
        self.render_mode = render_mode
        self.logging_path = logging_path
        self.seed = seed
        self.env_name = env_name
        self.color_dict = load_yaml_to_dict(f"{data_dir}/color_dict.yaml")
        self.color_dict_str_to_int = {v: k for k, v in self.color_dict.items()}

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, "", self.seed)
        self.inited = False
        self.num_stack_frames = self.interpreter.get_frame_rate() // 2 if self.stack_frames else 1
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
            return observation, 0, self.is_terminal, {"terminal_condition": "quit"}
        elif action.text_data == "go-to-test":
            self.is_terminal = True
            observation = self.get_observation()
            return observation, 0, self.is_terminal, {}
        elif action.text_data == "reset":
            self.interpreter = Interpreter()
            self.interpreter.run_script(self.prog, autumnstdlib, "", self.seed)
            observation = self.get_observation()
            return observation, 0, self.is_terminal, {}
        else:
            if not interpreter_action_to_text(self.interpreter, action.text_data):
                logger.warning(f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}
            self.interpreter.step()
            observation = self.get_observation()
            if self.num_stack_frames > 1:
                if self.render_mode == "text":
                    stacked_frames = [{"frame_time": self.time, "action_took": "noop", "render": observation.text_data}]
                    for i in range(self.num_stack_frames):
                        self.interpreter.step()
                        observation = self.get_observation()
                        stacked_frames.append({"frame_time": self.time * self.num_stack_frames + i, "action_took": "noop", "render": observation.text_data})
                    if not self.skip_frames:
                        observation = "\n\nObservation: ".join([f"Frame time: {frame['frame_time']}\nAction took: noop\nObservation: {frame['render']}" for frame in stacked_frames])
                        observation = env_pb2.Observation(text_data=observation)
                    else:
                        observation = env_pb2.Observation(
                            text_data=f"Skipped {self.num_stack_frames} frames. Current frame: {observation.text_data}"
                        )
                elif self.render_mode == "image":
                    stacked_frames = [observation.image_data]
                    for _ in range(self.num_stack_frames):
                        self.interpreter.step()
                        observation = self.get_observation()
                        stacked_frames.append(observation.image_data)
                    if not self.skip_frames:
                        # For image mode, we can't join image data as strings, so we'll use the last frame
                        observation = env_pb2.Observation(
                            image_data=stacked_frames[-1]
                        )
                    else:
                        observation = env_pb2.Observation(
                            text_data=f"Skipped {self.stack_frames} frames. Here is the current frame: ",
                            image_data=observation.image_data,
                        )
            logger.debug(f"Observation: {observation} for id: {self.id}")
            return observation, 0, self.is_terminal, {}

    def get_observation(self) -> env_pb2.Observation:
        text_data = ""
        if not self.inited:
            self.inited = True
            text_data = self.get_instruction_text() + "\nHere is the initial state of the grid: \n"
        render_dict = json.loads(self.interpreter.render_all())
        render_img_str = render_grid_matplotlib(
            render_dict,
            output_path=f"{self.logging_path}/{self.env_name}/interactive/interactive_{self.time}.jpeg",
            background_color=self.interpreter.get_background(),
            color_dict=self.color_dict_str_to_int
        )
        render_img_bytes = base64.b64decode(render_img_str)
        render_dict = render_grid(render_dict, background_color=self.interpreter.get_background(), color_dict=self.color_dict_str_to_int)
        if self.render_mode == "text":
            return env_pb2.Observation(text_data=text_data + json.dumps(render_dict))
        elif self.render_mode == "image":
            return env_pb2.Observation(
                text_data=text_data, image_data=render_img_bytes
            )
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def get_instruction_text(self):
        # TODO: add task description

        return f"""Welcome, you are now in the interactive phase, where you can interact with the grid using the available actions.
During the interactive phase your goal is to act in the environment to understand the underlying rules of the environment. 
Understand the environment and the dynamics of the environment well. Once you have understood the environment, you can select 'go-to-test' to go to the test phase.
After the interactive phase you will be asked to use this knowledge about the environment to answer some questions about it.

"""
# Task Description: {task_description}

    def terminal(self):
        return self.is_terminal


class ChangeDetectionEnvironment:

    def __init__(self, env_name):
        self.prog = open(f"{CURR_DIR}/modified_programs/{env_name}_wrong.sexp",
                         "r").read()
        self.event = open(
            f"{CURR_DIR}/modified_programs/{env_name}_condition.sexp",
            "r").read()
        self.is_terminal = False
        self.triggering_state = False
        self.trigger_start_time = None # the trigger frame is not set yet
        self.frames = []
        self.id = str(uuid.uuid4())
        self.inited = False

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, self.event)
        self.triggering_state = False
        self.trigger_start_time = None # the trigger frame is not set yet
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
            observation = self.change_observation()
            return observation, self.change_reward(), self.is_terminal, {}
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

    def change_observation(self) -> env_pb2.Observation:
        orig_observation = self.get_observation()
        text_data = json.loads(orig_observation.text_data)
        if self.triggering_state and self.trigger_start_time:
            return env_pb2.Observation(text_data=json.dumps(
                {
                    "render": text_data,
                    "change": True,
                    "trigger_time": time.time() - self.trigger_start_time,
                    "detect_time": time.time() - self.trigger_start_time
                }))
        else:
            return env_pb2.Observation(text_data=json.dumps({
                "render": text_data,
                "change": False,
                "trigger_time": 0,
                "detect_time": 0
            }))

    def change_reward(self) -> float:
        if self.triggering_state and self.trigger_start_time:
            delta = time.time() - self.trigger_start_time
            return 1 / (1 + delta)
        return -1

    def get_observation(self) -> env_pb2.Observation:
        if not self.inited:
            self.inited = True
            return env_pb2.Observation(
                text_data=
                "Welcome, you will now be playing an Autumn change detection environment."
                +
                "Remember what you see and how different the environment is from the normal Autumn environment."
                +
                "Once you have detected the change, you have to click 'Fault!' to terminate the environment."
                +
                "You will be penalized if you click 'Fault!' before the change is detected."
            )
        render_dict = json.loads(self.interpreter.render_all())
        render_dict = render_grid(render_dict, background_color=self.interpreter.get_background(), color_dict=self.color_dict_str_to_int)
        return env_pb2.Observation(text_data=json.dumps(render_dict))

    def terminal(self):
        return self.is_terminal


class CDSliderEnvironment:

    def __init__(self,
                 env_name,
                 render_mode="text",
                 stack_frames=False,
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
        self.color_dict = load_yaml_to_dict(f"{self.data_dir}/color_dict.yaml")
        self.color_dict_str_to_int = {v: k for k, v in self.color_dict.items()}
        self.last_interpreting_action = None
        self.reset()

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, self.event,
                                    self.seed)
        self.triggering_state = False
        self.trigger_start_time = None # the trigger frame is not set yet
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
        if not os.path.exists(f"{self.logging_path}/{self.env_name}/cd"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/cd")

    def get_action_space(self) -> List[env_pb2.Action]:
        if self.state == "interactive":
            _, grid_size = parse_grid(self.interpreter.render_all())
            action_space = get_action_space_interactive(grid_size,
                                                        time_step=int(
                                                            self.inited))
            # remove go-to-test action
            action_space = [action for action in action_space if action.text_data != "go-to-test"]
            if self.time > 2:
                action_space.append(
                    env_pb2.Action(text_data="I found the change!"))
            action_space.append(env_pb2.Action(text_data="quit"))
            return action_space
        elif self.state == "change":
            action_space = [
                env_pb2.Action(text_data=f"choose_frame_{i}")
                for i in range(len(self.frames))
            ]
            action_space.append(env_pb2.Action(text_data="Submit choice"))
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
        # Test if change is triggered
        self.triggering_state |= ('true' in self.interpreter.evaluate_to_string(self.event))
        if curr_state != self.triggering_state:
            self.trigger_start_time = self.time
        self.curr_frame = self.time-1
        if (action not in self.get_action_space()) and (not action.text_data.startswith("click")): # TODO: ensure we don't allow invalid click actions
            observation = env_pb2.Observation(text_data="Invalid action. Please select a valid action.")
            return observation, 0, self.is_terminal, {}

        reward = 0
        if action.text_data == "quit":
            self.is_terminal = True
            self.last_interpreting_action = None
            observation = self.get_observation()
            return observation, 0, self.is_terminal, {"terminal_condition": "quit"}
        elif action.text_data == "I found the change!":
            self.state = "change"
            self.curr_frame = 0
            self.last_interpreting_action = None
            # Give agent extra chance to choose the frame
            observation = self.get_observation()
        elif action.text_data == "Submit choice":
            self.is_terminal = True
            self.last_interpreting_action = None
            observation = self.change_observation()
            reward = self.change_reward()
            return observation, reward, self.is_terminal, {"terminal_condition": "finish"}
        elif action.text_data.startswith("choose_frame_"):
            self.curr_frame = int(action.text_data.split("_")[-1])
            observation = self.get_observation()
        elif action.text_data == "reset":
            self.interpreter = Interpreter()
            self.interpreter.run_script(self.prog, autumnstdlib, self.event,
                                        self.seed)
            self.triggering_state = False
            self.trigger_start_time = None # the trigger frame is not set yet
            self.frames = []
            self.curr_frame = 0
        else:
            self.last_interpreting_action = action.text_data
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return self.get_observation(), 0, self.is_terminal, {}

            self.triggering_state |= ('true' in self.interpreter.evaluate_to_string(self.event))            
            if curr_state != self.triggering_state:
                self.trigger_start_time = self.time

            self.interpreter.step()
            observation = self.get_observation()
            logger.debug(f"Observation: {observation} for id: {self.id}")
        self.frames.append(observation)
        return observation, reward, self.is_terminal, {}

    def change_observation(self) -> env_pb2.Observation:
        # orig_observation = self.get_observation()
        # text_data = orig_observation.text_data
        # image_data = orig_observation.image_data
        if self.trigger_start_time is None:
            text_data = f"The environment has not changed yet."
        elif self.triggering_state:
            text_data = f"The environment has changed! The change offset from the start of the changed behavior is {self.curr_frame - self.trigger_start_time} frames."
        else:
            text_data = f"You have not detected the change. The change offset from the start of the changed behavior is {self.curr_frame - self.trigger_start_time} frames."
        if self.render_mode == "text":
            return env_pb2.Observation(text_data=text_data)
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def change_reward(self) -> float:
        """Calculate score for change detection task"""
        if self.trigger_start_time is None:
            return -1
        if self.curr_frame < (self.trigger_start_time-1):
            return -1
        elif self.curr_frame == (self.trigger_start_time-1) or self.curr_frame == self.trigger_start_time:
            return 1
        else:
            fx = 1 / (1 - self.curr_frame / self.trigger_start_time * math.exp(-self.curr_frame / self.trigger_start_time))
            return max(0.0, min(1.377 * fx - 1.178, 1.0))

    def get_observation(self) -> env_pb2.Observation:
        text_data = ""
        if not self.inited:
            self.inited = True
            text_data = (
                "You are now in the test phase." +
                "You will now interact with a changed version of the environment - where one of the dynamics rules has been changed. " +
                "Your goal is to use you understanding of the environemnt from the interaction phase to detect the change. The environment will start in a normal state and then at some point, the environment will transition to a defective state. " +
                "As soon as you detect the change, you have to select 'I found the change!' action to go to the next phase, wherein you have to choose exactly which frame the change occurred, then submit it. You may choose as many times as you want to see the frames. You will be penalized if you click 'I found the change!' before the change is detected. " +
                "Here is the initial frame: "
            )
        else:
            text_data = "Here is the current frame: "
        if self.state == "interactive":
            if self.render_mode == "text":
                render_dict = json.loads(self.interpreter.render_all())
                render_dict = render_grid(render_dict, background_color=self.interpreter.get_background(), color_dict=self.color_dict_str_to_int)
                return env_pb2.Observation(text_data=text_data + json.dumps(render_dict))
            elif self.render_mode == "image":
                render_dict = json.loads(self.interpreter.render_all())
                render_img_str = render_grid_matplotlib(
                    render_dict,
                    output_path=
                    f"{self.logging_path}/{self.env_name}/cd/cd_{self.time}.jpeg",
                    color_dict=self.color_dict_str_to_int
                )
                render_img_bytes = base64.b64decode(render_img_str)
                return env_pb2.Observation(
                    text_data=text_data,
                    image_data=render_img_bytes)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")
        elif self.state == "change":
            if self.render_mode == "text":
                return env_pb2.Observation(text_data=self.frames[self.curr_frame].text_data)
            elif self.render_mode == "image":
                return env_pb2.Observation(text_data=text_data, image_data=self.frames[self.curr_frame].image_data)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")
        else:
            if self.render_mode == "text":
                return env_pb2.Observation(text_data=self.frames[self.curr_frame].text_data)
            elif self.render_mode == "image":
                return env_pb2.Observation(
                    text_data=text_data,
                    image_data=self.frames[self.curr_frame].image_data)
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")

    def terminal(self):
        return self.is_terminal


class PlanningEnvironment:

    def __init__(self,
                 env_name,
                 render_mode="text",
                 stack_frames=False,
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
        self.color_dict_str_to_int = {v: k for k, v in self.color_dict.items()}
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
        action_space = [action for action in action_space if action.text_data != "reset" and action.text_data != "go-to-test"]
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
        # action invalid message if reset is called
        if action.text_data == "reset":
            return env_pb2.Observation(text_data="Reset is not allowed in the test phase."), 0, self.is_terminal, {}
        else:
            if not interpreter_action_to_text(self.interpreter,
                                              action.text_data):
                logger.warning(
                    f"Invalid action: {action.text_data} for id: {self.id}")
                return env_pb2.Observation(text_data="Invalid action. Please select a valid action."), 0, self.is_terminal, {}
            self.interpreter.step()
            observation = self.get_observation()
            if self.reached_goal():
                self.is_terminal = True
                return observation, 1, self.is_terminal, {"terminal_condition": "finish"}
            logger.debug(f"Observation: {observation} for id: {self.id}")
            return observation, 0, self.is_terminal, {}

    def get_quit_observation(self) -> env_pb2.Observation:
        return env_pb2.Observation(
            text_data="You quit the environment. No reward will be given.")

    def reached_goal(self) -> bool:
        render_dict = json.loads(self.interpreter.render_all())
        grid_matrix = render_grid_to_matrix(render_dict, background_color=self.interpreter.get_background(), color_dict=self.color_dict_str_to_int)
        return check_grid_same(grid_matrix, self.goal_state, self.inv_mask)

    def get_observation(self) -> env_pb2.Observation:
        text_data = ""
        if not self.inited:
            self.inited = True
            text_data = f"""The interaction phase is over, you have entered the test phase. You will now be given a goal state and a highlight mask of the same size as the grid where 1 indicates the region to be reached and 0 indicates the region to be ignored. 
            Your aim is to solve a planning task in the environment you interacted by reaching the goal state in the highlighted region.
            Note that you can no longer reset the environment, so plan carefully. You will be given the same environment as you interacted with in the interaction phase, you need to interact with it to reach the goal state in the highlighted region.
            Your grid will be checked against the goal state and the highlight mask at every timestep. If you reach the goal state in the highlighted region, you will be given a reward. You may choose to quit at any time if you are stuck.
            The initial grid is:
            """
        else:
            text_data = "The current grid is: "
        if self.render_mode == "text":
            render_dict = json.loads(self.interpreter.render_all())
            render_grid_matrix = render_grid(render_dict, background_color=self.interpreter.get_background(), color_dict=self.color_dict_str_to_int)
            return env_pb2.Observation(
                text_data=text_data + json.dumps({
                    "render": render_grid_matrix,
                    "goal": self.goal_state,
                    "highlight_mask": self.inv_mask
                }))
        elif self.render_mode == "image":
            render_dict = json.loads(self.interpreter.render_all())
            render_img_str = render_grid_matplotlib(
                render_dict,
                output_path=
                f"{self.logging_path}/{self.env_name}/planning/planning_{self.time}.jpeg",
                color_dict=self.color_dict_str_to_int
            )

            goal_color_grid = "\n".join(
                [" ".join(row) for row in self.goal_state])
            goal_img_str = render_string_grid_matplotlib(
                goal_color_grid,
                output_path=
                f"{self.logging_path}/{self.env_name}/planning/goal_state_{self.time}.jpeg",
                color_dict=self.color_dict_str_to_int
            )

            # Create a JSON structure with both images
            image_data = {"grid": render_img_str, "goal_state": goal_img_str}
            image_json_str = json.dumps(image_data)

            return env_pb2.Observation(
                text_data=text_data + json.dumps({"highlight_mask": self.inv_mask}),
                image_data=image_json_str.encode('utf-8'))
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def terminal(self):
        return self.is_terminal


class MARAMFPEnvironment:

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
        self.color_dict = load_yaml_to_dict(f"{self.data_dir}/color_dict.yaml")
        self.color_dict_str_to_int = {v: k for k, v in self.color_dict.items()}

    def reset(self) -> None:
        with open(f"{self.data_dir}/prompts/{self.env_name}_mfp.json", "r") as f:
            self.prompt = json.load(f)
        
        with open(f"{self.data_dir}/answers/{self.env_name}_mfp.json", "r") as f:
            self.answer = json.load(f)
        self.is_terminal = False
        self.is_finished = False
        self.current_time = 0
        # Load and process colors
        self.colors: Dict[int, str] = load_yaml_to_dict(
            f"{self.data_dir}/color_dict.yaml")
        self.colors_str_to_int = {v: k for k, v in self.colors.items()}
        if not os.path.exists(f"{self.logging_path}/{self.env_name}"):
            os.makedirs(f"{self.logging_path}/{self.env_name}")
        if not os.path.exists(f"{self.logging_path}/{self.env_name}/mfp"):
            os.makedirs(f"{self.logging_path}/{self.env_name}/mfp")

    def get_action_space(self) -> List[env_pb2.Action]:
        # If not finished
        actions = ["step"]
        if self.is_finished:
            choices = self.get_choices()
            actions.extend(["rewind"])
            actions.extend(
                ["choose_option_" + str(i) for i in range(len(choices))])
        return [env_pb2.Action(text_data=act) for act in actions]

    def get_observation(self) -> env_pb2.Observation:
        # The observation consists of:
        # Current video frame, current video location, current grid (masked)
        # Choices if any
        convert_to_text_color = lambda x: [[
            "mask" if cell == 0 else self.colors.get(cell, "black") for cell in row
        ] for row in x]

        render = self.prompt["observations"][self.current_time]["masked_grid"]
        color_grid = convert_to_text_color(render)
        # Convert list of lists to string format for render_string_grid_matplotlib
        color_grid_str = "\n".join([" ".join(row) for row in color_grid])

        if self.render_mode == "text":
            if self.is_finished:
                if self.prompt["observations"][-1]["action"][
                        "type"] == "click":
                    action_took = self.prompt["observations"][-1]["action"][
                        "type"] + " " + str(
                            self.prompt["observations"][-1]["action"]["x"]
                        ) + " " + str(
                            self.prompt["observations"][-1]["action"]["y"])
                else:
                    action_took = self.prompt["observations"][-1]["action"][
                        "type"]
                choices = self.get_choices()
                choices = [convert_to_text_color(option) for option in choices]
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "render": color_grid,
                        "action_took": action_took,
                        # "choices": {0: choices[0], 1: choices[1], 2: choices[2], 3: choices[3], 4: choices[4], 5: choices[5]},
                        "choices": choices,
                        "is_finished": self.is_finished,
                    }))
            else:
                if self.current_time == 0:
                    text_data =\
                    """The interaction phase is now over. You will now step through frames from a trajectory in this same environment you interacted with (use the `step` action to step through the trajectory). Each frame is structured as a json object with the following fields:
\"render\": the grid observed,
\"video_location\": timestep at which the frame was observed,
\"action_took\": action taken at this timestep,
\"is_finished\": whether the episode is finished.
You will step through the trajectory one frame at a time. Towards the end of the trajectory, parts of the grid will be masked (where the masked locations are marked as `mask`) and you will be given a set of choices to fill in the masked region at the final timestep. You need to choose option that fits the masked region at the final timestep. You can also use the `rewind` action to go back to the previous frame.\n"""+\
                    json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "render": color_grid,
                        "action_took": "start",
                        "is_finished": self.is_finished,})
                else:
                    if self.prompt["observations"][
                            self.current_time]["action"]["type"] == "click":
                        click_x = self.prompt["observations"][
                            self.current_time]["action"]["x"]
                        click_y = self.prompt["observations"][
                            self.current_time]["action"]["y"]
                        action_took = f"click {click_x} {click_y}"
                    else:
                        action_took = self.prompt["observations"][
                            self.current_time]["action"]["type"]
                    text_data = json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "render": color_grid,
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    })
                return env_pb2.Observation(text_data=text_data)
        elif self.render_mode == "image":
            if self.is_finished:
                choices = self.get_choices()
                choices = [convert_to_text_color(option) for option in choices]
                choices = [
                    render_string_grid_matplotlib(
                        option,
                        output_path=
                        f"{self.logging_path}/{self.env_name}/mfp/mfp_option_{i}.jpeg",
                        color_dict=self.color_dict_str_to_int
                    ) for i, option in enumerate(choices)
                ]
                grid_image = render_string_grid_matplotlib(
                    color_grid_str,
                    output_path=
                    f"{self.logging_path}/{self.env_name}/mfp/mfp_render_{self.current_time}.jpeg",
                    color_dict=self.color_dict_str_to_int
                )

                # Create a JSON structure with all images
                image_data = {"choices": choices, "grid": grid_image}
                image_json_str = json.dumps(image_data)
                if self.prompt["observations"][-1]["action"][
                        "type"] == "click":
                    action_took = self.prompt["observations"][-1]["action"][
                        "type"] + " " + str(
                            self.prompt["observations"][-1]["action"]["x"]
                        ) + " " + str(
                            self.prompt["observations"][-1]["action"]["y"])
                else:
                    action_took = self.prompt["observations"][-1]["action"][
                        "type"]
                return env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=image_json_str.encode('utf-8'))
            else:
                if "action" in self.prompt["observations"][self.current_time]:
                    if self.prompt["observations"][
                            self.current_time]["action"]["type"] == "click":
                        action_took = self.prompt["observations"][
                            self.current_time]["action"]["type"] + " " + str(
                                self.prompt["observations"][
                                    self.current_time]["action"]["x"]) + " " + str(
                                        self.prompt["observations"][
                                            self.current_time]["action"]["y"])
                    else:
                        action_took = self.prompt["observations"][
                            self.current_time]["action"]["type"]
                else:
                    action_took = "start"
                return env_pb2.Observation(
                    text_data=
                    """The interaction phase is now over. You will now step through frames from a trajectory in this same environment you interacted with. Each frame is structured as a json object with the following fields:
                                           \"video_location\": timestep at which the frame was observed,
                                           \"action_took\": action taken at this timestep,
                                           \"is_finished\": whether the episode is finished.
                                           You will step through the trajectory one frame at a time. Towards the end of the trajectory, you will be given masked states (where the masked locations are colored slategrey) and at the end of the trajectory, you will be given a set of choices to fill in the masked region. You need to choose the correct option.\n"""
                    + json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=base64.b64decode(
                        render_string_grid_matplotlib(
                            color_grid,
                            output_path=
                            f"{self.logging_path}/{self.env_name}/mfp/mfp_render_{self.current_time}.jpeg",
                            color_dict=self.color_dict_str_to_int
                        ))
                ) if self.current_time == 0 else env_pb2.Observation(
                    text_data=json.dumps({
                        "video_location": str(self.current_time)+"/"+str(len(self.prompt["observations"])-1),
                        "action_took": action_took,
                        "is_finished": self.is_finished,
                    }),
                    image_data=base64.b64decode(
                        render_string_grid_matplotlib(
                            color_grid,
                            output_path=
                            f"{self.logging_path}/{self.env_name}/mfp/mfp_render_{self.current_time}.jpeg",
                            color_dict=self.color_dict_str_to_int
                        )))

    def terminal(self):
        return self.is_terminal

    def step(
        self, action: env_pb2.Action
    ) -> Tuple[env_pb2.Observation, float, bool, Dict[str, str]]:
        assert isinstance(action, env_pb2.Action)
        if self.terminal():
            return self.get_observation(), 0, self.terminal(), {}
        if action not in self.get_action_space():
            observation = env_pb2.Observation(text_data="Invalid action. Please choose from the available actions.")
            return observation, 0, self.terminal(), {}
        if action.text_data == "rewind":
            self.current_time -= 1
            if self.current_time < 0:
                self.current_time = 0
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}
        elif action.text_data == "step":
            self.current_time = min(self.current_time + 1,
                                    len(self.prompt["observations"]) - 1)
            if self.current_time >= len(self.prompt["observations"]) - 1:
                self.is_finished = True
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}
        elif action.text_data.startswith("choose_option_"):
            self.is_terminal = True
            self.current_option = int(action.text_data.split("_")[-1])
            if self.current_option == self.answer["correct_idx"]:
                observation = self.get_observation()
                return observation, 1, self.terminal(), {}
            else:
                observation = self.get_observation()
                return observation, -1, self.terminal(), {}
        elif action.text_data == "quit":
            self.is_terminal = True
            observation = self.get_observation()
            return observation, 0, self.terminal(), {"terminal_condition": "quit"}
        else:
            observation = self.get_observation()
            return observation, 0, self.terminal(), {}

    def get_choices(self) -> List[str]: 
        return self.prompt["choices"]


if __name__ == "__main__":
    env = PlanningEnvironment("space_invaders")
    print(env.get_action_space())
    print(env.get_observation())
    print(env.step(env.get_action_space()[0]))
