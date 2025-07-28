from .environment_interfaces import MARAInteractiveServicer, env_grpc, env_pb2, env_service_pb2
from .concrete_envs import MARAMFPEnvironment
import os
import json
import random
from typing import Dict, Any, Tuple, List
# Log to a file
import logging

logging.basicConfig(level=logging.INFO,
                    filename="log_environment_interfaces_mfp.txt")
# Create file if it doesn't exist
if not os.path.exists("log_environment_interfaces_mfp.txt"):
    with open("log_environment_interfaces_mfp.txt", "w") as f:
        pass
logger = logging.getLogger(__name__)
fh = logging.FileHandler("log_environment_interfaces_mfp.txt")
logger.addHandler(fh)

import numpy as np

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class MARAMFPServicer(MARAInteractiveServicer):

    def __init__(self):
        pass

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = MARAMFPEnvironment(
            request.config["env_name"],
            render_mode=request.config["render_mode"],
            logging_path=request.config["logging_path"],
            data_dir=request.config["data_dir"])
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"MFP Environment initialized: {request.config['env_name']}"
        )
        reactive_env = env_pb2.ReactiveEnvironment(
            environment_id=f"{request.config['env_name']}_v1_mfp",
            version="1.0.0",
            metadata={
                "author": "MARA Developer",
                "domain": "Autumn"
            })
        response.reactive_env.CopyFrom(reactive_env)
        return response

    def GetEnvironmentInfo(self, request, context):
        return env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{self.environment.env_name}_v1_mfp",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={
                "text_input": "true",
                "text_output": "true"
            },
            metadata={
                "genre": "adventure",
                "difficulty": "beginner"
            })


class MARACompositeAutumnMFPServicer(MARAInteractiveServicer):

    def __init__(self):
        self.interactive_environment = MARAInteractiveServicer()
        self.mfp_environment = MARAMFPServicer()
        self.current_environment = self.interactive_environment
        self.transiting_state = "Interactive"  # InteractiveReset-> Interactive -> Transit -> MFPReset -> MFP -> End.
        self.env_name = None

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.env_name = request.config["env_name"]
        self.max_interaction_steps = int(request.config["max_interaction_steps"])
        self.render_mode = request.config["render_mode"]
        self.data_dir = request.config["data_dir"]
        self.logging_path = request.config["logging_path"]
        self.current_environment = self.interactive_environment
        return self.current_environment.Initialize(request, context)

    def Reset(self, request, context):
        self.steps = 0
        return self.current_environment.Reset(request, context)

    def GetEnvironmentInfo(self, request, context):
        return self.current_environment.GetEnvironmentInfo(request, context)

    def Step(self, request, context):
        self.steps += 1
        if self.transiting_state == "Interactive":
            step_response = self.current_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if (is_terminal and self.current_environment == self.interactive_environment) or \
               (self.steps >= self.max_interaction_steps):
                self.current_environment = self.mfp_environment
                self.transiting_state = "Transition"
                step_response.is_terminal = False
                self.steps = 0

                init_req = env_service_pb2.InitializeRequest(
                    config={
                        "env_name": self.env_name,
                        "render_mode": self.render_mode,
                        "data_dir": self.data_dir,
                        "logging_path": self.logging_path
                    })
                self.mfp_environment.Initialize(init_req, context)
                reset_req = env_service_pb2.ResetRequest()
                self.mfp_environment.Reset(reset_req, context)
                observation = self.mfp_environment.environment.get_observation()
                self.transiting_state = "MFP"
                return env_service_pb2.StepResponse(observation=observation,
                                                    reward=0,
                                                    is_terminal=False,
                                                    info={})
            return step_response
        elif self.transiting_state == "MFP":
            step_response = self.mfp_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting_state = "End"
                step_response.is_terminal = True
            return step_response
        elif self.transiting_state == "End":
            return env_service_pb2.StepResponse(
                observation=env_pb2.Observation(
                    text_data="MFP environment ended."),
                reward=0,
                is_terminal=True,
                info={})
        else:
            raise ValueError(
                f"Invalid transiting state: {self.transiting_state}")

    def QuerySpaces(self, request, context):
        if self.transiting_state == "Interactive":
            return self.current_environment.QuerySpaces(request, context)
        elif self.transiting_state == "MFP":
            return self.mfp_environment.QuerySpaces(request, context)
        elif self.transiting_state == "MFPReset":
            return self.mfp_environment.QuerySpaces(request, context)
        else:
            return env_service_pb2.SpaceQueryResponse(
                reactive_response=env_pb2.ReactiveEnvironment.
                ActionSpaceResponse(
                    action_space=env_pb2.ReactiveEnvironment.ActionSpace(
                        available_actions=[env_pb2.Action(text_data="noop")])))

    def Close(self, request, context):
        return self.current_environment.Close(request, context)


class MARAROBOSIMMFPEnvironment:

    def __init__(self, env_name):
        self.env_name = env_name
        self.is_terminal = False
        self.is_finished = False
        self.current_video = "reference"  # Start with reference video
        self.score = 0
        self.total_questions = 0

    def reset(self):
        self.is_terminal = False
        self.is_finished = False
        self.current_video = "reference"
        # Load the test cases from a JSON file
        self.data = json.load(
            open(f"{CURR_DIR}/robosim_mfps/{self.env_name}.json"))
        self.current_time = 0
        self.current_question = 0
        self.choices_made = []

    def get_action_space(self):
        actions = ["rewind", "step", "next_video"]
        # Only allow choosing after all videos have been watched
        if self.current_video == "choice2" and self.is_finished:
            actions.extend(["choose_1", "choose_2"])
        else:
            actions.extend(["quit"])
        return actions

    def get_observation(self):
        question = self.data["questions"][self.current_question]

        # If we're in a finished state
        if self.is_finished and self.current_video == "choice2":
            return {
                "video":
                self.current_video,
                "frame_number":
                self.current_time,
                "total_frames":
                len(question[self.current_video]),
                "question_number":
                self.current_question + 1,
                "total_questions":
                len(self.data["questions"]),
                "is_finished":
                True,
                "message":
                "Which video (1 or 2) has the same environment dynamics as the reference?"
            }
        else:
            # Get current frame data - would be image data in your implementation
            current_frame = question[self.current_video][min(
                self.current_time,
                len(question[self.current_video]) - 1)]

            return {
                "video": self.current_video,
                "frame_number": self.current_time + 1,
                "total_frames": len(question[self.current_video]),
                "question_number": self.current_question + 1,
                "total_questions": len(self.data["questions"]),
                "image_data": current_frame,
                "is_finished": self.is_finished
            }

    def terminal(self):
        return self.is_terminal

    def step(self, action):
        question = self.data["questions"][self.current_question]

        if action == "rewind":
            self.current_time -= 1
            if self.current_time < 0:
                self.current_time = 0

        elif action == "step":
            self.current_time += 1
            if self.current_time >= len(question[self.current_video]):
                self.is_finished = True
                self.current_time = len(question[self.current_video]) - 1

        elif action == "next_video":
            if self.current_video == "reference":
                self.current_video = "choice1"
                self.current_time = 0
                self.is_finished = False
            elif self.current_video == "choice1":
                self.current_video = "choice2"
                self.current_time = 0
                self.is_finished = False

        elif action == "choose_1" or action == "choose_2":
            chosen_option = 1 if action == "choose_1" else 2
            correct_option = question["correct_choice"]

            # Record the choice
            self.choices_made.append({
                "question": self.current_question,
                "chosen": chosen_option,
                "correct": correct_option
            })

            # Calculate reward
            if chosen_option == correct_option:
                reward = 1
                self.score += 1
            else:
                reward = 0

            self.total_questions += 1

            # Move to next question or end if all questions answered
            if self.current_question < len(self.data["questions"]) - 1:
                self.current_question += 1
                self.current_video = "reference"
                self.current_time = 0
                self.is_finished = False
                return {
                    'image_data':
                    b'',
                    'message':
                    f"Correct! Moving to next question."
                    if reward else "Incorrect. Moving to next question."
                }, reward, False, {}
            else:
                self.is_terminal = True
                final_score = self.score / self.total_questions
                return {'image_data': b'', 'message': f"Test complete! Final score: {self.score}/{self.total_questions}"}, \
                       reward, True, {"final_score": final_score}

        elif action == "quit":
            self.is_terminal = True
            return {
                'image_data': b'',
                'message': "Quitting test early."
            }, 0, True, {}

        # Return current observation after action
        observation = self.get_observation()
        return {'image_data': observation.get('image_data', b''),
                'text_data': json.dumps({k: v for k, v in observation.items() if k != 'image_data'})}, \
               0, self.terminal(), {}


class MARAROBOSIMMFPServicer(MARAInteractiveServicer):

    def __init__(self):
        pass

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = MARAROBOSIMMFPEnvironment(
            request.config["env_name"])
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=
            f"Robosim MFP Environment initialized: {request.config['env_name']}"
        )
        reactive_env = env_pb2.ReactiveEnvironment(
            environment_id=f"{request.config['env_name']}_v1_robosim_mfp",
            version="1.0.0",
            metadata={
                "author": "MARA Developer",
                "domain": "Robosim"
            })
        response.reactive_env.CopyFrom(reactive_env)
        return response

    def Reset(self, request, context):
        if not hasattr(self, 'environment'):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        self.environment.reset()
        initial_obs = self.environment.get_observation()

        # Handle both image and text data
        response = env_service_pb2.ResetResponse(
            initial_observation=env_pb2.Observation(
                image_data=initial_obs.get('image_data', b''),
                text_data=json.dumps({
                    k: v
                    for k, v in initial_obs.items() if k != 'image_data'
                })),
            info={})
        return response

    def Step(self, request, context):
        if not hasattr(self, 'environment'):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        action_text = request.action.text_data
        try:
            observation, reward, is_terminal, info = self.environment.step(
                action_text)
        except Exception as e:
            logging.error(f"Error in step: {e} for action: {action_text}")
            observation, reward, is_terminal, info = {
                'image_data': b'',
                'text_data': '{}'
            }, 0, False, {}

        response = env_service_pb2.StepResponse(
            observation=env_pb2.Observation(
                image_data=observation.get('image_data', b''),
                text_data=observation.get('text_data', '{}')),
            reward=reward,
            is_terminal=is_terminal,
            info=info)
        return response
