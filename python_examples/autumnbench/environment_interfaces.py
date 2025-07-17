import logging
logging.basicConfig(level=logging.INFO)
import sys
import grpc
from concurrent import futures
import time

# Import generated protocol buffer code
from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_environment_service_pb2 as env_service_pb2
from generated.mara import mara_environment_service_pb2_grpc as env_grpc
from .concrete_envs import InteractiveEnvironment, DefectDetectionEnvironment, DDSliderEnvironment, ActionPredictionEnvironment
import os
# Setup logging to a file
logging.basicConfig(level=logging.INFO, filename="log_environment_interfaces.txt")
# Create file if it doesn't exist
if not os.path.exists("log_environment_interfaces.txt"):
    with open("log_environment_interfaces.txt", "w") as f:
        pass
# Filehandle for logging
fh = logging.FileHandler("log_environment_interfaces.txt")
logger = logging.getLogger(__name__)
logger.addHandler(fh)


class MARAInteractiveServicer(env_grpc.MARAEnvironmentServicer):
    def __init__(self):
        pass
    
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = InteractiveEnvironment(
            request.config["env_name"], 
            stack_frames=int(request.config["stack_frames"]), 
            skip_frames=request.config["skip_frames"].lower() == "true",
            render_mode=request.config["render_mode"],
            logging_path=request.config["logging_path"],
            seed=int(request.config["seed"]),
            data_dir=request.config["data_dir"]
        )
        
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Interactive Environment initialized: {request.config['env_name']}"
        )
        
        # Set environment-specific fields
        reactive_env = env_pb2.ReactiveEnvironment(
            environment_id=f"{request.config['env_name']}_v1_interactive",
            version="1.0.0",
            metadata={"author": "MARA Developer", "domain": "Autumn"}
        )
        
        response.reactive_env.CopyFrom(reactive_env)
        return response
    
    def Reset(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        self.environment.reset()
        initial_obs = self.environment.get_observation()
        
        response = env_service_pb2.ResetResponse(
            initial_observation=initial_obs,
            info={}
        )
        return response
    
    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{context['env_name']}_v1",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )
        return response
    
    def Step(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        try:
            observation, reward, is_terminal, info = self.environment.step(request.action)
        except Exception as e:
            logging.error(f"Error in step: {e} for action: {request.action}, env: {self.environment}")
            info = {}
        
        info_dict = {}
        if "message" in info:
            info_dict["message"] = info["message"]
        
        response = env_service_pb2.StepResponse(
            observation=observation,
            reward=reward,
            is_terminal=is_terminal,
            info=info_dict
        )
        return response
    
    def QuerySpaces(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        available_actions = self.environment.get_action_space()
        
        # Create the action space with constraints as a proper map field
        constraints_map = {}
        constraints_map["type"] = "discrete"
        constraints_map["text_commands"] = "true"
        
        action_space = env_pb2.ReactiveEnvironment.ActionSpace(
            available_actions=available_actions,
            constraints=constraints_map,
            is_continuous=False
        )
        
        response = env_service_pb2.SpaceQueryResponse(
            reactive_response=env_pb2.ReactiveEnvironment.ActionSpaceResponse(
                action_space=action_space
            )
        )
        return response
    
    def Close(self, request, context):
        self.environment = None
        return env_service_pb2.CloseResponse(
            success=True,
            message="Environment closed successfully"
        )


class MARADefectDetectionServicer(MARAInteractiveServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = DefectDetectionEnvironment(request.config["env_name"])
        
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Defect Detection Environment initialized: {request.config['env_name']}"
        )
        
        return response

    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{context['env_name']}_v1_dd",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )
        return response

class MARADefectDetectionSliderServicer(MARADefectDetectionServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = DDSliderEnvironment(request.config["env_name"],
                                               render_mode=request.config["render_mode"],
                                               logging_path=request.config["logging_path"],
                                               stack_frames=int(request.config["stack_frames"]),
                                               skip_frames=request.config["skip_frames"].lower() == "true",
                                               seed=int(request.config["seed"]),
                                               data_dir=request.config["data_dir"])
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Defect Detection Environment initialized: {request.config['env_name']}"
        )
        
        return response

    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{context['env_name']}_v1_dd_slider",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )
        return response

class MARAActionPredictionServicer(MARAInteractiveServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = ActionPredictionEnvironment(request.config["env_name"],
                                                      render_mode=request.config["render_mode"],
                                                      logging_path=request.config["logging_path"],
                                                      stack_frames=int(request.config["stack_frames"]),
                                                      skip_frames=request.config["skip_frames"].lower() == "true",
                                                      seed=int(request.config["seed"]),
                                                      data_dir=request.config["data_dir"])
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Action Prediction Environment initialized: {request.config['env_name']}"
        )
        
        return response

    def GetEnvironmentInfo(self, request, context):
        return env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{self.environment.env_name}_v1_action_prediction",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )

class MARACompositeAutumnDefectDetectionServicer(env_grpc.MARAEnvironmentServicer):
    def __init__(self):
        self.interactive_environment = MARAInteractiveServicer()
        self.defect_detection_environment = MARADefectDetectionSliderServicer()
        self.current_environment = self.interactive_environment
        self.env_name = None
        self.transiting = "Interactive" # InteractiveReset-> Interactive -> Transit -> DefectReset -> Defect -> End.

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.env_name = request.config["env_name"]
        self.per_env_max_steps = int(request.config["per_env_max_steps"])
        self.seed = request.config["seed"]
        self.data_dir = request.config["data_dir"]
        self.render_mode = request.config["render_mode"]
        self.logging_path = request.config["logging_path"]
        return self.current_environment.Initialize(request, context)

    def QuerySpaces(self, request, context):
        if self.transiting == "Interactive":
            return self.current_environment.QuerySpaces(request, context)
        elif self.transiting == "Transition" or self.transiting == "DefectReset":
            response = env_service_pb2.SpaceQueryResponse(
                reactive_response=env_pb2.ReactiveEnvironment.ActionSpaceResponse(
                    action_space=env_pb2.ReactiveEnvironment.ActionSpace(
                        available_actions=[env_pb2.Action(text_data="NOP")]
                    )
                )
            )
            return response
        elif self.transiting == "Defect":
            return self.defect_detection_environment.QuerySpaces(request, context)
        else:
            raise ValueError(f"Invalid transiting state: {self.transiting}")

    def Reset(self, request, context):
        self.steps = 0
        return self.current_environment.Reset(request, context)

    def GetEnvironmentInfo(self, request, context):
        return self.current_environment.GetEnvironmentInfo(request, context)
    
    def Step(self, request, context):
        self.steps += 1
        if self.transiting == "Interactive":
            step_response = self.current_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "Transition"
                step_response.is_terminal = False
            elif self.steps >= self.per_env_max_steps:
                self.transiting = "Transition"
                step_response.is_terminal = False
                self.steps = 0
            return step_response
        elif self.transiting == "Transition":
            self.transiting = "DefectReset"
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Interactive environment ended, you will now transit to the defect detection environment."), reward=0, is_terminal=False, info={})
        elif self.transiting == "DefectReset":
            self.steps = 0
            # Send initialize and reset to the new environment
            self.current_environment = self.defect_detection_environment
            init_req = env_service_pb2.InitializeRequest(
                config={"env_name": self.env_name, "seed": self.seed, "data_dir": self.data_dir, "render_mode": self.render_mode, "logging_path": self.logging_path, "stack_frames": "1", "skip_frames": "false"}
            )
            self.defect_detection_environment.Initialize(init_req, context)
            reset_req = env_service_pb2.ResetRequest()
            self.defect_detection_environment.Reset(reset_req, context)
            observation = self.defect_detection_environment.environment.get_observation()
            self.transiting = "Defect"
            return env_service_pb2.StepResponse(observation=observation, reward=0, is_terminal=False, info={})
        elif self.transiting == "Defect":
            step_response = self.defect_detection_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "End"
                step_response.is_terminal = True
            return step_response
        elif self.transiting == "End":
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Defect detection environment ended."), reward=0, is_terminal=True, info={})
        else:
            raise ValueError(f"Invalid transiting state: {self.transiting}")

    def Close(self, request, context):
        return env_service_pb2.CloseResponse(
            success=True,
            message="Environment closed successfully"
        )

class MARACompositeAutumnActionPredictionServicer(env_grpc.MARAEnvironmentServicer):
    def __init__(self):
        self.interactive_environment = MARAInteractiveServicer()
        self.action_prediction_environment = MARAActionPredictionServicer()
        self.current_environment = self.interactive_environment
        self.env_name = None
        self.transiting = "Interactive" # InteractiveReset-> Interactive -> Transit -> ActionPredictionReset -> ActionPrediction -> End.

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.env_name = request.config["env_name"]
        self.per_env_max_steps = int(request.config["per_env_max_steps"])
        self.seed = request.config["seed"]
        self.data_dir = request.config["data_dir"]
        self.render_mode = request.config["render_mode"]
        self.logging_path = request.config["logging_path"]
        return self.current_environment.Initialize(request, context)

    def QuerySpaces(self, request, context):
        if self.transiting == "Interactive":
            return self.current_environment.QuerySpaces(request, context)
        elif self.transiting == "Transition" or self.transiting == "ActionPredictionReset":
            response = env_service_pb2.SpaceQueryResponse(
                reactive_response=env_pb2.ReactiveEnvironment.ActionSpaceResponse(
                    action_space=env_pb2.ReactiveEnvironment.ActionSpace(
                        available_actions=[env_pb2.Action(text_data="NOP")]
                    )
                )
            )
            return response
        elif self.transiting == "ActionPrediction":
            return self.action_prediction_environment.QuerySpaces(request, context)
        else:
            raise ValueError(f"Invalid transiting state: {self.transiting}")

    def Reset(self, request, context):
        self.steps = 0
        return self.current_environment.Reset(request, context)

    def GetEnvironmentInfo(self, request, context):
        return self.current_environment.GetEnvironmentInfo(request, context)
    
    def Step(self, request, context):
        self.steps += 1
        if self.transiting == "Interactive":
            step_response = self.current_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "Transition"
                step_response.is_terminal = False
            elif self.steps >= self.per_env_max_steps:
                self.transiting = "Transition"
                step_response.is_terminal = False
                self.steps = 0
            return step_response
        elif self.transiting == "Transition":
            self.transiting = "ActionPredictionReset"
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Interactive environment ended, you will now transit to the action prediction environment."), reward=0, is_terminal=False, info={})
        elif self.transiting == "ActionPredictionReset":
            self.steps = 0
            # Send initialize and reset to the new environment
            self.current_environment = self.action_prediction_environment
            init_req = env_service_pb2.InitializeRequest(
                config={"env_name": self.env_name, "seed": self.seed, "data_dir": self.data_dir, "render_mode": self.render_mode, "logging_path": self.logging_path, "stack_frames": "1", "skip_frames": "false"}
            )
            self.action_prediction_environment.Initialize(init_req, context)
            reset_req = env_service_pb2.ResetRequest()
            self.action_prediction_environment.Reset(reset_req, context)
            observation = self.action_prediction_environment.environment.get_observation()
            self.transiting = "ActionPrediction"
            return env_service_pb2.StepResponse(observation=observation, reward=0, is_terminal=False, info={})
        elif self.transiting == "ActionPrediction":
            step_response = self.action_prediction_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "End"
                step_response.is_terminal = True
            # Commenting out to allow the max_steps during evaluation to be 
            # handled by the evaluation_controller run_environment
            # elif self.steps >= self.per_env_max_steps:
            #     self.transiting = "End"
            #     step_response.is_terminal = True
            #     self.steps = 0
            return step_response
        elif self.transiting == "End":
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Action prediction environment ended."), reward=0, is_terminal=True, info={})
        else:
            raise ValueError(f"Invalid transiting state: {self.transiting}")

    def Close(self, request, context):
        return env_service_pb2.CloseResponse(
            success=True,
            message="Environment closed successfully"
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    env_grpc.add_MARAEnvironmentServicer_to_server(
        MARAInteractiveServicer(), server
    )
    
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Autumn Interactive Environment server started on port {port}")

    server_defect_detection = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    env_grpc.add_MARAEnvironmentServicer_to_server(
        MARADefectDetectionServicer(), server_defect_detection
    )

    port = 50052
    server_defect_detection.add_insecure_port(f'[::]:{port}')
    server_defect_detection.start()
    print(f"Autumn Defect Detection Environment server started on port {port}")
    
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

def test_interactive_environment():
    environment = InteractiveEnvironment("ants")
    environment.reset()
    print(environment.step("left"))
    environment.step("click 1 1")
    print(environment.get_observation())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # First, try an interactive environment 
    # serve()
    test_interactive_environment()