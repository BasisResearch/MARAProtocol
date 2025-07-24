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
from .concrete_envs import InteractiveEnvironment, ChangeDetectionEnvironment, CDSliderEnvironment, PlanningEnvironment
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


class MARAChangeDetectionServicer(MARAInteractiveServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = ChangeDetectionEnvironment(request.config["env_name"])
        
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Change Detection Environment initialized: {request.config['env_name']}"
        )
        
        return response

    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{context['env_name']}_v1_cd",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )
        return response

class MARAChangeDetectionSliderServicer(MARAChangeDetectionServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = CDSliderEnvironment(request.config["env_name"],
                                               render_mode=request.config["render_mode"],
                                               logging_path=request.config["logging_path"],
                                               stack_frames=int(request.config["stack_frames"]),
                                               skip_frames=request.config["skip_frames"].lower() == "true",
                                               seed=int(request.config["seed"]),
                                               data_dir=request.config["data_dir"])
        response = env_service_pb2.InitializeResponse(
            success=True,
            message=f"Change Detection Environment initialized: {request.config['env_name']}"
        )
        
        return response

    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Environment not initialized")
        
        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id=f"{context['env_name']}_v1_cd_slider",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )
        return response

class MARAPlanningServicer(MARAInteractiveServicer):
    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.environment = PlanningEnvironment(request.config["env_name"],
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
            environment_id=f"{self.environment.env_name}_v1_planning",
            version="1.0.0",
            env_type=env_pb2.REACTIVE,
            capabilities={"text_input": "true", "text_output": "true"},
            metadata={"genre": "adventure", "difficulty": "beginner"}
        )

class MARACompositeAutumnChangeDetectionServicer(env_grpc.MARAEnvironmentServicer):
    def __init__(self):
        self.interactive_environment = MARAInteractiveServicer()
        self.change_detection_environment = MARAChangeDetectionSliderServicer()
        self.current_environment = self.interactive_environment
        self.env_name = None
        self.transiting = "Interactive" # InteractiveReset-> Interactive -> Transit -> ChangeReset -> Change -> End.

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.env_name = request.config["env_name"]
        self.max_interaction_steps = int(request.config["max_interaction_steps"])
        self.seed = request.config["seed"]
        self.data_dir = request.config["data_dir"]
        self.render_mode = request.config["render_mode"]
        self.logging_path = request.config["logging_path"]
        return self.current_environment.Initialize(request, context)

    def QuerySpaces(self, request, context):
        if self.transiting == "Interactive":
            return self.current_environment.QuerySpaces(request, context)
        elif self.transiting == "Transition" or self.transiting == "ChangeReset":
            response = env_service_pb2.SpaceQueryResponse(
                reactive_response=env_pb2.ReactiveEnvironment.ActionSpaceResponse(
                    action_space=env_pb2.ReactiveEnvironment.ActionSpace(
                        available_actions=[env_pb2.Action(text_data="noop")]
                    )
                )
            )
            return response
        elif self.transiting == "Change":
            return self.change_detection_environment.QuerySpaces(request, context)
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
            elif self.steps >= self.max_interaction_steps:
                self.transiting = "Transition"
                step_response.is_terminal = False
                self.steps = 0
            return step_response
        elif self.transiting == "Transition":
            self.transiting = "ChangeReset"
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Interactive environment ended, you will now transit to the defect detection environment."), reward=0, is_terminal=False, info={})
        elif self.transiting == "ChangeReset":
            self.steps = 0
            # Send initialize and reset to the new environment
            self.current_environment = self.change_detection_environment
            init_req = env_service_pb2.InitializeRequest(
                config={"env_name": self.env_name, "seed": self.seed, "data_dir": self.data_dir, "render_mode": self.render_mode, "logging_path": self.logging_path, "stack_frames": "1", "skip_frames": "false"}
            )
            self.change_detection_environment.Initialize(init_req, context)
            reset_req = env_service_pb2.ResetRequest()
            self.change_detection_environment.Reset(reset_req, context)
            observation = self.change_detection_environment.environment.get_observation()
            self.transiting = "Change"
            return env_service_pb2.StepResponse(observation=observation, reward=0, is_terminal=False, info={})
        elif self.transiting == "Change":
            step_response = self.change_detection_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "End"
                step_response.is_terminal = True
            return step_response
        elif self.transiting == "End":
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Change detection environment ended."), reward=0, is_terminal=True, info={})
        else:
            raise ValueError(f"Invalid transiting state: {self.transiting}")

    def Close(self, request, context):
        return env_service_pb2.CloseResponse(
            success=True,
            message="Environment closed successfully"
        )

class MARACompositeAutumnPlanningServicer(env_grpc.MARAEnvironmentServicer):
    def __init__(self):
        self.interactive_environment = MARAInteractiveServicer()
        self.planning_environment = MARAPlanningServicer()
        self.current_environment = self.interactive_environment
        self.env_name = None
        self.transiting = "Interactive" # InteractiveReset-> Interactive -> Transit -> PlanningReset -> Planning -> End.

    def Initialize(self, request: env_service_pb2.InitializeRequest, context):
        self.env_name = request.config["env_name"]
        self.max_interaction_steps = int(request.config["max_interaction_steps"])
        self.seed = request.config["seed"]
        self.data_dir = request.config["data_dir"]
        self.render_mode = request.config["render_mode"]
        self.logging_path = request.config["logging_path"]
        return self.current_environment.Initialize(request, context)

    def QuerySpaces(self, request, context):
        if self.transiting == "Interactive":
            return self.current_environment.QuerySpaces(request, context)
        elif self.transiting == "Transition" or self.transiting == "PlanningReset":
            response = env_service_pb2.SpaceQueryResponse(
                reactive_response=env_pb2.ReactiveEnvironment.ActionSpaceResponse(
                    action_space=env_pb2.ReactiveEnvironment.ActionSpace(
                        available_actions=[env_pb2.Action(text_data="noop")]
                    )
                )
            )
            return response
        elif self.transiting == "Planning":
            return self.planning_environment.QuerySpaces(request, context)
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
            elif self.steps >= self.max_interaction_steps:
                self.transiting = "Transition"
                step_response.is_terminal = False
                self.steps = 0
            return step_response
        elif self.transiting == "Transition":
            self.transiting = "PlanningReset"
            return env_service_pb2.StepResponse(observation=env_pb2.Observation(text_data="Interactive environment ended, you will now transit to the action prediction environment."), reward=0, is_terminal=False, info={})
        elif self.transiting == "PlanningReset":
            self.steps = 0
            # Send initialize and reset to the new environment
            self.current_environment = self.planning_environment
            init_req = env_service_pb2.InitializeRequest(
                config={"env_name": self.env_name, "seed": self.seed, "data_dir": self.data_dir, "render_mode": self.render_mode, "logging_path": self.logging_path, "stack_frames": "1", "skip_frames": "false"}
            )
            self.planning_environment.Initialize(init_req, context)
            reset_req = env_service_pb2.ResetRequest()
            self.planning_environment.Reset(reset_req, context)
            observation = self.planning_environment.environment.get_observation()
            self.transiting = "Planning"
            return env_service_pb2.StepResponse(observation=observation, reward=0, is_terminal=False, info={})
        elif self.transiting == "Planning":
            step_response = self.planning_environment.Step(request, context)
            observation, reward, is_terminal, info = step_response.observation, step_response.reward, step_response.is_terminal, step_response.info
            if is_terminal:
                self.transiting = "End"
                step_response.is_terminal = True
            # Commenting out to allow the max_steps during evaluation to be 
            # handled by the evaluation_controller run_environment
            # elif self.steps >= self.max_interaction_steps:
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

    server_change_detection = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    env_grpc.add_MARAEnvironmentServicer_to_server(
        MARAChangeDetectionServicer(), server_change_detection
    )

    port = 50052
    server_change_detection.add_insecure_port(f'[::]:{port}')
    server_change_detection.start()
    print(f"Autumn Change Detection Environment server started on port {port}")
    
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