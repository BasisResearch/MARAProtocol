from .agent import agent_grpc, env_pb2, agent_pb2
from .llm_utils import AIFactory, extract_json_response, extract_tagged_response, Message
from .prompts import (
    ACTION_PROMPT_REACT, 
    ACTION_PROMPT_REFLEXION, 
    RESPONSE_PROMPT_SCRATCHPAD, 
    RESPONSE_PROMPT_DEFAULT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_HINT
)

from typing import List, Tuple
import logging
import grpc
from concurrent import futures
import time
import argparse
import os
import json
import base64
# Log to file
logger = logging.getLogger(__name__)


class ReactVLMAgent(agent_grpc.MARAAgentServicer):
    def Initialize(self, request, context):
        super().__init__()
        self.thought: List[str] = []
        self.action: List[str] = []
        self.observation: List[env_pb2.Observation] = []
        self.init_observation = None
        logger.info(f"Initializing agent with model: {request.config}")
        self.llm = AIFactory.create_provider(request.config.get("llm_provider", "openai"), request.config.get("llm_model", "openai/gpt-4o"))
        self.env_name = request.config.get("env_name", "environment")
        self.logging_path = request.config.get("logging_path", "./logs")
        self.max_history_length = int(request.config.get("max_history_length", -1))
        self.history = []

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="MARAReactVLMAgent initialized",
            agent_id="MARAReactVLMAgent",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            }
        )
    
    def Reset(self, request, context):
        self.thought = []
        self.action = []
        self.observation = []
        self.init_observation = None
        self.history = []
        self.trajectory = {
            "observations": [],
        }
        return agent_pb2.AgentResetResponse(
            success=True,
            message="MARAReactVLMAgent reset"
        )
    
    def Act(self, request, context) -> agent_pb2.ActResponse:
        observation = request.observation
        available_actions = request.reactive_action_space.available_actions
        if observation.image_data == b'' and observation.text_data == "":
            return agent_pb2.ActResponse(
                action=env_pb2.Action(text_data="step"),
                confidence=0.5,
                metadata={"strategy": "fallback"}
            )
        imgs = None
        if observation.image_data != b'':
            try:
                # Try to decode as JSON first (for MCQ with options)
                image_json_str = observation.image_data.decode('utf-8')
                image_data = json.loads(image_json_str)
                if "options" in image_data and "grid" in image_data:
                    # This is MCQ data with options
                    imgs = image_data["options"] + [image_data["grid"]]
                elif "grid" in image_data and "goal_state" in image_data:
                    # This is ActionPredictionEnvironment data with grid and goal state
                    imgs = [image_data["grid"], image_data["goal_state"]]
                else:
                    # Regular single image
                    img = base64.b64encode(observation.image_data).decode('utf-8')
                    imgs = [img]
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to treating as regular image data
                img = base64.b64encode(observation.image_data).decode('utf-8')
                imgs = [img]

        if len(self.thought) == 0:
            self.init_observation = observation
            self.history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            self.history.append({
                "role": "user",
                "content": f"""Observation:\n{observation.text_data}\n\n
                Available actions: {",\n".join([a.text_data for a in available_actions])}
                Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
                Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
                """
            })
        else:
            if imgs is not None and len(imgs) > 1:
                # Check if this is MCQ with options or ActionPrediction with goal state
                try:
                    image_json_str = observation.image_data.decode('utf-8')
                    image_data = json.loads(image_json_str)
                    if "options" in image_data and "grid" in image_data:
                        # MCQ case with options
                        content_parts = [
                            {
                                "type": "text",
                                "text": f"""Observation:\n{observation.text_data}\n\nCurrent grid:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{imgs[-1]}"  # Grid is last
                            }
                        ]
                        
                        # Add options
                        for i in range(len(imgs[:-1])):
                            content_parts.extend([
                                {
                                    "type": "text",
                                    "text": f"""Option {i}:"""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{imgs[i]}"
                                }
                            ])
                    elif "grid" in image_data and "goal_state" in image_data:
                        # ActionPrediction case with goal state
                        content_parts = [
                            {
                                "type": "text",
                                "text": f"""Observation:\n{observation.text_data}\n\nCurrent grid:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{imgs[0]}"  # Current grid
                            },
                            {
                                "type": "text",
                                "text": """Goal state:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{imgs[1]}"  # Goal state
                            }
                        ]
                    else:
                        # Fallback - treat as generic multi-image
                        content_parts = [
                            {
                                "type": "text",
                                "text": f"""Observation:\n{observation.text_data}\n\n"""
                            }
                        ]
                        for i, img in enumerate(imgs):
                            content_parts.extend([
                                {
                                    "type": "text",
                                    "text": f"""Image {i}:"""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{img}"
                                }
                            ])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fallback - treat as generic multi-image
                    content_parts = [
                        {
                            "type": "text",
                            "text": f"""Observation:\n{observation.text_data}\n\n"""
                        }
                    ]
                    for i, img in enumerate(imgs):
                        content_parts.extend([
                            {
                                "type": "text",
                                "text": f"""Image {i}:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{img}"
                            }
                        ])
                
                content_parts.append({
                    "type": "text",
                    "text": f"""Available actions: {",\n".join([a.text_data for a in available_actions])}
                            Think carefully, step by step about the next action that should be taken. 
                            Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
                            """
                })
                
                self.history.append({
                    "role": "user",
                    "content": content_parts
                })
            elif imgs is not None and len(imgs) == 1:
                # Single image case
                self.history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Observation:\n{observation.text_data}\n\n"""
                            
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{imgs[0]}"
                        },
                        {
                            "type": "text",
                            "text": f"""Available actions: {",\n".join([a.text_data for a in available_actions])}
                            Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
                            Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
                            """
                        }
                    ]
                })
            else:
                # Text only case
                self.history.append({
                    "role": "user",
                    "content": f"""Observation:\n{observation.text_data}\n\n
                    Available actions: {",\n".join([a.text_data for a in available_actions])}
                    Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
                    Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
                    """
                })

        self.observation.append(observation)
        response = self.llm.generate_from_history(self.history)

        self.history.append({
            "role": "assistant",
            "content": response
        })
        
        logger.info(f"Response: {response}")
        
        response = extract_tagged_response(response)

        if not response or "action" not in response or "thought" not in response:
            return agent_pb2.ActResponse(
                action=env_pb2.Action(text_data="NOP"),
                confidence=0.5,
                metadata={"strategy": "fallback"}
            )
        
        thought = response["thought"]
        action = env_pb2.Action(text_data=response["action"])
        self.thought.append(thought)
        self.action.append(action)
        if len(self.history) > 4:
            action_log = action.text_data.split(" ")[0] if action.text_data.startswith("click") else action.text_data
            idx = None if action_log != "click" else [int(i) for i in action.text_data.split(" ")[1:]]
            self.trajectory["observations"].append({
                "time_step": len(self.trajectory["observations"]) + 1,
                "grid": observation.text_data,
                "action": action.text_data,
                "action_idx": idx,
                "thinking": thought
            })
        
        return agent_pb2.ActResponse(
            action=action,
            confidence=0.8,
            metadata={"strategy": "exploration"}
        )        

    def Feedback(self, request, context) -> agent_pb2.FeedbackResponse:
        # TODO: Consider the feedback and update the agent observation
        return agent_pb2.FeedbackResponse(
            acknowledged=True
        )
    
    def EndEpisode(self, request, context) -> agent_pb2.EndEpisodeResponse:
        # TODO: Consider the end of episode and update the agent observation
        with open(os.path.join(self.logging_path, f"{self.env_name}_history.json"), "w") as f:
            json.dump(self.history, f)
        with open(os.path.join(self.logging_path, f"{self.env_name}_trajectory.json"), "w") as f:
            json.dump(self.trajectory, f)
        
        return agent_pb2.EndEpisodeResponse(
            acknowledged=True
        )
    
    def GetAgentInfo(self, request, context) -> agent_pb2.AgentInfoResponse:
        return agent_pb2.AgentInfoResponse(
            agent_id="MARALLMInteractiveAgent",
            version="1.0.0",
            agent_type=agent_pb2.POLICY,
            compatible_environment_types=["REACTIVE"],
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            },
            metadata={
                "author": "MARA Developer",
                "domain": "Autumn",
                "description": "A simple interactive agent that uses LLM to think and reason about what action to take next"
            }
        )
    
    def Close(self, request, context):
        
        return agent_pb2.CloseResponse(
            success=True,
            message="MARAReactVLMAgent closed"
        )



class SummaryReactLLMAgent(agent_grpc.MARAAgentServicer):
    def Initialize(self, request, context):
        super().__init__()
        self.thought: List[str] = []
        self.action: List[str] = []
        self.observation: List[env_pb2.Observation] = []
        self.init_observation = None
        logger.info(f"Initializing agent with model: {request.config}")
        self.llm = AIFactory.create_provider(request.config.get("llm_provider", "openai"), request.config.get("llm_model", "openai/gpt-4o"))
        self.env_name = request.config.get("env_name", "environment")
        self.logging_path = request.config.get("logging_path", "./logs")
        self.max_history_length = int(request.config.get("max_history_length", -1))
        self.history = []

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="SummaryReactLLMAgent initialized",
            agent_id="SummaryReactLLMAgent",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            }
        )
    
    def summarize_history(self) -> str:
        messages = self.history[1:] + [{
            "role": "user",
            "content": f"""Summarize the history of this conversation thus far. 
            Make sure to include all the information about the actions you have taken, observations you have make and the underlying rules of the environment that you have inferred.
            """
        }]
        response = self.llm.generate_from_history(messages)
        summary = f"Here is the summary of the first {len(self.history) - 1 / 2} steps of the interaction with the environment: " + response
        return summary

    def Reset(self, request, context):
        self.thought = []
        self.action = []
        self.observation = []
        self.init_observation = None
        self.history = []
        self.trajectory = {
            "observations": [],
        }
        self.summaries = []
        self.messages = []
        return agent_pb2.AgentResetResponse(
            success=True,
            message="SummaryReactLLMAgent reset"
        )
    
    def Act(self, request, context) -> agent_pb2.ActResponse:
        observation = request.observation
        available_actions = request.reactive_action_space.available_actions
        
        if len(self.history) == 0:
            self.init_observation = observation
            self.history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            self.messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            
        if (len(self.messages) - 1) / 2 > self.max_history_length:
            summary = self.summarize_history()
            self.summaries.append(summary)
            self.messages = [self.messages[0]] + [{"role": "user", "content": summary}]
            logger.info(f"Summary: {summary}")
            logger.info(f"Messages: {self.messages}")
        
        self.history.append({
            "role": "user",
            "content": f"""Observation:\n{observation}\n\n
            Available actions: {",\n".join([a.text_data for a in available_actions])}
            Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
            Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
            """
        })
        self.messages.append({
            "role": "user",
            "content": f"""Observation:\n{observation}\n\n
            Available actions: {",\n".join([a.text_data for a in available_actions])}
            Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
            """
        })

        self.observation.append(observation)
        response = self.llm.generate_from_history(self.messages)

        self.history.append({
            "role": "assistant",
            "content": response
        })
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        
        logger.info(f"Response: {response}")
        
        response = extract_tagged_response(response)

        if not response or "action" not in response or "thought" not in response:
            return agent_pb2.ActResponse(
                action=env_pb2.Action(text_data="NOP"),
                confidence=0.5,
                metadata={"strategy": "fallback"}
            )
        
        thought = response["thought"]
        action = env_pb2.Action(text_data=response["action"])
        self.thought.append(thought)
        self.action.append(action)
        logger.info(f"{"*" * 100}")
        logger.info(f"History: {len(self.history)}")
        if len(self.history) > 4:
            action_log = action.text_data.split(" ")[0] if action.text_data.startswith("click") else action.text_data
            idx = None if action_log != "click" else [int(i) for i in action.text_data.split(" ")[1:]]
            logger.info(f"{"*" * 100}")
            self.trajectory["observations"].append({
                "time_step": len(self.trajectory["observations"]) + 1,
                "grid": observation,
                "action": action,
                "action_idx": idx,
                "thinking": thought
            })
        
        return agent_pb2.ActResponse(
            action=action,
            confidence=0.8,
            metadata={"strategy": "exploration"}
        )        

    def Feedback(self, request, context) -> agent_pb2.FeedbackResponse:
        # TODO: Consider the feedback and update the agent observation
        return agent_pb2.FeedbackResponse(
            acknowledged=True
        )
    
    def EndEpisode(self, request, context) -> agent_pb2.EndEpisodeResponse:
        # TODO: Consider the end of episode and update the agent observation
        with open(os.path.join(self.logging_path, f"{self.env_name}_history.json"), "w") as f:
            json.dump(self.history, f)
        with open(os.path.join(self.logging_path, f"{self.env_name}_trajectory.json"), "w") as f:
            json.dump(self.trajectory, f)
        with open(os.path.join(self.logging_path, f"{self.env_name}_summaries.json"), "w") as f:
            json.dump(self.summaries, f)
        
        return agent_pb2.EndEpisodeResponse(
            acknowledged=True
        )
    
    def GetAgentInfo(self, request, context) -> agent_pb2.AgentInfoResponse:
        return agent_pb2.AgentInfoResponse(
            agent_id="MARALLMInteractiveAgent",
            version="1.0.0",
            agent_type=agent_pb2.POLICY,
            compatible_environment_types=["REACTIVE"],
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            },
            metadata={
                "author": "MARA Developer",
                "domain": "Autumn",
                "description": "A simple interactive agent that uses LLM to think and reason about what action to take next"
            }
        )
    
    def Close(self, request, context):
        
        return agent_pb2.CloseResponse(
            success=True,
            message="SummaryReactLLMAgent closed"
        )






class ReactLLMAgent2(agent_grpc.MARAAgentServicer):
    def Initialize(self, request, context):
        super().__init__()
        self.thought: List[str] = []
        self.action: List[str] = []
        self.observation: List[env_pb2.Observation] = []
        self.init_observation = None
        logger.info(f"Initializing agent with model: {request.config}")
        self.llm = AIFactory.create_provider(request.config.get("llm_provider", "openai"), request.config.get("llm_model", "openai/gpt-4o"))
        self.env_name = request.config.get("env_name", "environment")
        self.logging_path = request.config.get("logging_path", "./logs")
        self.max_history_length = int(request.config.get("max_history_length", -1))
        self.history = []

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="MARAReactLLMAgent2 initialized",
            agent_id="MARAReactLLMAgent2",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            }
        )
    
    def Reset(self, request, context):
        self.thought = []
        self.action = []
        self.observation = []
        self.init_observation = None
        self.history = []
        self.trajectory = {
            "observations": [],
        }
        return agent_pb2.AgentResetResponse(
            success=True,
            message="MARAReactLLMAgent2 reset"
        )
    
    def Act(self, request, context) -> agent_pb2.ActResponse:
        observation = request.observation
        available_actions = request.reactive_action_space.available_actions
        
        if len(self.thought) == 0:
            self.init_observation = observation
            self.history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        self.history.append({
            "role": "user",
            "content": f"""Observation:\n{observation}\n\n
            Available actions: {",\n".join([a.text_data for a in available_actions])}
            Think carefully, step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules.
            Output your thinking and reasoning within a <think> tag and the action within a <action> tag.
            """
        })

        self.observation.append(observation)
        response = self.llm.generate_from_history(self.history)

        self.history.append({
            "role": "assistant",
            "content": response
        })
        
        logger.info(f"Response: {response}")
        
        response = extract_tagged_response(response)

        if not response or "action" not in response or "thought" not in response:
            return agent_pb2.ActResponse(
                action=env_pb2.Action(text_data="NOP"),
                confidence=0.5,
                metadata={"strategy": "fallback"}
            )
        
        thought = response["thought"]
        action = env_pb2.Action(text_data=response["action"])
        self.thought.append(thought)
        self.action.append(action)
        if len(self.history) > 4:
            action_log = action.text_data.split(" ")[0] if action.text_data.startswith("click") else action.text_data
            idx = None if action_log != "click" else [int(i) for i in action.text_data.split(" ")[1:]]
            self.trajectory["observations"].append({
                "time_step": len(self.trajectory["observations"]) + 1,
                "grid": observation.text_data,
                "action": action.text_data,
                "action_idx": idx,
                "thinking": thought
            })
        
        return agent_pb2.ActResponse(
            action=action,
            confidence=0.8,
            metadata={"strategy": "exploration"}
        )        

    def Feedback(self, request, context) -> agent_pb2.FeedbackResponse:
        # TODO: Consider the feedback and update the agent observation
        return agent_pb2.FeedbackResponse(
            acknowledged=True
        )
    
    def EndEpisode(self, request, context) -> agent_pb2.EndEpisodeResponse:
        # TODO: Consider the end of episode and update the agent observation
        with open(os.path.join(self.logging_path, f"{self.env_name}_history.json"), "w") as f:
            json.dump(self.history, f)
        with open(os.path.join(self.logging_path, f"{self.env_name}_trajectory.json"), "w") as f:
            json.dump(self.trajectory, f)
        
        return agent_pb2.EndEpisodeResponse(
            acknowledged=True
        )
    
    def GetAgentInfo(self, request, context) -> agent_pb2.AgentInfoResponse:
        return agent_pb2.AgentInfoResponse(
            agent_id="MARALLMInteractiveAgent",
            version="1.0.0",
            agent_type=agent_pb2.POLICY,
            compatible_environment_types=["REACTIVE"],
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            },
            metadata={
                "author": "MARA Developer",
                "domain": "Autumn",
                "description": "A simple interactive agent that uses LLM to think and reason about what action to take next"
            }
        )
    
    def Close(self, request, context):
        
        return agent_pb2.CloseResponse(
            success=True,
            message="MARAReactLLMAgent2 closed"
        )


class ReactLLMAgentServicer(agent_grpc.MARAAgentServicer):
    def Initialize(self, request, context):
        super().__init__()
        self.thought: List[str] = []
        self.action: List[str] = []
        self.observation: List[env_pb2.Observation] = []
        self.init_observation = None
        logger.info(f"Initializing agent with model: {request.config}")
        self.llm = AIFactory.create_provider(request.config.get("llm_provider", "openai"), request.config.get("llm_model", "openai/gpt-4o"))
        self.env_name = request.config.get("env_name", "environment")
        self.logging_path = request.config.get("logging_path", "./logs")
        self.max_history_length = int(request.config.get("max_history_length", -1))
        self.history = []
        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="MARAReactLLMAgent initialized",
            agent_id="MARAReactLLMAgentServicer",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            }
        )
    
    def Reset(self, request, context):
        self.thought = []
        self.action = []
        self.observation = []
        self.init_observation = None
        return agent_pb2.AgentResetResponse(
            success=True,
            message="MARAReactLLMAgentServicer reset"
        )

    def format_trace(self) -> List[Message]:
        messages: List[Message] = []
        for i, (thought, action, observation) in enumerate(zip(self.thought, self.action, self.observation)):
            message_content = []
            message_base64_images = []
            if thought != "":
                message_content.append(f"Thought {i+1}:\n{thought}")
            if action != "":
                message_content.append(f"Action {i+1}:\n{action.text_data}")
            if observation != "":
                if observation.text_data != "":
                    message_content.append(f"Observation {i+1}:\n{observation.text_data}")
                if observation.image_data != b"":
                    message_base64_images.append(observation.image_data)
            message_content = "\n".join(message_content)
            messages.append(Message(role="assistant", content=message_content, base64_images=message_base64_images))
        # Every time called, write the trace to a file
        # with open("trace.txt", "w") as f:
        #     for message in messages:
        #         f.write(f"{message.role}: {message.content}\n")
        logging.info(f"length of messages: {len(messages[-self.max_history_length:])}")
        return messages[:-self.max_history_length] if self.max_history_length > 0 else messages


    def think_action_initial(self, observation, available_actions) -> Tuple[str, env_pb2.Action]:
        system_prompt = "You are an agent exploring a grid like environment. You will be given observations and available actions to choose from at each step. Your goal is to interact with the environment and try to understand the underlying rules of the environment."
        prompt = f"""
        You are given the following initial observation and a list of available actions.
        
        Observations: {observation}
        
        Available actions:
        {",\n".join([a.text_data for a in available_actions])}

        You need to think and reason about what action to take next to explore and understand the environment.
        Write your thought within a <thought> tag and your action within a <action> tag.
        """
        response, inp_data = self.llm.generate(system_prompt=system_prompt, content=prompt)
        self.history.append(inp_data)
        response = extract_tagged_response(response)
        if not response or "action" not in response or "thought" not in response:
            # Fallback to a safe default if parsing fails
            return "Could not determine next action", env_pb2.Action(text_data="NOP")
        
        # Create the action directly from the response
        action = env_pb2.Action(text_data=response["action"])
        return response["thought"], action

    def Act(self, request, context) -> agent_pb2.ActResponse:
        observation = request.observation
        available_actions = request.reactive_action_space.available_actions
        try:
            if len(self.thought) == 0:
                self.init_observation = observation
                thought, action = self.think_action_initial(observation, available_actions)
                self.thought.append(thought)
                self.action.append(action)
                return agent_pb2.ActResponse(
                    action=action,
                    confidence=0.8,
                    metadata={"strategy": "exploration"}
                )
            else:
                self.observation.append(observation)
                system_prompt = f"""You are an agent exploring a grid like environment. You will be given observations and available actions to choose from at each step. Your goal is to interact with the environment and try to understand the underlying rules of the environment.

                Your thoughts, actions, and observations of the last {self.max_history_length} steps are:
                Observation 0: {self.init_observation}
                """ if self.max_history_length > 0 else f"""You are an agent exploring a grid like environment. You will be given observations and available actions to choose from at each step. Your goal is to interact with the environment and try to understand the underlying rules of the environment.

                Your thoughts, actions, and observations of all the previous steps are:
                Observation 0: {self.init_observation}
                
                """
                prompt_parts = self.format_trace()
                prompt_parts.append(Message(role="user", content=f"""
                Now, you are given the following actions you can take: {",\n".join([a.text_data for a in available_actions])}
                Choose the next action to take and reason about it.
                Write your thought within a <thought> tag and your action within a <action> tag.                
                """))
                response, inp_data = self.llm.generate_formatted_prompt(system_prompt, prompt_parts)
                self.history.append(inp_data)
                logger.info(f"Response: {response}")
                response = extract_tagged_response(response)
                if not response or "action" not in response or "thought" not in response:
                    # Fallback to a safe default
                    return agent_pb2.ActResponse(
                        action=env_pb2.Action(text_data="NOP"),
                        confidence=0.5,
                        metadata={"strategy": "fallback"}
                    )
                
                thought = response["thought"]
                action = env_pb2.Action(text_data=response["action"])
                self.thought.append(thought)
                self.action.append(action)
                return agent_pb2.ActResponse(
                    action=action,
                    confidence=0.8,
                    metadata={"strategy": "exploration"}
                )
        except Exception as e:
            logger.error(f"Error in Act: {e}")
            # Fallback to a safe default action
            return agent_pb2.ActResponse(
                action=env_pb2.Action(text_data="NOP"),
                confidence=0.5,
                metadata={"strategy": "error_fallback"}
            )

    def Feedback(self, request, context) -> agent_pb2.FeedbackResponse:
        # TODO: Consider the feedback and update the agent observation
        return agent_pb2.FeedbackResponse(
            acknowledged=True
        )
    
    def GetAgentInfo(self, request, context) -> agent_pb2.AgentInfoResponse:
        return agent_pb2.AgentInfoResponse(
            agent_id="MARALLMInteractiveAgent",
            version="1.0.0",
            agent_type=agent_pb2.POLICY,
            compatible_environment_types=["REACTIVE"],
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            },
            metadata={
                "author": "MARA Developer",
                "domain": "Autumn",
                "description": "A simple interactive agent that uses LLM to think and reason about what action to take next"
            }
        )
    
    def Close(self, request, context):
        
        return agent_pb2.CloseResponse(
            success=True,
            message="MARAReactLLMAgentServicer closed"
        )

class UnifiedReactAgent(ReactLLMAgentServicer):
    def Initialize(self, request, context):
        super().__init__()
        self.thought: List[str] = []
        self.action: List[str] = []
        self.observation: List[env_pb2.Observation] = []
        self.init_observation = None
        logger.info(f"Initializing agent with model: {request.config}")
        
        self.llm = AIFactory.create_provider(request.config.get("llm_provider", "openai"), request.config.get("llm_model", "openai/gpt-4o"))
        self.env_name = request.config.get("env_name", "environment")
        self.logging_path = request.config.get("logging_path", "./logs")
        self.max_history_length = int(request.config.get("max_history_length", -1))
        self.use_scratchpad = request.config.get("use_scratchpad", "false").lower() == "true"
        self.instruction_type = request.config.get("instruction_type", "react")
        self.hint = request.config.get("hint", "false").lower() == "true"
        self.task_name = request.config.get("task_name", "mcq")

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="UnifiedReactAgent initialized",
            agent_id="UnifiedReactAgent",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "image_input": "true",
                "exploration": "basic"
            }
        )
    
    def Reset(self, request, context):
        self.thoughts = []
        self.actions = []
        self.observation = []
        self.history = []
        self.scratchpad = ""
        self.user_messages = []
        self.assistant_messages = []
        self.trajectory = {
            "observations": [],
        }

        return agent_pb2.AgentResetResponse(
            success=True,
            message="UnifiedReactAgent reset"
        )
    
    def format_messages(self, user_messages, assistant_messages, current_message):
        if self.hint:
            system_message = SYSTEM_PROMPT_WITH_HINT
        else:
            system_message = SYSTEM_PROMPT
        
        messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        if self.max_history_length == -1:
            for user_message, assistant_message in zip(user_messages, assistant_messages):
                messages.append(user_message)
                messages.append(assistant_message)
        else:
            if self.use_scratchpad:
                messages.append({
                    "role": "user",
                    "content": f"""
                    You have a scratchpad that you can use to store information about your interaction with the environment. The scratchpad contains the following:
                    {self.scratchpad}
                    """
                })
            for user_message, assistant_message in zip(user_messages[-self.max_history_length:], assistant_messages[-self.max_history_length:]):
                messages.append(user_message)
                messages.append(assistant_message)
        
        messages.append(current_message)
        return messages

    def generate_user_message(self, observation, available_actions):
        imgs = None
        contains_img, contains_options, contains_goal_state = False, False, False
        content = []
        instructions = f"The following actions are available at this step: {",\n".join([a.text_data for a in available_actions])}"
        if self.instruction_type == "react":
            instructions += f"\n{ACTION_PROMPT_REACT}"
        elif self.instruction_type == "reflection":
            instructions += f"\n{ACTION_PROMPT_REFLEXION}"
        
        if self.use_scratchpad:
            instructions += f"\n{RESPONSE_PROMPT_SCRATCHPAD}"
        else:
            instructions += f"\n{RESPONSE_PROMPT_DEFAULT}"
        
        if observation.image_data != b'':
            try:
                # Try to decode as JSON first (for MCQ with options)
                image_json_str = observation.image_data.decode('utf-8')
                image_data = json.loads(image_json_str)
                if "options" in image_data and "grid" in image_data:
                    # This is MCQ data with options
                    imgs = image_data["options"] + [image_data["grid"]]
                    contains_options = True
                    contains_img = True
                elif "grid" in image_data and "goal_state" in image_data:
                    # This is MCQ data with options
                    imgs = [image_data["grid"], image_data["goal_state"]]
                    contains_goal_state = True
                    contains_img = True
                else:
                    # Regular single image
                    img = base64.b64encode(observation.image_data).decode('utf-8')
                    imgs = [img]
                    contains_img = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to treating as regular image data
                img = base64.b64encode(observation.image_data).decode('utf-8')
                imgs = [img]
                contains_img = True

        if contains_img:
            if contains_options:
                content.extend([
                    {
                        "type": "text",
                        "text": f"""The current grid is as follows:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_data['grid']}"  # Grid is last
                    }
                ])
                
                # Add options
                for i in range(len(image_data["options"])):
                    content.extend([
                        {
                            "type": "text",
                            "text": f"""Option {i} is as follows:"""
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_data['options'][i]}"
                        }
                    ])
            elif contains_goal_state:
                content.extend([
                    {
                        "type": "text",
                        "text": f"""The current grid is as follows:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_data['grid']}"
                    },
                    {
                        "type": "text",
                        "text": f"""The goal state is as follows:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_data['goal_state']}"
                    }
                ])
            else:
                content = []
                content.extend([
                    {
                        "type": "text",
                        "text": f"""The current grid is as follows:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{imgs[0]}"
                    }
                ])
            content.append({
                "type": "text",
                "text": f"""{instructions}"""
            })
        else:
            content = f"""The current grid is as follows:\n{observation.text_data}\n\n{instructions}"""
        

        return {
            "role": "user",
            "content": content
        }


    def Act(self, request, context) -> agent_pb2.ActResponse:
        observation = request.observation
        available_actions = request.reactive_action_space.available_actions

        user_message = self.generate_user_message(observation, available_actions)
        messages = self.format_messages(self.user_messages, self.assistant_messages, user_message)
        self.history.append(messages)
        response = self.llm.generate_from_history(messages)
        
        self.user_messages.append(user_message)
        self.assistant_messages.append({
            "role": "assistant",
            "content": response
        })

        response_json = extract_tagged_response(response)
        if "action" in response_json:
            action = response_json["action"]
        else:
            action = "NOP"
        self.actions.append(action)
        if "scratchpad_add" in response_json:
            scratchpad_add = response_json["scratchpad_add"]
        else:
            scratchpad_add = None
        if "scratchpad_del" in response_json:
            scratchpad_del = response_json["scratchpad_del"]
        else:
            scratchpad_del = None
        if self.use_scratchpad:
            self.scratchpad = self.update_scratchpad(scratchpad_add, scratchpad_del)
        if "thought" in response_json or "think" in response_json:
            thought = response_json["thought"] if "thought" in response_json else response_json["think"]
            self.thoughts.append(thought)
        
        return agent_pb2.ActResponse(
            action=env_pb2.Action(text_data=action),
            confidence=0.8,
            metadata={"strategy": "exploration"}
        )

    def update_scratchpad(self, scratchpad_add, scratchpad_del):
        if scratchpad_add:
            self.scratchpad += scratchpad_add
        if scratchpad_del:
            self.scratchpad = self.scratchpad.replace(scratchpad_del, "")
        return self.scratchpad
    
    
    def EndEpisode(self, request, context) -> agent_pb2.EndEpisodeResponse:
        # TODO: Consider the end of episode and update the agent observation
        with open(os.path.join(self.logging_path, self.env_name, self.task_name, "history.json"), "w") as f:
            json.dump(self.history, f)
        with open(os.path.join(self.logging_path, self.env_name, self.task_name, "user_messages.json"), "w") as f:
            json.dump(self.user_messages, f)
        with open(os.path.join(self.logging_path, self.env_name, self.task_name, "assistant_messages.json"), "w") as f:
            json.dump(self.assistant_messages, f)
        if len(self.actions) > 0:
            with open(os.path.join(self.logging_path, self.env_name, self.task_name, "actions.json"), "w") as f:
                json.dump(self.actions, f)
        if len(self.thoughts) > 0:
            with open(os.path.join(self.logging_path, self.env_name, self.task_name, "thoughts.json"), "w") as f:
                json.dump(self.thoughts, f)
        if self.use_scratchpad:
            with open(os.path.join(self.logging_path, self.env_name, self.task_name, "scratchpad_final.txt"), "w") as f:
                f.write(self.scratchpad)
        return agent_pb2.EndEpisodeResponse(
            acknowledged=True
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_grpc.add_MARAAgentServicer_to_server(
        ReactLLMAgentServicer(), server
    )

    port = 50254
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"MARALLMInteractiveAgent server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LLM Agent server.")
    parser.add_argument("--logfile", type=str, default="llm_agent.log", help="Path to the log file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    serve()