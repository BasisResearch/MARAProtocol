#!/usr/bin/env python3
import grpc
from concurrent import futures
import time
import logging

# Import generated protocol buffer code
from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_agent_pb2 as agent_pb2
from generated.mara import mara_agent_pb2_grpc as agent_grpc
import random


# TODO: Implement a more intelligent agent
class RandomInteractiveAgent:

    def __init__(self):
        pass

    def reset(self):
        return True

    def act(self, observation, available_actions) -> env_pb2.Action:
        _ = observation.lower()
        random_action = random.choice(available_actions)
        logging.info(f"available_actions: {available_actions}")
        action_str = random_action.text_data
        if 'click' in action_str:
            # pick where to click
            x_range, y_range = action_str.split(' ')[1:]
            
            # Parse the ranges from [min-max] format
            x_min, x_max = map(int, x_range.strip('[]').split('-'))
            y_min, y_max = map(int, y_range.strip('[]').split('-'))
            
            # Sample random x, y position within the ranges
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            random_action = env_pb2.Action(
                text_data=f"click {x} {y}")
        return random_action


class MARARandomAgentServicer(agent_grpc.MARAAgentServicer):

    def __init__(self):
        self.agent = None

    def Initialize(self, request, context):
        self.agent = RandomInteractiveAgent()

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="Autumn Random Interactive Agent initialized",
            agent_id="autumn_random_interactive_agent_v1",
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            })

    def Reset(self, request, context):
        if not self.agent:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Agent not initialized")

        success = self.agent.reset()

        return agent_pb2.AgentResetResponse(success=success,
                                            message="Agent reset successful")

    def Act(self, request, context):
        if not self.agent:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Agent not initialized")

        observation = request.observation.text_data
        available_actions = []

        if request.HasField("reactive_action_space"):
            available_actions = request.reactive_action_space.available_actions

        action_text = self.agent.act(observation, available_actions).text_data 

        return agent_pb2.ActResponse(
            action=env_pb2.Action(text_data=action_text),
            confidence=0.8,
            metadata={"strategy": "exploration"})

    def Feedback(self, request, context):
        # In this simple agent, we don't use feedback for learning
        # But we could update internal state based on rewards
        return agent_pb2.FeedbackResponse(acknowledged=True)

    def EndEpisode(self, request, context):
        # Episode summary could be used to improve agent for next episode
        return agent_pb2.EndEpisodeResponse(acknowledged=True)

    def GetAgentInfo(self, request, context):
        return agent_pb2.AgentInfoResponse(
            agent_id="autumn_random_interactive_agent_v1",
            version="1.0.0",
            agent_type=agent_pb2.POLICY,
            compatible_environment_types=["REACTIVE"],
            capabilities={
                "text_input": "true",
                "text_output": "true",
                "exploration": "basic"
            },
            metadata={
                "author":
                    "MARA Developer",
                "domain":
                    "Autumn",
                "description":
                    "A simple interactive agent that randomly selects actions"
            })

    def Close(self, request, context):
        self.agent = None
        return agent_pb2.AgentCloseResponse(success=True,
                                            message="Agent closed successfully")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_grpc.add_MARAAgentServicer_to_server(MARAAgentServicer(), server)

    port = 50253
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Autumn Random Agent server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
