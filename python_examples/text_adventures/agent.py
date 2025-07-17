#!/usr/bin/env python3
import sys
import grpc
from concurrent import futures
import time
import logging

# Import generated protocol buffer code
from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_agent_pb2 as agent_pb2
from generated.mara import mara_agent_pb2_grpc as agent_grpc
import random


class TextAdventureAgent:

    def __init__(self):
        self.inventory = []
        self.visited_rooms = set()
        self.exploration_priority = ["look", "take", "go"]

    def reset(self):
        self.inventory = []
        self.visited_rooms = set()
        return True

    def act(self, observation, available_actions):
        obs_text = observation.lower()

        # Randomly choose an action
        if available_actions:
            return random.choice(available_actions)

        # # Process observation to understand the environment
        # if "you are in" in obs_text:
        #     # Extract room description and add to visited rooms
        #     room_desc = obs_text.split('.')[0]
        #     self.visited_rooms.add(room_desc)
        #
        # # Extract items from observation
        # items_mentioned = []
        # if "you see:" in obs_text:
        #     items_part = obs_text.split("you see:")[-1].split(".")[0]
        #     items_mentioned = [item.strip() for item in items_part.split(",")]
        #
        # # Choose an action based on priority
        #
        # # First priority: Take all items
        # for action in available_actions:
        #     if action.startswith("take "):
        #         item = action.split("take ")[1]
        #         if item not in self.inventory:
        #             return action
        #
        # # Second priority: Explore unvisited directions
        # for action in available_actions:
        #     if action.startswith("go "):
        #         direction = action.split("go ")[1]
        #         # Heuristic: If we haven't gone this way before, try it
        #         potential_state = f"{obs_text} after going {direction}"
        #         if potential_state not in self.visited_rooms:
        #             return action
        #
        # # Third priority: Look around
        # if "look" in available_actions:
        #     return "look"
        #
        # # Fourth priority: Just pick the first available action
        # if available_actions:
        #     return available_actions[0]

        return "look"  # Fallback action


class MARAAgentServicer(agent_grpc.MARAAgentServicer):

    def __init__(self):
        self.agent = None

    def Initialize(self, request, context):
        self.agent = TextAdventureAgent()

        return agent_pb2.AgentInitializeResponse(
            success=True,
            message="Text Adventure Agent initialized",
            agent_id="text_adventure_agent_v1",
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

        action_text = self.agent.act(observation, available_actions)

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
            agent_id="text_adventure_agent_v1",
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
                "domain": "text_adventure",
                "description":
                "A simple rule-based agent for text adventure games"
            })

    def Close(self, request, context):
        self.agent = None
        return agent_pb2.AgentCloseResponse(
            success=True, message="Agent closed successfully")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_grpc.add_MARAAgentServicer_to_server(MARAAgentServicer(), server)

    port = 50052
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Text Adventure Agent server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
