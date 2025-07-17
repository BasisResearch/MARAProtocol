#!/usr/bin/env python3
from concurrent import futures
import logging
import time

import grpc

logging.basicConfig(level=logging.INFO)

# Import generated protocol buffer code
from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_environment_service_pb2 as env_service_pb2
from generated.mara import mara_environment_service_pb2_grpc as env_grpc


class TextAdventureEnvironment:

    def __init__(self):
        self.rooms = {
            "start": {
                "description":
                "You are in a dimly lit room. There are doors to the north and east.",
                "exits": {
                    "north": "hallway",
                    "east": "library"
                },
                "items": ["lamp"]
            },
            "hallway": {
                "description":
                "A long hallway stretches before you. There's a door to the south and a door to the west.",
                "exits": {
                    "south": "start",
                    "west": "kitchen"
                },
                "items": []
            },
            "library": {
                "description":
                "Bookshelves line the walls. There's a door to the west and a desk with a key.",
                "exits": {
                    "west": "start"
                },
                "items": ["book", "key"]
            },
            "kitchen": {
                "description":
                "A small kitchen with a table in the center. There's a door to the east.",
                "exits": {
                    "east": "hallway"
                },
                "items": ["apple"]
            }
        }

        self.current_room = "start"
        self.inventory = []
        self.terminal = False

    def reset(self):
        self.current_room = "start"
        self.inventory = []
        self.terminal = False
        return self._get_observation()

    def step(self, action_text):
        action_parts = action_text.lower().split()

        if len(action_parts) < 1:
            return self._get_observation(), 0, False, {
                "message": "I don't understand that command."
            }

        command = action_parts[0]

        if command == "go" and len(action_parts) > 1:
            direction = action_parts[1]
            return self._handle_movement(direction)
        elif command == "take" and len(action_parts) > 1:
            item = action_parts[1]
            return self._handle_take(item)
        elif command == "inventory":
            return self._handle_inventory()
        elif command == "look":
            return self._get_observation(), 0, False, {
                "message": "You take a careful look around."
            }
        elif command == "quit":
            self.terminal = True
            return self._get_observation(), 0, True, {"message": "Game over."}
        else:
            return self._get_observation(), 0, False, {
                "message": f"I don't understand '{action_text}'."
            }

    def _handle_movement(self, direction):
        if direction in self.rooms[self.current_room]["exits"]:
            self.current_room = self.rooms[
                self.current_room]["exits"][direction]
            return self._get_observation(), 1, False, {
                "message": f"You move {direction}."
            }
        else:
            return self._get_observation(), 0, False, {
                "message": f"You can't go {direction} from here."
            }

    def _handle_take(self, item):
        if item in self.rooms[self.current_room]["items"]:
            self.rooms[self.current_room]["items"].remove(item)
            self.inventory.append(item)
            return self._get_observation(), 1, False, {
                "message": f"You take the {item}."
            }
        else:
            return self._get_observation(), 0, False, {
                "message": f"There's no {item} here."
            }

    def _handle_inventory(self):
        if not self.inventory:
            inventory_text = "Your inventory is empty."
        else:
            inventory_text = "You are carrying: " + ", ".join(self.inventory)
        return self._get_observation(), 0, False, {"message": inventory_text}

    def _get_observation(self):
        room = self.rooms[self.current_room]
        obs_text = room["description"]

        if room["items"]:
            obs_text += " You see: " + ", ".join(room["items"]) + "."

        return obs_text

    def get_action_space(self):
        available_actions = ["look", "inventory", "quit"]

        # Add movement actions
        for direction in self.rooms[self.current_room]["exits"]:
            available_actions.append(f"go {direction}")

        # Add take actions
        for item in self.rooms[self.current_room]["items"]:
            available_actions.append(f"take {item}")

        return available_actions


class MARATextAdventureServicer(env_grpc.MARAEnvironmentServicer):

    def __init__(self):
        self.environment = None

    def Initialize(self, request, context):
        self.environment = TextAdventureEnvironment()

        response = env_service_pb2.InitializeResponse(
            success=True,
            message="Text Adventure Environment initialized successfully")

        # Set environment-specific fields
        reactive_env = env_pb2.ReactiveEnvironment(
            environment_id="text_adventure_v1",
            version="1.0.0",
            metadata={
                "author": "MARA Developer",
                "domain": "text_adventure"
            })

        response.reactive_env.CopyFrom(reactive_env)
        return response

    def Reset(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        initial_obs = self.environment.reset()

        response = env_service_pb2.ResetResponse(
            initial_observation=env_pb2.Observation(text_data=initial_obs),
            info={})
        return response

    def GetEnvironmentInfo(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        response = env_service_pb2.EnvironmentInfoResponse(
            environment_id="text_adventure_v1",
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
        return response

    def Step(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        action_text = request.action.text_data
        observation, reward, is_terminal, info = self.environment.step(
            action_text)

        info_dict = {}
        if "message" in info:
            info_dict["message"] = info["message"]

        response = env_service_pb2.StepResponse(
            observation=env_pb2.Observation(text_data=observation),
            reward=reward,
            is_terminal=is_terminal,
            info=info_dict)
        return response

    def QuerySpaces(self, request, context):
        if not self.environment:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Environment not initialized")

        available_actions = self.environment.get_action_space()

        action_space = env_pb2.ReactiveEnvironment.ActionSpace(
            available_actions=available_actions,
            constraints={
                "type": "discrete",
                "text_commands": "true"
            },
            is_continuous=False)

        response = env_service_pb2.SpaceQueryResponse()
        response.reactive_response.action_space.CopyFrom(action_space)
        return response

    def Close(self, request, context):
        self.environment = None
        return env_service_pb2.CloseResponse(
            success=True, message="Environment closed successfully")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    env_grpc.add_MARAEnvironmentServicer_to_server(MARATextAdventureServicer(),
                                                   server)

    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Text Adventure Environment server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
