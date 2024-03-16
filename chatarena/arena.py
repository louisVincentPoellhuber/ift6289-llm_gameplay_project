import csv
import json
import logging
import uuid
from typing import Dict, List, Union

from .agent import Player
from .backends import Human
from .config import ArenaConfig
from .environments import Environment, TimeStep, load_environment


class TooManyInvalidActions(Exception):
    pass


class Arena:
    """Utility class that manages the game environment and players."""

    def __init__(
        self, players: List[Player], environment: Environment, global_prompt: str = None
    ):
        # Create a container for the players and environment and reset the game
        self.players = players
        self.environment = environment
        self.global_prompt = global_prompt

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.invalid_actions_retry = 5

    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.players}

    def reset(self) -> TimeStep:
        # Reset the environment
        self.current_timestep = self.environment.reset()
        # Reset the players
        for player in self.players:
            player.reset()
        # Reset the uuid
        self.uuid = uuid.uuid4()
        return self.current_timestep

    def step(self) -> TimeStep:
        """Take a step in the game: one player takes an action and the environment updates."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(
            player_name
        )  # get the observation for the player

        timestep = None
        for i in range(self.invalid_actions_retry):  # try to take an action for a few times
            action = player(observation)  # take an action
            if self.environment.check_action(action, player_name):  # action is valid
                timestep = self.environment.step(
                    player_name, action
                )  # update the environment
                break
            else:  # action is invalid
                logging.warning(f"{player_name} made an invalid action {action}")
                continue

        if (
            timestep is None
        ):  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            logging.warning(warning_msg)
            raise TooManyInvalidActions(warning_msg)

        return timestep

    def next_is_human(self):
        """Check if the next player is human."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]
        return isinstance(player.backend, Human)

    def run(self, num_steps: int = 1):
        """Run the game for num_turns."""
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                break

    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig]):
        """Create an arena from a config."""
        # If config is a path, load the config
        if isinstance(config, str):
            config = ArenaConfig.load(config)

        global_prompt = config.get("global_prompt", None)

        # Create the players
        players = []
        for player_config in config.players:
            # Add public_prompt to the player config
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt

            player = Player.from_config(player_config)
            players.append(player)

        # Check that the player names are unique
        player_names = [player.name for player in players]
        assert len(player_names) == len(
            set(player_names)
        ), "Player names must be unique"

        # Create the environment
        config.environment[
            "player_names"
        ] = player_names  # add the player names to the environment config
        env = load_environment(config.environment)

        return cls(players, env, global_prompt=global_prompt)

    def to_config(self) -> ArenaConfig:
        """Convert the arena to a config."""
        # return {
        #     "players": [player.to_config() for player in self.players],
        #     "environment": self.environment.to_config(),
        #     "global_prompt": self.global_prompt
        # }
        return ArenaConfig(
            players=[player.to_config() for player in self.players],
            environment=self.environment.to_config(),
            global_prompt=self.global_prompt,
        )

    def launch_cli(self, max_steps: int = None, interactive: bool = True):
        """Launch the command line interface."""
        from chatarena.ui.cli import ArenaCLI

        cli = ArenaCLI(self)
        cli.launch(max_steps=max_steps, interactive=interactive)

    def save_config(self, path: str):
        """Save the config to a file."""
        config = self.to_config()
        config.save(path)

    def save_history(self, path: str):
        """
        Save the history of the game to a file.

        Supports csv and json formats.
        """
        messages = self.environment.get_observation()
        message_rows = []

        if path.endswith(".csv"):
            header = [
                "agent_name",
                "content",
                "turn",
                "timestamp",
                "visible_to",
                "msg_type",
            ]
            for message in messages:
                message_row = [
                    message.agent_name,
                    message.content,
                    message.turn,
                    str(message.timestamp),
                    message.visible_to,
                    message.msg_type,
                ]
                message_rows.append(message_row)

            with open(path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(message_rows)
        elif path.endswith(".json"):
            for message in messages:
                message_row = {
                    "agent_name": message.agent_name,
                    "content": message.content,
                    "turn": message.turn,
                    "timestamp": str(message.timestamp),
                    "visible_to": message.visible_to,
                    "msg_type": message.msg_type,
                }
                message_rows.append(message_row)

            with open(path, "w") as f:
                json.dump(message_rows, f, indent=4)
        else:
            raise ValueError("Invalid file format")
        
    def save_chat(self, path: str):
        if not path.endswith(".json"):
            raise ValueError("Invalid file format. Please save the chat as a JSON file.")
        chat_dict = {}  

        disposition = self.environment.get_disposition()
        chat_dict["disposition"] = disposition
        
        player_list = []

        for player in self.players:
            player_dict = {}
            player_dict["name"] = player.name
            player_dict["backend"] = player.backend.type_name

            player_dict["role"] = [role for role in disposition["roles"] if player.name in disposition["roles"][role]][0]

            #for role in disposition["roles"]:
            #    if disposition["roles"][role] == player.name:
            #        player_dict["role"] = role

            player_dict["role_desc"] = player.role_desc

            player_list.append(player_dict)

        chat_dict["players"] = player_list

        metrics = self.environment.get_metrics()

        chat_dict["metrics"] = metrics

        # Messages
        messages = self.environment.get_observation()
        message_rows = []

        for message in messages:
            message_row = {
                "agent_name": message.agent_name,
                "content": message.content,
                "turn": message.turn,
                "timestamp": str(message.timestamp),
                "visible_to": message.visible_to,
                "msg_type": message.msg_type,
            }
            message_rows.append(message_row)
        
        chat_dict["messages"] = message_rows

        with open(path, "w") as f:
            json.dump(chat_dict, f, indent=4)
