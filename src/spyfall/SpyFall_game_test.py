import os
import sys
import yaml
import cohere
from time import strftime
from dotenv import load_dotenv
import random

# sys.stdout = open("output_spyfall.txt", "w")

# Configuring path from the library
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

# Loading dotenv
load_dotenv()

from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.spyfall import SpyFall

# Prompts yaml
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
PROMPT_CONFIG_FILE = dir_path + "/spyfall_prompt_config.yaml"

# Starting cohere backend
client = cohere.Client(os.getenv("COHEREAI_API_KEY"))
backend = CohereAIChat()

# Name of the players
number_of_players = 6
players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"][:number_of_players]


with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)

# First description of the game -> taken from GameEval
role_description = prompts["role_description"].format(number_of_players=len(players))
# Chatarena format
format_specification = prompts["json_format_specification"]

random.shuffle(players)
players_list = [
    Player(
        name=players[i],
        role_desc=role_description + format_specification,
        backend=backend,
    )
    for i in range(len(players[:number_of_players]))
]

prompt_mode = "remember_json_not_to_interact"
prompt_modes = [
    "baseline",
    "remember_json",
    "remember_json_and_not_to_say_word",
    "remember_json_not_to_interact",
]
for prompt_mode in prompt_modes:
    # Running experiment
    env = SpyFall(
        player_names=players,
        prompt_config_file=PROMPT_CONFIG_FILE,
        prompt_config_mode=prompt_mode,
    )
    arena = Arena(players_list, env)
    arena.launch_cli(interactive=False, max_steps=40)

    # Saving history
    arena.save_chat(
        f"src/spyfall/chat_history/spyfall_{strftime('%Y_%m_%d_%H_%M_%S')}_{prompt_mode}.json"
    )
