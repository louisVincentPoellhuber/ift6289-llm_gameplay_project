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
PROMPT_CONFIG_FILE = dir_path + "/spyfall_final_experiments.yaml"

# Opening configurations
with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)

# Name of the players
number_of_players = 6
# Starting cohere backend
client = cohere.Client(os.getenv("COHEREAI_API_KEY"))

# prompt_mode = "remember_json"
# prompt_mode = "test"
prompt_mode = "chain_of_thoughts"
repetitions = 1

backend = CohereAIChat(
    temperature=prompts[prompt_mode]["temperature"],
    max_tokens=prompts[prompt_mode]["max_tokens"],
    model=prompts[prompt_mode]["model_name"],
    preamble=prompts[prompt_mode]["preamble"],
)

print("\n\n", "-----------------------------------")
print(f"PROMPT MODE:", prompt_mode, "\n\n")

# Defining players
players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"][:number_of_players]
random.shuffle(players)
# First description of the game -> taken from GameEval
role_description = prompts["role_description"].format(number_of_players=len(players))
# Chatarena output format
format_specification = prompts["json_format_specification"]
if prompts[prompt_mode]["response_format"] == "string":
    format_specification = ""

players_list = [
    Player(
        name=players[i],
        role_desc="Your name is "
        + players[i]
        + ","
        + role_description
        + format_specification,
        backend=backend,
    )
    for i in range(len(players[:number_of_players]))
]

# Running experiment
env = SpyFall(
    player_names=players,
    prompt_config_file=PROMPT_CONFIG_FILE,
    prompt_config_mode=prompt_mode,
    restrict_info=prompts[prompt_mode]["restrict_info"],
)
arena = Arena(players_list, env)
arena.launch_cli(interactive=False, max_steps=40)

postfix = f"{prompt_mode}_{strftime('%Y_%m_%d_%H_%M_%S')}"
# Saving history
arena.save_chat(
    f"src/spyfall/chat_history/test/spyfall_{prompt_mode}_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
)
