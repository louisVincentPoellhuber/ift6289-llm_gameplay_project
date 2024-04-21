# NEW
# This is a brand-new Environment that our team has implemented. It holds all the 
# game logic for the game Ask-Guess.
# 
# A typical game will start with the Arena running the __init__ function. Then, it
# will run the step function until an ending state is reached. 
#
# Ask-guess iterates through two phases between steps: the "give clues" phase and 
# the "guess" phase. In the step function, you can find a large IF statement that 
# switches the step logic depending on each phase. Other functions simply support
# the main game logic. 

import random
import string
from unidecode import unidecode
import re
from typing import Dict, List, Union
import yaml

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from ..utils import extract_jsons


# Super hard word
DEFAULT_WORD_LIST = ["Ornithorhynchus"]

@register_env
class AskGuess(Environment):
    type_name = "ask-guess"

    def __init__(
        self,
        prompt_config_mode: str,
        prompt_config_file: str,
        player_names: List[str],
        word_list: List[str] = None,
        **kwargs,
    ):
        super().__init__(player_names=player_names, word_list=word_list, **kwargs)

        if word_list is None:
            word_list = DEFAULT_WORD_LIST
        self.word_list = word_list
        self.word_guess = None

        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()

        # Randomly sample a random word and roles
        self.word = None
        self.guesser = None
        self.speaker = None

        # Game states
        self._current_turn = 0
        #self._next_player_idx = 0
        self._current_phase = "give clues"  # "give clues", "guess"
        self._initialized = False
        self._correct_guess = False
        self._ending_condition = None
        self._is_terminal = False

        self._prompt_config_mode = prompt_config_mode
        self._prompt_config_prompt_config_file = prompt_config_file

        # Reading prompt configs
        with open(self._prompt_config_prompt_config_file, "r") as file:
            self._prompts = yaml.safe_load(file)


        self.reset()  # To initialize the game (select a random word and roles)

    def get_next_player(self) -> str:
        """Get the next player."""
        if self._current_phase == "guess":
            return self.guesser
        else:
            return self.speaker

    def reset(self):
        """Sample a random word and roles."""
        self.word = random.choice(self.word_list)

        self.guesser = random.choice(self.player_names)
        self.speaker = [name for name in self.player_names if name != self.guesser][0]

        self._current_turn = 0
        #self._next_player_idx = 0
        self._current_phase = "give clues"

        self.message_pool.reset()

        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["speaker_role"],
            visible_to=self.speaker,
        )
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["guesser_role"],
            visible_to=self.guesser,
        )
        self._moderator_speak(f"Now the game starts!")
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["secret_word_message"]
            .format(word=self.word)
            .replace(r"{{", "{")
            .replace(r"}}", "}"),
            visible_to=self.speaker,
        )
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["initial_clues_phase"]
        )
        self._current_turn = 1

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """Get observation for the player."""
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self._current_turn
            )

    def _text2guess(self, text) -> str:
        """Convert text to word guess, return the word guess."""
        text = text.lower()
        pattern = r'<([^>]*)>'
        match = re.search(pattern, text)

        return match

    def _normalize_text(self, s):
        """Lower text and remove punctuation, and extra whitespace."""

        def white_space_fix(text, space_char=" "):
            return space_char.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            exclude.add('"')
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def remove_accents(text):
            return unidecode(text)

        def apply_funcs(text):
            text = lower(text)
            text = remove_punc(text)
            text = white_space_fix(text, space_char="_")
            text = remove_accents(text)
            return text

        return apply_funcs(s)


    def _is_true_word(self, guess) -> bool:
        """Check whether the guess is the true word."""
        # Get the word enclosed by quote marks with regex
        guess = self._normalize_text(guess)
        word = self._normalize_text(self.word)
        
        return guess == word

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """Moderator say something."""
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def get_rewards(self, correct_guess: bool) -> Dict[str, float]:
        """Get rewards for each player."""
        rewards = {}
        for name in self.player_names:
            # They both get a point if the guesser guessed correctly, or they both get none if not. 
            rewards[name] = float(correct_guess)

        return rewards

    def is_terminal(self) -> bool:
        """Check if the conversation is over."""
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content.startswith(
            SIGNAL_END_OF_CONVERSATION
        ):
            return True
    
    def get_disposition(self) -> Dict:
        disposition = {}
        disposition["secret_word"] = self.word
        disposition["nb_players"] = len(self.player_names)
        disposition["roles"] = {"speaker":[self.speaker], "guesser":[self.guesser]} # has to be lists!

        return disposition

    def get_metrics(self) -> Dict:
        metrics = {}

        metrics["nb_turns"] = self._current_turn

        if self._correct_guess:
            metrics["end_condition"] = "ST" # Successful trial
        elif self._ending_condition != None:
            metrics["end_condition"] = self._ending_condition # Ending error, Chat error or Answer Mentioned Error
        else:
            metrics["end_condition"] = "RLE" # Round Limit Error (reaches the number of max steps)

        return metrics

    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Step function that is called by the arena.

        Args:
            player_name: the name of the player that takes the action
            action: the action that the agents wants to take
        """
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."
        if self._current_phase == "give clues":

            # Switch formats
            if (self._prompt_config_mode=="bracket_format") | (self._prompt_config_mode=="best"):
                clue, timestep = self.get_bracket_response(action)
            elif self._prompt_config_mode=="sentence_format":
                clue, timestep = self.get_sentence_response(action)
            else:
                clue, timestep = self.get_json_response(action)

            if timestep != None:
                final_output = f"[Final Message] {action}"

                self._moderator_speak(final_output)
                return timestep
            
            if self.word in clue: 
                self._ending_condition = "AME"
                self._is_terminal = True
                print(f"The answer was mentioned in the clue.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
            
            message = Message(
                agent_name=player_name, content=clue, turn=self._current_turn
            )

            # Update the counters
            self._current_turn += 1
            self._current_phase = "guess"

            if (self._prompt_config_mode == "format_reminder"):
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["guesser_format"],
                    visible_to=self.guesser
                )
            
            self.message_pool.append_message(message)
            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages
        elif self._current_phase == "guess":

            # Switch formats
            if (self._prompt_config_mode=="bracket_format") | (self._prompt_config_mode=="best"):
                guess, arguments, timestep = self.get_bracket_response(action)
            elif self._prompt_config_mode=="sentence_format":
                guess, arguments, timestep = self.get_sentence_response(action)
            else:
                guess, arguments, timestep = self.get_json_response(action)

            if timestep != None:
                final_output = f"[Final Message] {action}"

                self._moderator_speak(final_output)
                return timestep

            message = Message(
                agent_name=player_name,
                content=arguments,
                turn=self._current_turn,
            )
            self.message_pool.append_message(message)

            if self._is_true_word(guess):
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["win_message"]
                    .format(player_name=self.guesser, word=self.word)
                    .replace(r"{{", "{")
                    .replace(r"}}", "}"),
                )
                rewards = self.get_rewards(correct_guess=True)
                self._correct_guess = True
                self._is_terminal = True
            else:
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["wrong_guess_message"]
                    .format(player_name=self.guesser)
                    .replace(r"{{", "{")
                    .replace(r"}}", "}"),
                )
                rewards = self.get_rewards(correct_guess=False)

            self._current_phase = "give clues"
            
            if (self._prompt_config_mode == "format_reminder") & ~(self._is_terminal):
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["speaker_format"],
                    visible_to=self.speaker
                )
            elif (self._prompt_config_mode == "word_reminder") & ~(self._is_terminal):
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["secret_word_reminder"]
                    .format(word=self.word)
                    .replace(r"{{", "{")
                    .replace(r"}}", "}"),
                    visible_to=self.speaker
                )

            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=self._is_terminal)
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")
    
        if self.is_terminal():
            timestep.terminal = True

        return timestep
    
    def get_json_response(self, action):
        if self._current_phase == "give clues":

            json_list = extract_jsons(action)
            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                clue = None
            
            elif len(json_list) != 1:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} is not a valid json.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                clue = None
            else:
                timestep = None
                clue = json_list[0].get("clue", None)

            return clue, timestep
        
        elif self._current_phase == "guess":

            json_list = extract_jsons(action)
            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                guess = None
                arguments = None
                
            elif len(json_list) != 1:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} is not a valid json.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                guess = None
                arguments = None
            else:
                guess = json_list[0].get("guess", None)
                arguments = json_list[0].get("arguments", None)
                timestep = None

            return guess, arguments, timestep

    def get_bracket_response(self, action):
        
        if self._current_phase == "give clues":
            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                clue = None
            else:
                clue = action
                timestep = None
            
            return clue, timestep
            
        elif self._current_phase == "guess":
            pattern = r'\[(.*?)\]'
    
            match = re.search(pattern, action)

            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                clue = None
            elif match:
                guess = match.group(1)
                arguments = action
                timestep = None
            else:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} does not contain brackets.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                guess = None
                arguments = None

            
            return guess, arguments, timestep
        
    def get_sentence_response(self, action):
        MAX_NB_WORDS = 2 # How many words does the format allow. By default 2, as none of our secret words are longer than 2. 
        if self._current_phase == "give clues":
            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                clue = None
            else:
                clue = action
                timestep = None
            
            return clue, timestep
                
        elif self._current_phase == "guess":
            sentences = action.split(".")
            guess = sentences[0]
            arguments = action
            timestep = None

            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
            elif guess.count(" ")>MAX_NB_WORDS:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} is not in the correct format.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                
            return guess, arguments, timestep

