import random
import string
from unidecode import unidecode
import re
from typing import Dict, List, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from ..utils import extract_jsons



DEFAULT_WORD_LIST = ["Ornithorhynchus"]
speaker_format_specification = """
Your output should be formatted in a json with the following schema:
```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "clue": {
      "description": "a clue hinting at the secret word that you are going to give the guesser",
      "type": "string"
    }
  },
  "required": ["clue"]
}
```

For example:
```
{\n  "clue": "The secret word is a hairy animal." \n}

```
"""
speaker_role_description = """
You are the speaker! You will be given a secret word. You you must make your game partner 
guess your secret word correctly, without saying the word. You will need to give them precise and 
concise clues hinting at your secret word, excluding the secret word. 

You need to make your partner guess your secret word in as few rounds as possible.

You should never:
1. Say the secret word
2. Repeat a clue

"""

guesser_role_description = """
You are the guesser! You are going to guess a secret word using clues given to you by your game partner. 
You need to use these clues to guess the secret word. You also need to provide 
arguments for each of your guesses, to help your partner create better clues. 

You need to make your partner guess your secret word in as few rounds as possible.

You should never:
1. Repeat a guess
"""


guesser_format_specification = """
Your output should be formatted in a json with the following schema:
```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "guess": {
      "description": "a single word, your guess given the clues",
      "type": "string"
    }, 
    "arguments": {
      "description": "the reasoning behind your guess, why you chose this word",
      "type": "string"
    }
  },
  "required": ["guess", "arguments"]
}
```

For example:
```
{\n  "guess": "cat", "arguments": "Since the clue is that the secret word is a hairy animal, my guess is 'cat'." \n}

```
"""


@register_env
class AskGuess(Environment):
    type_name = "ask-guess"

    def __init__(
        self,
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

        self._moderator_speak(f"Now the game starts!")
        self._moderator_speak(speaker_role_description + speaker_format_specification +
            f"The secret word is: {self.word}",
            visible_to=self.speaker,
        )
        self._moderator_speak(guesser_role_description+guesser_format_specification, visible_to=self.guesser)
        self._moderator_speak(
            "Now the speaker now gives one clue (but don't give away the secret word). "
            f"You cannot repeat a clue you've already given."
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

    # TODO: study the impact of point-based rewards or highscore-based rewards
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

            json_list = extract_jsons(action)
            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                return timestep # stop early to avoid json error
            elif len(json_list) != 1:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} is not a valid json.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                return timestep # stop early to avoid json error




            clue = json_list[0].get("clue", None)
            if self.word in clue: 
                self._ending_condition = "AME"
                self._is_terminal = True
                print(f"The answer was mentioned in the clue.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
            
            message = Message(
                agent_name=player_name, content=clue, turn=self._current_turn
            )
            self.message_pool.append_message(message)

            # Update the counters
            self._current_turn += 1
               
            self._current_phase = "guess"

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages
        elif self._current_phase == "guess":

            json_list = extract_jsons(action)
            if len(json_list) != 1:
                self._ending_condition = "EE"
                self._is_terminal = True
                print(f"Player output {action} is not a valid json.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                return timestep

            if "END_OF_CONVERSATION" in action:
                self._ending_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=self._is_terminal)
                return timestep
            
            guess = json_list[0].get("guess", None)
            arguments = json_list[0].get("arguments", None)

            message = Message(
                agent_name=player_name,
                content=arguments,
                turn=self._current_turn,
            )
            self.message_pool.append_message(message)

            if self._is_true_word(guess):
                self._moderator_speak(
                    f"{player_name} guessed the word correctly! The secret word is {self.word}. "
                    f"You both won!"
                )
                rewards = self.get_rewards(correct_guess=True)
                self._correct_guess = True
                self._is_terminal = True
            else:
                self._moderator_speak(
                    f"{player_name} guessed the word wrong. Now the speaker will give another clue! Don't forget to answer in the correct JSON format. "
                )
                rewards = self.get_rewards(correct_guess=False)

            self._current_phase = "give clues"
            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=self._is_terminal)
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")
    
        if self.is_terminal():
            timestep.terminal = True

        return timestep
