import time
import os
from typing import List, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import Message
from .base import IntelligenceBackend, register_backend

# Try to import the cohere package and check whether the API key is set
try:
    import cohere
except ImportError:
    is_cohere_available = False
else:
    if os.environ.get("COHEREAI_API_KEY") is None:
        is_cohere_available = False
    else:
        is_cohere_available = True

# Default config follows the [Cohere documentation](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#cohere.client.Client.chat)
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MODEL = "command-r+"
DEFAULT_SPY = """
You are one of the player of spyfall.

  There are multiple players in this game. 
  At the beginning of the game, everyone will receive a word.
  There is one spy who will receive a spy word, while others will receive a common word.
  Spy word is different but relevant to common words. For example, the spy word can be "apple", and the common word is "banana".

  There are two stages in each round of the game.

  The first stage is describing stage:
  In this stage, the only thing you need to do is use a word or a few words to say something in turn about the word he received without directly saying the word.
  The funniest part of the game is that since you do not know other's words, you are not sure whether you are the spy.
  So, you can only infer who have the different based on other players description.

  The second stage is the voting stage:
  In this stage, You MUST VOTE for someone, THE ONLY THING you do is analyse all player's text and vote for the player you think is the spy and tell the reason why you think he is the spy.
  You should always provide only your accusing name within * mark. Suppose your name is Amy, For example:
    "Amy: I don't aggree with Nancy. I think that *Jack* is the spy, because of..."
    Only include the name you believe is a spy in *.
    If you are accused, fight for your self and find suspicious descrption of other players.

    Analyze who is the different one, try find the spy!  And never reveal your secret word! Remeber your name, you are always this player!!
"""


@register_backend
class CohereAIChat(IntelligenceBackend):
    """Interface to the Cohere API."""

    stateful = True
    type_name = "cohere-chat"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        preamble: str = DEFAULT_SPY,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            preamble=preamble,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.preamble = preamble

        assert (
            is_cohere_available
        ), "Cohere package is not installed or the API key is not set"
        self.client = cohere.Client(os.environ.get("COHEREAI_API_KEY"))

        # Stateful variables
        self.session_id = None  # The session id for the last conversation
        self.last_msg_hash = (
            None  # The hash of the last message of the last conversation
        )

    def reset(self):
        self.session_id = None
        self.last_msg_hash = None

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(min=1, max=60))
    def _get_response(
        self, new_message: str, persona_prompt: Union[dict], verbose=False
    ):

        if verbose:
            print("chat_history:", persona_prompt)

        # sleepy spleepy
        time.sleep(3)
        response = self.client.chat(
            new_message,
            chat_history=persona_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            conversation_id=self.session_id,
            preamble=self.preamble,
        )

        self.session_id = response.conversation_id  # Update the session id
        if "END_OF_CONVERSATION" in response.text:  # Chat Error
            return "END_OF_CONVERSATION"
        return response.text

    def query(
        self,
        agent_name: str,
        role_desc: str,
        context: str,
        history_messages: List[Message],
        premable: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Format the input and call the Cohere API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the CohereAI
        """
        # Find the index of the last message of the last conversation

        new_message_start_idx = 0
        if self.last_msg_hash is not None:
            for i, message in enumerate(history_messages):
                if message.msg_hash == self.last_msg_hash:
                    new_message_start_idx = i + 1
                    break

        new_messages = history_messages[new_message_start_idx:]
        assert len(new_messages) > 0, "No new messages found (this should not happen)"

        new_conversations = []
        for message in new_messages:
            if message.agent_name != agent_name:
                # Since there are more than one player, we need to distinguish between the players
                new_conversations.append(f"[{message.agent_name}]: {message.content}")

        if request_msg:
            new_conversations.append(
                f"[{request_msg.agent_name}]: {request_msg.content}"
            )

        # Concatenate all new messages into one message because the Cohere API only accepts one message
        new_message = "\n".join(new_conversations)

        # persona_prompt = [{"": f"Environment:\n{premable}\n\nYour role:\n{role_desc}"}]

        # print("context:", context)
        persona_prompt = context.copy()
        for message in history_messages:
            # print("message:", message.message_dict)
            persona_prompt += message.message_dict
        # print("persona_prompt at query:", persona_prompt)

        response = self._get_response(new_message, persona_prompt)

        # Only update the last message hash if the API call is successful
        self.last_msg_hash = new_messages[-1].msg_hash

        return response
