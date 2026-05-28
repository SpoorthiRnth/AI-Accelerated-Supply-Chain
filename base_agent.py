# Base class for all supply chain AI agents

import json
import time
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from colorama import Fore, Style, init

init(autoreset=True)


class BaseAgent:
    """
    Base class for all supply chain agents.
    Every agent must implement run(input_data) -> dict
    """

    def __init__(self, agent_name: str, zone: str):
        self.agent_name = agent_name
        self.zone = zone
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.model = AZURE_OPENAI_DEPLOYMENT

    def _print_header(self):
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"  AGENT  : {self.agent_name}")
        print(f"  ZONE   : {self.zone}")
        print(f"{'='*70}{Style.RESET_ALL}")

    def _print_section(self, title: str):
        print(f"\n{Fore.YELLOW}  >> {title}{Style.RESET_ALL}")

    def _print_output(self, label: str, value):
        if isinstance(value, (dict, list)):
            print(f"{Fore.GREEN}  {label}:{Style.RESET_ALL}")
            print(f"  {json.dumps(value, indent=4, default=str)[:2000]}")
        else:
            print(f"{Fore.GREEN}  {label}: {Style.RESET_ALL}{str(value)[:500]}")

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        #Call Azure OpenAI GPT-5 with system and user prompts. Returns the text response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"{Fore.RED}  LLM call failed: {e}{Style.RESET_ALL}")
            return json.dumps({"error": str(e), "fallback": True})

    def run(self, input_data: dict) -> dict:
        raise NotImplementedError("Each agent must implement run()")
