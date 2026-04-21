# LLM mission planner: sends mission briefs to Claude API, parses JSON zone assignments and priorities.

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))