import logging
from dotenv import load_dotenv
import os
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logging.getLogger("my_agents").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def test_llm_output():
    from my_agents.core.llm_adapters.openai_adapter import OpenAIAdapter
    adapter = OpenAIAdapter(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=120,
        model=os.getenv("LLM_MODEL_ID"),
    )
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ]
    response = adapter.stream_invoke(
        messages=messages,
    )
    for chunk in response:
        print(chunk, flush=True)

def test_simple_agent():
    from my_agents.agents import SimpleAgent
    from my_agents.core import AgenticLLM
    #from my_agents.tools import tool
    agent = SimpleAgent(
        name="simple_agent",
        llm= AgenticLLM(),
    )
    query = "什么是递归？"
    r = agent.run(query)
    print(r)


if __name__ == "__main__":
    #test_llm_output()
    test_simple_agent()
