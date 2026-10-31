"""Simple chat example using MCPAgent with built-inI conversation memory.

This example demonstrates how to use the MCPAgent with its built-in
conversation history capabilities for better contextual interactions.

Special thanks to https://github.com/microsoft/playwright-mcp for the server."""


import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


async def run_memory_chat():

    # Load environment variables
    load_dotenv()

    # MCP configuration
    config_file = "browser_mcp.json"

    print("Initializing chat...")

    # Create MCP client
    client = MCPClient.from_config_file(config_file)

    # Initialize LLM
    llm = ChatGroq(model="openai/gpt-oss-120b")

    # Create MCP agent with memory
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True
    )

    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end")
    print("Type 'clear' to clear conversation history")

    try:
        while True:

            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            try:
                print("\nAssistant: ", end="", flush=True)

                response = await agent.run(user_input)

                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())
