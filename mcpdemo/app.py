import asyncio 

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPAgent , MCPClient
import os 



async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("API_KEY")


    config_file = "browser_mcp.json"
    print("Intializing chat...")


    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
    )

    agent = MCPAgent(
        llm=llm, 
        client=client,
        max_steps=15,
        memory_enabled=True
    )

    print("\n===== INTERACTIVE MCP CHAT =====")
    print("Type 'exit' or quit to end the conversation.")
    print("Type 'clear' to clear the conversation history.")
    print("===================================")

    try:

        while True: 
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            
            if user_input.lower() == "clear":
                print("Clearing conversation history...")
                agent.clear_memory()
                continue
            
            
            print("\nAssistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"An error occurred: {e}")

    finally:
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())
                
                
            
  
