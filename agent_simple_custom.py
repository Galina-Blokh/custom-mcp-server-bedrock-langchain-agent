#!/usr/bin/env python3
"""
Simple Dataiku DSS Documentation Assistant using Custom Document Loading
This version uses RecursiveUrlLoader directly without MCP server complexity
"""

import asyncio
import json
import sys
import argparse
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import nest_asyncio
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from dotenv import load_dotenv
import os
import dataiku_tools as duk_tools

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Access environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Configure Bedrock model
model = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    temperature=0.1,
    max_tokens=1000,
    verbose=True
)

# All tool functions and caching moved to dataiku_tools.py

async def run_agent(iter_mode: str = "llm"):
    # Create tools list (heuristic as default search; iterative crawl first; LLM and cosine optional)
    tools = [
        duk_tools.fetch_dataiku_docs,
        duk_tools.iterative_expand_crawl,       # guided expansion first
        duk_tools.search_dataiku_docs_heuristic,  # default search
        duk_tools.search_dataiku_docs_llm,        # optional re-ranker
        duk_tools.search_dataiku_docs,            # optional cosine
        duk_tools.get_doc_urls
    ]
    
    # Create and run the agent
    agent = create_react_agent(model, tools)
    
    # Start conversation history with optimized system prompt
    # Normalize iterative mode
    iter_mode = (iter_mode or "llm").lower()
    if iter_mode not in {"cosine", "heuristic", "llm"}:
        iter_mode = "llm"

    messages = [
        {
            "role": "system",
            "content": f"""You are a specialized Dataiku DSS documentation assistant. Your role is to:

1. ONLY answer questions related to Dataiku DSS documentation and features
2. ALWAYS use the custom tools to fetch and search Dataiku documentation from https://doc.dataiku.com/dss/latest/
3. For EACH user query, FIRST run an iterative guided expansion with iterative_expand_crawl using steps=2 and branch=3 and mode=\"{iter_mode}\" to collect promising URLs. Use the returned top_overall as the final URLs for the answer. Do NOT backfill from other tools. If top_overall has fewer than 5, return fewer and explicitly state that fewer than 5 were found.
4. Use the heuristic search tool by default to compute relevance (search_dataiku_docs_heuristic) only when separately asked to run a non-iterative search.
5. Provide SHORT, concise answers with up to 5 most relevant URLs that you have ACTUALLY fetched and verified. If fewer than 5 exist for the topic, provide all available and explicitly state that fewer than 5 were found.
6. If asked about topics NOT related to Dataiku, respond with: "This is not about Dataiku. Please ask again with a Dataiku-related question."

CRITICAL INSTRUCTIONS:
- You MUST use the fetch_dataiku_docs tool first to load the documentation
- Use the first time fetched data as a source of context and knowledge base for all subsequent questions
- If there will be a next user question, continue fetching from the place you left off in the previous time to expand the knowledge base
- For each user query: run iterative_expand_crawl(steps=2, branch=3, mode=\"{iter_mode}\") to expand a relevance-guided frontier, then use its top_overall as the answer URLs. Do NOT backfill; if fewer than 5 results were found, return fewer and explicitly state that.
- Use get_doc_urls to get available URLs when needed
- PRIORITIZE instructional content over release notes, changelogs, or version information
- Focus on tutorial, guide, how-to, and getting-started pages for user questions
- AVOID release_notes URLs unless specifically asked about version changes
- NEVER make up or guess URLs - only provide URLs you have actually fetched
- Only provide URLs that exist and that you have successfully retrieved
- If a URL doesn't exist or returns an error, don't include it in your response
- Always verify URLs work before including them in your answer
- Start each conversation by fetching the documentation if not already loaded
- Build upon previous fetched content to provide increasingly comprehensive answers

Your responses should be helpful, accurate, and strictly focused on Dataiku DSS topics only."""
        }
    ]

    # Start a background prefetch with a smaller, faster crawl
    prefetch_task = asyncio.create_task(duk_tools.prefetch_docs(initial_max_docs=50, depth=1))

    # Start the conversation
    print("Dataiku DSS Documentation Assistant (Custom Document Loader)")
    print("Ask me anything about Dataiku DSS documentation!")
    print(f"Iterative mode: {iter_mode}")
    print("Type 'exit' or 'quit' to end the chat.\n")
    
    while True:
        # Get the user's message
        user_input = input("You: ")

        # Check if the user wants to end the conversation
        try:
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Goodbye!")
                return 'Goodbye!'
        except Exception as e:
            print('Goodbye!')
            return 'Goodbye!'

        # Removed Dataiku-related question filter; accept all queries

        # Add the user's message to the conversation history
        messages.append({"role": "user", "content": user_input})

        try:
            # If docs aren't ready yet, try to wait briefly for the background prefetch
            if duk_tools.cached_docs is None and prefetch_task and not prefetch_task.done():
                try:
                    await asyncio.wait_for(prefetch_task, timeout=5)
                except asyncio.TimeoutError:
                    # Continue without blocking; the agent can still call fetch tool if needed
                    print("Agent: Prefetch still running; proceeding while background loading continues...")

            # Invoke the agent with the full message history
            agent_response = await agent.ainvoke({"messages": messages})

            # Get the agent's reply
            ai_message = agent_response["messages"][-1].content

            # Add the agent's reply to the conversation history
            messages.append({"role": "assistant", "content": ai_message})

            # Print the agent's reply
            print(f"Agent: {ai_message}")
            
        except Exception as e:
            print(f"Agent: Error processing your request: {str(e)}")
            continue

# Run the async function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Dataiku DSS Documentation Assistant")
    parser.add_argument("--iter-mode", choices=["cosine", "heuristic", "llm"], default="llm", help="Iterative crawl scoring mode")
    args = parser.parse_args()

    result = asyncio.run(run_agent(iter_mode=args.iter_mode))
    print(result)
