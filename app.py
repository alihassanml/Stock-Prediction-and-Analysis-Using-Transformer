from fastapi import FastAPI, Query
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="AI Multi-Agent API")

# Define Web Search Agent
web_Search_agent = Agent(
    name='Web Search Agent',
    role='Search the web for information',
    model=Groq(id='deepseek-r1-distill-llama-70b', api_key=groq_api),
    tools=[DuckDuckGo],
    instructions=["Always include source"],
    show_tool_calls=True,
    markdown=True,
)

# Define Finance Agent
finance_agent = Agent(
    name='Finance AI Agent',
    model=Groq(id='deepseek-r1-distill-llama-70b', api_key=groq_api),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use instructions to display the data.",
                 "Financial analyst specialized in breaking news and investment insights."],
    show_tool_calls=True,
    markdown=True,
)

# Define Multi-Agent
multi_AI_Agent = Agent(
    team=[web_Search_agent, finance_agent],
    model=Groq(id='deepseek-r1-distill-llama-70b', api_key=groq_api),
    instructions=[
    "When searching for news, prioritize the most recent and reliable sources.",
    "Return structured results with headline, summary, date, and source.",
    "Ignore outdated or speculative news unless specifically requested.",
    "When showing stock data, include current price, trend, and latest analyst sentiment."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Define endpoint to handle queries
@app.get("/ask")
async def ask_agent(query: str = Query(..., description="Query for the multi-agent system")):
    response = multi_AI_Agent.run(query)
    return {"response": response.content}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
