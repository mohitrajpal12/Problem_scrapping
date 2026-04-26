"""
crew_setup.py - CrewAI setup for discovering real-world problems and generating business insights.

Pipeline (sequential):
    ResearchAgent → ProblemAnalystAgent → OpportunityAgent
"""

import logging
from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from pydantic import Field

from app.ai.gemini_client import GeminiClient
from app.ai.problem_analyzer import ProblemAnalyzer
from app.data.web_search import WebSearchTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI-compatible tool wrappers
# ---------------------------------------------------------------------------

class WebSearchCrewTool(BaseTool):
    """Wraps WebSearchTool as a CrewAI-compatible tool."""

    name: str = "web_search"
    description: str = "Search the web for a query and return scraped text content from top results."
    _searcher: WebSearchTool = None

    def __init__(self, searcher: WebSearchTool):
        super().__init__()
        self._searcher = searcher

    def _run(self, query: str) -> str:
        results = self._searcher.get_problem_data(query)
        return "\n\n".join(
            f"URL: {r['url']}\n{r['content']}" for r in results
        ) or "No results found."


class ProblemAnalyzerCrewTool(BaseTool):
    """Wraps ProblemAnalyzer as a CrewAI-compatible tool."""

    name: str = "problem_analyzer"
    description: str = "Analyze raw web content and extract structured problem insights as JSON."
    _analyzer: ProblemAnalyzer = None

    def __init__(self, analyzer: ProblemAnalyzer):
        super().__init__()
        self._analyzer = analyzer

    def _run(self, content: str) -> str:
        import json
        result = self._analyzer.analyze(content)
        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def _build_agents(
    web_tool: WebSearchCrewTool,
    analyzer_tool: ProblemAnalyzerCrewTool,
) -> tuple[Agent, Agent, Agent]:
    """Instantiate and return the three pipeline agents."""

    research_agent = Agent(
        role="Internet Research Specialist",
        goal="Find real-world problems people discuss online by searching the web.",
        backstory=(
            "You are an expert at discovering pain points from forums, Reddit threads, "
            "product reviews, and online communities. You surface raw, unfiltered user frustrations."
        ),
        tools=[web_tool],
        verbose=True,
    )

    analyst_agent = Agent(
        role="Problem Analyst",
        goal="Convert raw scraped content into structured, actionable problem intelligence.",
        backstory=(
            "You are a seasoned analyst who breaks down messy information into clear insights — "
            "identifying root causes, affected audiences, and gaps in existing solutions."
        ),
        tools=[analyzer_tool],
        verbose=True,
    )

    opportunity_agent = Agent(
        role="Business Opportunity Strategist",
        goal="Identify monetizable opportunities and refine solution ideas from problem insights.",
        backstory=(
            "You are a startup strategist with deep expertise in business models and market gaps. "
            "You turn problem insights into concrete, practical business opportunities."
        ),
        tools=[],
        verbose=True,
    )

    return research_agent, analyst_agent, opportunity_agent


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _build_tasks(
    query: str,
    research_agent: Agent,
    analyst_agent: Agent,
    opportunity_agent: Agent,
) -> list[Task]:
    """Build and return the three sequential tasks."""

    research_task = Task(
        description=(
            f"Search the web for: '{query}'.\n"
            "Use the web_search tool to collect content from top results.\n"
            "Return all scraped text as-is for the next agent."
        ),
        expected_output="Raw scraped text content from multiple web pages related to the query.",
        agent=research_agent,
    )

    analysis_task = Task(
        description=(
            "Take the raw web content from the previous task.\n"
            "Use the problem_analyzer tool to extract structured problem insights.\n"
            "Return the full JSON output from the tool."
        ),
        expected_output=(
            "A JSON object with keys: problem_summary, who_faces_it, root_cause, "
            "existing_solutions, gaps_in_solutions, business_opportunity, solution_ideas."
        ),
        agent=analyst_agent,
        context=[research_task],
    )

    opportunity_task = Task(
        description=(
            "Review the structured problem insights from the previous task.\n"
            "Refine and expand the business_opportunity and solution_ideas fields.\n"
            "Add market sizing hints, potential revenue models, and go-to-market angles.\n"
            "Return a final enriched JSON with all original fields plus:\n"
            "  - 'refined_opportunity': expanded business opportunity narrative\n"
            "  - 'revenue_models': list of potential monetization approaches\n"
            "  - 'go_to_market': brief go-to-market strategy"
        ),
        expected_output=(
            "An enriched JSON object containing all original problem insight fields "
            "plus refined_opportunity, revenue_models, and go_to_market."
        ),
        agent=opportunity_agent,
        context=[analysis_task],
    )

    return [research_task, analysis_task, opportunity_task]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_crew(query: str) -> dict:
    """
    Execute the full problem-discovery pipeline for a given search query.

    Runs three agents sequentially:
        1. ResearchAgent   — scrapes web content
        2. AnalystAgent    — extracts structured problem insights
        3. OpportunityAgent — refines business opportunities

    Args:
        query: The topic or problem domain to research (e.g. "remote work burnout").

    Returns:
        A dict containing the final enriched output from the OpportunityAgent,
        or an error dict if the crew fails.
    """
    import json

    logger.info("Starting crew for query: '%s'", query)

    gemini = GeminiClient()
    analyzer = ProblemAnalyzer(gemini)
    searcher = WebSearchTool()

    web_tool = WebSearchCrewTool(searcher)
    analyzer_tool = ProblemAnalyzerCrewTool(analyzer)

    research_agent, analyst_agent, opportunity_agent = _build_agents(web_tool, analyzer_tool)
    tasks = _build_tasks(query, research_agent, analyst_agent, opportunity_agent)

    crew = Crew(
        agents=[research_agent, analyst_agent, opportunity_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        raw_output = str(result)

        # Attempt to parse JSON from the final agent's output
        cleaned = raw_output.strip().strip("```json").strip("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_output": raw_output}

    except Exception as e:
        logger.error("Crew execution failed: %s", e)
        return {"error": str(e)}
