"""
main.py - Entry point for the Problem Discovery & Business Insights AI system.
"""

import json
import logging

from app.config import GEMINI_API_KEY
from app.ai.gemini_client import GeminiClient
from app.ai.problem_analyzer import ProblemAnalyzer
from app.data.web_search import WebSearchTool
from app.pipeline.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QUERY = "common problems people face in daily life"


def main():
    """Initialize all components and run the problem discovery pipeline."""

    print("\n🔍 Problem Discovery & Business Insights AI")
    print("=" * 50)

    query = input("Enter search query (or press Enter for default): ").strip()
    if not query:
        query = DEFAULT_QUERY
        print(f"Using default query: '{query}'")

    print(f"\n▶ Starting pipeline for: '{query}'")
    print("=" * 50)

    try:
        # Initialize components
        gemini = GeminiClient()
        analyzer = ProblemAnalyzer(gemini)
        searcher = WebSearchTool()

        # Optional: plug in crew runner for enriched output
        # from app.agents.crew import run_crew
        # orchestrator = Orchestrator(searcher, analyzer, crew_runner=run_crew)

        orchestrator = Orchestrator(searcher, analyzer)

        # Run pipeline
        result = orchestrator.run(query)

        # Print readable summary
        print("\n✅ Pipeline complete!")
        print("=" * 50)
        print(f"Query          : {result['query']}")
        print(f"Total Sources  : {result['total_sources']}")
        print(f"Insights Found : {len(result['insights'])}")
        print(f"\nFinal Summary  :\n{result['final_summary'] or 'N/A'}")

        if result["top_opportunities"]:
            print("\nTop Opportunities:")
            for i, opp in enumerate(result["top_opportunities"], 1):
                print(f"  {i}. {opp}")

        # Full JSON output
        print("\n--- Full JSON Output ---")
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
