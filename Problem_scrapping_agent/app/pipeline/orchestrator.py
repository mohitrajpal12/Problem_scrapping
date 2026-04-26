"""
orchestrator.py - Pipeline orchestrator for problem discovery and business insight generation.

Flow: WebSearch → ProblemAnalyzer (per item) → Aggregate → (optional) Crew → Final Output
"""

import logging
from typing import Callable

from app.ai.problem_analyzer import ProblemAnalyzer
from app.data.web_search import WebSearchTool

logger = logging.getLogger(__name__)

MAX_SOURCES_TO_ANALYZE = 5  # cap LLM calls to avoid excessive usage


class Orchestrator:
    """Controls the full pipeline from web search to structured business insights."""

    def __init__(
        self,
        searcher: WebSearchTool,
        analyzer: ProblemAnalyzer,
        crew_runner: Callable[[str], dict] | None = None,
    ):
        """
        Args:
            searcher:     Initialized WebSearchTool instance.
            analyzer:     Initialized ProblemAnalyzer instance.
            crew_runner:  Optional callable — run_crew(query) from crew.py.
                          If provided, aggregated insights are passed through the crew
                          for a refined final output.
        """
        self.searcher = searcher
        self.analyzer = analyzer
        self.crew_runner = crew_runner

    def run(self, query: str) -> dict:
        """
        Execute the full problem-discovery pipeline for a given query.

        Steps:
            1. Fetch raw content from web search results.
            2. Analyze each content item with ProblemAnalyzer.
            3. Aggregate insights, deduplicating solution ideas.
            4. Optionally pass through crew runner for enriched output.
            5. Return structured final result.

        Args:
            query: The search topic (e.g. "remote work burnout").

        Returns:
            A dict with keys: query, total_sources, insights,
            final_summary, top_opportunities.
        """
        logger.info("=== Orchestrator starting for query: '%s' ===", query)

        # Step 1: Fetch data
        raw_items = self._fetch(query)
        if not raw_items:
            logger.warning("No content fetched. Returning empty result.")
            return self._empty_result(query)

        # Step 2: Analyze each item
        insights = self._analyze(raw_items)
        if not insights:
            logger.warning("No insights extracted. Returning empty result.")
            return self._empty_result(query)

        # Step 3: Aggregate
        aggregated = self._aggregate(insights)

        # Step 4: Optional crew enrichment
        final_summary, top_opportunities = self._enrich(query, aggregated)

        logger.info("=== Orchestrator complete. Sources: %d, Insights: %d ===",
                    len(raw_items), len(insights))

        return {
            "query": query,
            "total_sources": len(raw_items),
            "insights": aggregated,
            "final_summary": final_summary,
            "top_opportunities": top_opportunities,
        }

    # ---------------------------------------------------------------------------
    # Private pipeline steps
    # ---------------------------------------------------------------------------

    def _fetch(self, query: str) -> list[dict]:
        """Step 1: Fetch raw content items from web search."""
        logger.info("Step 1: Fetching web content...")
        try:
            items = self.searcher.get_problem_data(query)
            logger.info("Fetched %d content items.", len(items))
            return items
        except Exception as e:
            logger.error("Fetch failed: %s", e)
            return []

    def _analyze(self, raw_items: list[dict]) -> list[dict]:
        """Step 2: Analyze each content item, capped at MAX_SOURCES_TO_ANALYZE."""
        logger.info("Step 2: Analyzing content (max %d items)...", MAX_SOURCES_TO_ANALYZE)
        insights = []

        for item in raw_items[:MAX_SOURCES_TO_ANALYZE]:
            url = item.get("url", "unknown")
            content = item.get("content", "")
            try:
                result = self.analyzer.analyze(content)
                # Skip empty/fallback results (all fields blank)
                if any(result.get(k) for k in ("problem_summary", "business_opportunity")):
                    result["source_url"] = url
                    insights.append(result)
                    logger.info("Analyzed: %s", url)
                else:
                    logger.warning("Skipping empty insight for: %s", url)
            except Exception as e:
                logger.warning("Analysis failed for %s: %s", url, e)

        logger.info("Extracted %d valid insights.", len(insights))
        return insights

    def _aggregate(self, insights: list[dict]) -> list[dict]:
        """
        Step 3: Deduplicate solution_ideas across all insights.

        Keeps each insight dict intact but removes duplicate solution ideas
        within each item to keep output clean.
        """
        logger.info("Step 3: Aggregating and deduplicating insights...")
        seen_summaries = set()
        aggregated = []

        for insight in insights:
            summary = insight.get("problem_summary", "").strip().lower()
            # Skip near-duplicate problem summaries
            if summary and summary in seen_summaries:
                logger.info("Skipping duplicate insight: '%s'", summary[:60])
                continue
            if summary:
                seen_summaries.add(summary)

            # Deduplicate solution_ideas within this insight
            ideas = insight.get("solution_ideas", [])
            insight["solution_ideas"] = list(dict.fromkeys(ideas))  # preserves order
            aggregated.append(insight)

        logger.info("Aggregated to %d unique insights.", len(aggregated))
        return aggregated

    def _enrich(self, query: str, aggregated: list[dict]) -> tuple[str, list[str]]:
        """
        Step 4: Optionally enrich via crew runner, otherwise derive from aggregated insights.

        Returns:
            (final_summary, top_opportunities)
        """
        if self.crew_runner:
            logger.info("Step 4: Running crew enrichment...")
            try:
                crew_output = self.crew_runner(query)
                final_summary = crew_output.get("refined_opportunity") or crew_output.get("raw_output", "")
                top_opportunities = crew_output.get("revenue_models") or crew_output.get("solution_ideas", [])
                return str(final_summary), [str(o) for o in top_opportunities]
            except Exception as e:
                logger.warning("Crew enrichment failed, falling back: %s", e)

        # Fallback: derive summary and opportunities directly from aggregated insights
        summaries = [i.get("business_opportunity", "") for i in aggregated if i.get("business_opportunity")]
        all_ideas = [idea for i in aggregated for idea in i.get("solution_ideas", [])]

        final_summary = summaries[0] if summaries else ""
        top_opportunities = list(dict.fromkeys(all_ideas))[:10]  # top 10 unique ideas

        return final_summary, top_opportunities

    @staticmethod
    def _empty_result(query: str) -> dict:
        return {
            "query": query,
            "total_sources": 0,
            "insights": [],
            "final_summary": "",
            "top_opportunities": [],
        }
