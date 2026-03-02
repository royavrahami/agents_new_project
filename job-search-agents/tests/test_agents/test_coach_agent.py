"""
Tests for CoachAgent — interview preparation and post-interview debriefing.
"""

import pytest
from unittest.mock import MagicMock

from agents.coach_agent import CoachAgent
from core.models import InterviewPrep, PipelineEntry
from config.settings import JobSearchStatus


class TestCoachAgentPreparation:
    """Tests for interview preparation generation."""

    def test_prepare_returns_interview_prep(self, mock_llm):
        mock_llm.complete.return_value = (
            "1. What testing framework do you prefer?\n"
            "2. Describe your CI/CD experience\n"
            "3. How do you handle flaky tests?\n"
            "4. Describe a time you found a critical bug\n"
            "5. How do you prioritize test cases?"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare(
            company_name="Acme Corp",
            role_title="QA Automation Engineer",
        )
        assert isinstance(prep, InterviewPrep)
        assert prep.company_research != ""

    def test_prepare_generates_questions(self, mock_llm):
        mock_llm.complete.return_value = (
            "1. Technical question one\n"
            "2. Behavioral question two\n"
            "3. Situational question three\n"
            "4. Culture-fit question four\n"
            "5. Architecture question five"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare("Acme", "QA Engineer")
        assert len(prep.likely_questions) >= 1

    def test_prepare_generates_star_examples(self, mock_llm):
        mock_llm.complete.return_value = (
            "STORY 1: Situation: [DESCRIBE SITUATION]. Task: [YOUR TASK]. "
            "Action: [ACTIONS YOU TOOK]. Result: [OUTCOME ACHIEVED].\n"
            "STORY 2: Situation: [CRISIS]. Task: [YOUR ROLE]. Action: [STEPS]. Result: [IMPACT].\n"
            "STORY 3: Situation: [LEADERSHIP]. Task: [OBJECTIVE]. Action: [HOW]. Result: [NUMBERS]."
        )
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare("Acme", "QA Engineer")
        assert isinstance(prep.star_examples, list)

    def test_prepare_generates_questions_to_ask(self, mock_llm):
        mock_llm.complete.return_value = (
            "1. How does the team handle production incidents?\n"
            "2. What does success look like in 90 days?\n"
            "3. How is the QA team structured?"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare("Acme", "QA Engineer")
        assert len(prep.questions_to_ask) >= 1

    def test_prepare_identifies_red_flags(self, mock_llm):
        mock_llm.complete.return_value = (
            "1. Vague role responsibilities\n"
            "2. No mention of team size\n"
            "3. Unrealistic 'startup pace' language"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare("Acme", "QA Engineer", job_description="Fast-paced environment")
        assert isinstance(prep.red_flags, list)

    def test_prepare_with_pipeline_entry(self, mock_llm, sample_pipeline_entry):
        mock_llm.complete.return_value = "1. Question one\n2. Question two"
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.execute(pipeline_entry=sample_pipeline_entry)
        assert result["status"] == "ok"
        assert isinstance(result["prep"], InterviewPrep)

    def test_prepare_with_job_description_provided(self, mock_llm, sample_job_description):
        mock_llm.complete.return_value = "1. JD-specific question"
        agent = CoachAgent(llm_tool=mock_llm)
        prep = agent.prepare(
            company_name="Acme",
            role_title="QA Engineer",
            job_description=sample_job_description,
        )
        assert prep.likely_questions is not None


class TestCoachAgentDebrief:
    """Tests for post-interview debriefing logic."""

    def test_debrief_rejection_returns_structured_output(self, mock_llm):
        mock_llm.complete.return_value = (
            "STRENGTHS: Good technical depth, clear communication\n"
            "AREAS TO IMPROVE: Improve system design answers, be more concise\n"
            "ACTION PLAN: Practice system design, mock interviews, STAR framework"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.debrief(
            outcome="rejected",
            interview_notes="Technical round went ok but system design was weak.",
            rejection_reason="System design skills not at required level",
        )
        assert "strengths" in result
        assert "areas_to_improve" in result
        assert "action_plan" in result

    def test_debrief_offer_returns_structured_output(self, mock_llm):
        mock_llm.complete.return_value = (
            "STRENGTHS: Strong automation background, proactive\n"
            "AREAS TO IMPROVE: Negotiate salary more confidently\n"
            "ACTION PLAN: Research market rates, prepare negotiation talking points"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.debrief(
            outcome="offer",
            interview_notes="All rounds went well. Received offer.",
        )
        assert isinstance(result, dict)

    def test_debrief_with_empty_notes(self, mock_llm):
        mock_llm.complete.return_value = (
            "STRENGTHS: Unable to assess\n"
            "AREAS TO IMPROVE: Provide more detail\n"
            "ACTION PLAN: Keep notes next time"
        )
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.debrief(outcome="rejected", interview_notes="")
        assert isinstance(result, dict)


class TestCoachAgentRun:
    """Tests for CoachAgent.run() entry point."""

    def test_run_without_pipeline_entry_uses_direct_args(self, mock_llm):
        mock_llm.complete.return_value = "1. Question one\n2. Question two"
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.execute(
            company_name="Acme Tech",
            role_title="QA Engineer",
            job_description="Python, Playwright required",
        )
        assert result["status"] == "ok"
        assert "prep" in result

    def test_run_summary_contains_company_name(self, mock_llm):
        mock_llm.complete.return_value = "1. Technical question\n2. Behavioral question"
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.execute(company_name="TargetCo", role_title="QA Lead")
        assert "TargetCo" in result["summary"]

    def test_run_returns_question_count_in_summary(self, mock_llm):
        mock_llm.complete.return_value = "1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5"
        agent = CoachAgent(llm_tool=mock_llm)
        result = agent.execute(company_name="Acme", role_title="QA Engineer")
        # Summary should mention the number of questions
        assert any(char.isdigit() for char in result["summary"])


class TestCoachAgentHelpers:
    """Tests for internal helper methods."""

    def test_parse_numbered_list_standard_format(self):
        text = "1. First item\n2. Second item\n3. Third item"
        result = CoachAgent._parse_numbered_list(text)
        assert len(result) == 3
        assert result[0] == "First item"

    def test_parse_numbered_list_with_parenthesis(self):
        text = "1) Question one\n2) Question two"
        result = CoachAgent._parse_numbered_list(text)
        assert len(result) == 2

    def test_parse_numbered_list_ignores_non_numbered(self):
        text = "Some intro text\n1. First\n2. Second\nConclusion"
        result = CoachAgent._parse_numbered_list(text)
        assert len(result) == 2

    def test_parse_empty_string_returns_empty_list(self):
        result = CoachAgent._parse_numbered_list("")
        assert result == []
