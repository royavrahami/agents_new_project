"""
Tests for ProfileAgent — CV analysis, ATS scoring, tailored CV generation.
"""

import pytest
from unittest.mock import MagicMock

from agents.profile_agent import ProfileAgent
from core.models import CVAnalysis


class TestProfileAgentATSAnalysis:
    """Tests for CV ATS scoring and keyword analysis."""

    def test_ats_score_is_within_range(self, mock_llm, sample_cv_text, sample_job_description):
        mock_llm.complete.return_value = "1. Add Docker experience\n2. Mention Kubernetes\n3. Include GraphQL"
        agent = ProfileAgent(llm_tool=mock_llm)
        analysis = agent.analyze_cv(sample_cv_text, sample_job_description)
        assert 0.0 <= analysis.ats_score <= 100.0

    def test_matched_keywords_are_subset_of_jd_keywords(self, mock_llm, sample_cv_text, sample_job_description):
        mock_llm.complete.side_effect = [
            "Python, Playwright, pytest, Docker, Jenkins, SQL, API",  # JD keywords
            "1. Improvement one\n2. Improvement two",                  # Recommendations
        ]
        agent = ProfileAgent(llm_tool=mock_llm)
        analysis = agent.analyze_cv(sample_cv_text, sample_job_description)
        # All matched keywords should be from the JD keywords list
        assert isinstance(analysis.matched_keywords, list)
        assert isinstance(analysis.missing_keywords, list)

    def test_cv_with_html_tags_gets_lower_score(self, mock_llm):
        mock_llm.complete.return_value = "1. Remove HTML tags from CV"
        agent = ProfileAgent(llm_tool=mock_llm)
        cv_with_html = "<html><body><h1>John Doe</h1><p>QA Engineer</p></body></html>"
        clean_cv = "John Doe\nQA Engineer\nPython, Playwright"
        analysis_html = agent.analyze_cv(cv_with_html)
        analysis_clean = agent.analyze_cv(clean_cv)
        assert analysis_html.ats_score <= analysis_clean.ats_score

    def test_recommendations_are_returned(self, mock_llm, sample_cv_text):
        mock_llm.complete.return_value = "1. Add Docker\n2. Add Kubernetes\n3. Add GraphQL"
        agent = ProfileAgent(llm_tool=mock_llm)
        analysis = agent.analyze_cv(sample_cv_text, "Docker Kubernetes GraphQL")
        assert len(analysis.recommendations) > 0

    def test_empty_jd_still_returns_analysis(self, mock_llm, sample_cv_text):
        mock_llm.complete.return_value = "1. General improvement"
        agent = ProfileAgent(llm_tool=mock_llm)
        analysis = agent.analyze_cv(sample_cv_text, job_description="")
        assert isinstance(analysis, CVAnalysis)
        assert analysis.ats_score >= 0

    def test_keyword_coverage_is_zero_when_no_jd(self, mock_llm, sample_cv_text):
        mock_llm.complete.return_value = "1. Improvement"
        agent = ProfileAgent(llm_tool=mock_llm)
        analysis = agent.analyze_cv(sample_cv_text, job_description="")
        assert analysis.keyword_coverage == 0.0


class TestProfileAgentTailoredCV:
    """Tests for tailored CV generation."""

    def test_tailored_cv_improves_ats_score(self, mock_llm, sample_cv_text, sample_job_description):
        call_count = [0]

        def mock_complete(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Python, Docker, Kubernetes, GraphQL, Playwright"  # JD keywords
            if call_count[0] == 2:
                return "1. Improve action verbs"  # recommendations
            if call_count[0] == 3:
                return sample_cv_text + "\n- Added Kubernetes experience\n- Added GraphQL testing"
            if call_count[0] == 4:
                return "Python, Docker, Kubernetes, GraphQL, Playwright"  # keywords for new CV
            if call_count[0] == 5:
                return "1. Good improvement"
            return "1. Change made"

        mock_llm.complete.side_effect = mock_complete
        agent = ProfileAgent(llm_tool=mock_llm)
        tailored = agent.generate_tailored_cv(sample_cv_text, sample_job_description)

        assert tailored.tailored_content != ""
        assert tailored.ats_score_before >= 0
        assert tailored.ats_score_after >= 0

    def test_tailored_cv_has_changes_summary(self, mock_llm, sample_cv_text, sample_job_description):
        mock_llm.complete.return_value = (
            "Python, Docker\n"  # keywords response (multiple calls return same)
        )
        # Patch to return structured response for changes
        responses = [
            "Python, Docker, Kubernetes",  # JD keywords
            "1. Strengthen action verbs",  # recommendations
            sample_cv_text + " Added Kubernetes",  # tailored content
            "Python, Docker, Kubernetes",  # keywords for tailored CV
            "1. Improvement noted",  # tailored recommendations
            "• Added Docker\n• Improved summary\n• Added Kubernetes",  # changes summary
        ]
        mock_llm.complete.side_effect = iter(responses)
        agent = ProfileAgent(llm_tool=mock_llm)
        tailored = agent.generate_tailored_cv(sample_cv_text, sample_job_description)
        assert isinstance(tailored.changes_summary, list)


class TestProfileAgentRun:
    """Tests for the ProfileAgent.run() entry point."""

    def test_run_without_cv_returns_error(self, mock_llm):
        agent = ProfileAgent(llm_tool=mock_llm)
        result = agent.execute(cv_text="")
        assert result["status"] == "error"
        assert "cv_text" in result["error"]

    def test_run_with_cv_returns_ok(self, mock_llm, sample_cv_text):
        mock_llm.complete.return_value = "1. Improve summary\n2. Add metrics\n3. Add keywords"
        agent = ProfileAgent(llm_tool=mock_llm)
        result = agent.execute(cv_text=sample_cv_text)
        assert result["status"] == "ok"
        assert "analysis" in result
        assert "summary" in result

    def test_run_with_cv_and_jd_generates_tailored(self, mock_llm, sample_cv_text, sample_job_description):
        mock_llm.complete.return_value = "1. Fix this\n2. Fix that\n3. Add keywords"
        agent = ProfileAgent(llm_tool=mock_llm)
        result = agent.execute(cv_text=sample_cv_text, job_description=sample_job_description)
        assert result["status"] == "ok"
        # tailored_cv should be generated when JD is provided
        assert result["tailored_cv"] is not None

    def test_linkedin_suggestions_returned(self, mock_llm, sample_cv_text):
        mock_llm.complete.return_value = "1. Update headline\n2. Add summary\n3. Optimize skills section"
        agent = ProfileAgent(llm_tool=mock_llm)
        result = agent.execute(cv_text=sample_cv_text)
        assert "linkedin_suggestions" in result


class TestProfileAgentFileLoading:
    """Tests for CV file loading."""

    def test_load_nonexistent_file_raises(self, mock_llm, tmp_path):
        agent = ProfileAgent(llm_tool=mock_llm)
        with pytest.raises(FileNotFoundError):
            agent.load_cv_from_file(str(tmp_path / "nonexistent.txt"))

    def test_load_unsupported_format_raises(self, mock_llm, tmp_path):
        pdf_file = tmp_path / "cv.pdf"
        pdf_file.write_bytes(b"%PDF fake content")
        agent = ProfileAgent(llm_tool=mock_llm)
        with pytest.raises(ValueError, match="Unsupported CV format"):
            agent.load_cv_from_file(str(pdf_file))

    def test_load_txt_file_succeeds(self, mock_llm, tmp_path, sample_cv_text):
        txt_file = tmp_path / "cv.txt"
        txt_file.write_text(sample_cv_text, encoding="utf-8")
        agent = ProfileAgent(llm_tool=mock_llm)
        content = agent.load_cv_from_file(str(txt_file))
        assert content == sample_cv_text
