"""
Profile Agent — CV Optimizer & ATS Scorer.

Responsibilities:
1. Analyze a CV against a job description → ATS score + recommendations
2. Generate a tailored version of the CV for a specific role
3. Optimize LinkedIn profile headline + summary for recruiter search
4. Track improvement over iterations
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from config.settings import settings
from core.logger import logger
from core.models import CVAnalysis, TailoredCV
from tools.llm_tool import LLMTool
from tools.cv_parser_tool import cv_parser

from .base_agent import BaseAgent


class ProfileAgent(BaseAgent):
    """
    Analyzes and improves candidate-facing materials (CV, LinkedIn)
    to maximize ATS pass-through rate and recruiter visibility.

    Args:
        llm_tool: LLMTool instance (injected for testability)
    """

    # Common ATS-disqualifying patterns
    _ATS_KILLERS = [
        r"<[^>]+>",           # HTML tags
        r"\.(png|jpg|jpeg)",  # Image references in CV
        r"[^\x00-\x7F]{3,}", # Long non-ASCII sequences (tables, graphics)
    ]

    def __init__(self, llm_tool: Optional[LLMTool] = None) -> None:
        super().__init__(name="ProfileAgent")
        self._llm = llm_tool or LLMTool(
            system_prompt=(
                "You are a senior technical recruiter and CV expert specializing in "
                "QA leadership and engineering management roles at Israeli hi-tech companies. "
                "The candidate is Roy Avrahami — QA Manager, 14+ years, ISTQB CTAL-TM, "
                "expert in quality architecture, CI/CD governance, KPI systems, and AI-driven QA. "
                "Optimize for VP Engineering, R&D Directors, and C-level hiring audiences. "
                "Focus on business impact metrics: escape rate reduction, release cadence improvement, "
                "team building, and strategic quality leadership — not just technical skills."
            )
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, cv_text: str = "", job_description: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Analyze a CV against a job description and generate a tailored version.

        Args:
            cv_text:          Raw text of the candidate's CV
            job_description:  Raw text of the target job description

        Returns:
            {
                "status": "ok",
                "analysis": CVAnalysis,
                "tailored_cv": TailoredCV | None,
                "linkedin_suggestions": List[str],
                "summary": str
            }
        """
        if not cv_text:
            return {"status": "error", "error": "cv_text is required", "data": None}

        self.logger.info("Starting CV analysis")

        # Step 1 — ATS analysis
        analysis = self.analyze_cv(cv_text, job_description)

        # Step 2 — Tailored CV (only if JD provided)
        tailored: Optional[TailoredCV] = None
        if job_description:
            tailored = self.generate_tailored_cv(cv_text, job_description, analysis)

        # Step 3 — LinkedIn optimization suggestions
        linkedin_suggestions = self.optimize_linkedin_summary(cv_text)

        summary = (
            f"CV Analysis complete. "
            f"ATS score: {analysis.ats_score:.0f}/100. "
            f"Missing {len(analysis.missing_keywords)} keywords. "
            f"{len(analysis.recommendations)} improvements suggested."
        )

        self.logger.success(summary)
        return {
            "status": "ok",
            "analysis": analysis,
            "tailored_cv": tailored,
            "linkedin_suggestions": linkedin_suggestions,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # CV Analysis
    # ------------------------------------------------------------------

    def analyze_cv(self, cv_text: str, job_description: str = "") -> CVAnalysis:
        """
        Score a CV on ATS compatibility and keyword coverage.

        ATS scoring factors:
        - Real CV parsing (Affinda API or fallback regex)
        - Structural score (no ATS killers, proper sections)
        - Keyword match ratio against job description
        - LLM-based holistic assessment

        Args:
            cv_text:         Raw CV content
            job_description: Target JD (optional; enables keyword matching)

        Returns:
            CVAnalysis with score, missing keywords, and recommendations
        """
        self.logger.info("Analyzing CV for ATS compatibility")

        # Step 1: Parse CV with real parser (Affinda or fallback)
        parse_result = cv_parser.parse_cv(cv_text)

        if parse_result.get("success"):
            self.logger.info(
                f"CV parsed successfully via {parse_result.get('parsing_method')} "
                f"({parse_result.get('parsing_success_rate', 0):.0f}% success)"
            )
        else:
            self.logger.warning(f"CV parsing failed: {parse_result.get('error', 'Unknown error')}")

        # Step 2: ATS compatibility scoring
        ats_issues = parse_result.get("formatting_issues", [])
        parsing_success = parse_result.get("parsing_success_rate", 50) / 100.0

        # Base score from parsing success and ATS issues
        base_score = (parsing_success * 60) + (max(0.0, 100.0 - len(ats_issues) * 10) * 0.4)

        # Step 3: Keyword analysis
        jd_keywords = self._extract_keywords_from_jd(job_description) if job_description else []
        matched, missing = self._match_keywords(cv_text, jd_keywords)

        # Also check against parsed skills
        parsed_skills = [s.get("name", "") for s in parse_result.get("skills_extracted", [])]
        for skill in parsed_skills:
            if skill.lower() in [m.lower() for m in matched]:
                continue
            matched.append(skill)

        keyword_coverage = len(matched) / max(len(jd_keywords), 1) if jd_keywords else 0.5
        keyword_score = keyword_coverage * 40  # Up to 40 points for keywords

        raw_score = min(base_score * 0.6 + keyword_score, 100.0)

        # Step 4: LLM refinement — get qualitative recommendations
        recommendations = self._get_llm_recommendations(cv_text, job_description, raw_score)

        # Add parsing-specific recommendations if issues found
        if ats_issues:
            recommendations.insert(0, f"ATS issues detected: {', '.join(ats_issues[:3])}")

        # Adjust score based on recommendation severity
        score_adjustment = sum(
            -5 for r in recommendations if any(
                word in r.lower() for word in ["missing", "add", "improve", "lack"]
            )
        )
        final_score = max(0.0, min(raw_score + score_adjustment, 100.0))

        return CVAnalysis(
            ats_score=round(final_score, 1),
            keyword_coverage=round(keyword_coverage, 3),
            missing_keywords=missing[:20],
            matched_keywords=matched[:20],
            recommendations=recommendations[:10],
        )

    def _check_ats_killers(self, cv_text: str) -> List[str]:
        """Return list of ATS-disqualifying patterns found in the CV."""
        found = []
        for pattern in self._ATS_KILLERS:
            if re.search(pattern, cv_text, re.IGNORECASE):
                found.append(pattern)
        return found

    def _extract_keywords_from_jd(self, jd: str) -> List[str]:
        """
        Extract meaningful keywords from a job description.
        Uses LLM to identify the 20 most important technical/soft keywords.

        Args:
            jd: Raw job description text

        Returns:
            List of keyword strings
        """
        prompt = (
            f"Extract the 20 most important keywords (technical skills, tools, "
            f"certifications, soft skills) from this job description. "
            f"Return ONLY a comma-separated list, nothing else.\n\nJD:\n{jd[:3000]}"
        )
        raw = self._llm.complete(prompt, max_tokens=200)
        keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
        return keywords

    def _match_keywords(
        self, cv_text: str, keywords: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Match keywords against CV text (case-insensitive).

        Returns:
            Tuple of (matched_keywords, missing_keywords)
        """
        cv_lower = cv_text.lower()
        matched = [kw for kw in keywords if kw.lower() in cv_lower]
        missing = [kw for kw in keywords if kw.lower() not in cv_lower]
        return matched, missing

    def _get_llm_recommendations(
        self, cv_text: str, jd: str, raw_score: float
    ) -> List[str]:
        """
        Use LLM to generate actionable CV improvement recommendations.

        Args:
            cv_text:   Current CV text
            jd:        Target job description
            raw_score: Current score (gives LLM context)

        Returns:
            List of recommendation strings
        """
        prompt = (
            f"Current ATS score: {raw_score:.0f}/100.\n\n"
            f"CV (first 2000 chars):\n{cv_text[:2000]}\n\n"
            f"{'JD (first 1000 chars):' + chr(10) + jd[:1000] if jd else ''}\n\n"
            f"List up to 8 specific, actionable improvements the candidate should make "
            f"to increase their ATS score and recruiter appeal. "
            f"Format as a numbered list, one per line."
        )
        raw = self._llm.complete(prompt, max_tokens=600)

        # Parse numbered list
        lines = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in raw.splitlines()
            if line.strip() and re.match(r"^\d+", line.strip())
        ]
        return lines if lines else [raw.strip()]

    # ------------------------------------------------------------------
    # Tailored CV Generation
    # ------------------------------------------------------------------

    def generate_tailored_cv(
        self,
        cv_text: str,
        job_description: str,
        analysis: Optional[CVAnalysis] = None,
        job_id: Optional[UUID] = None,
    ) -> TailoredCV:
        """
        Generate a version of the CV tailored to a specific job description.
        Adds missing keywords, reorders sections, and sharpens language.

        Args:
            cv_text:         Original CV text
            job_description: Target JD
            analysis:        Pre-computed CVAnalysis (will compute if not provided)
            job_id:          UUID of the associated HiddenJob

        Returns:
            TailoredCV with before/after scores and a summary of changes
        """
        self.logger.info("Generating tailored CV")
        if analysis is None:
            analysis = self.analyze_cv(cv_text, job_description)

        prompt = (
            f"You are a senior CV writer. Rewrite the candidate's CV to maximize "
            f"ATS score for the target job. Rules:\n"
            f"1. Naturally weave in these missing keywords: {', '.join(analysis.missing_keywords[:10])}\n"
            f"2. Keep all factual experience — never invent or exaggerate\n"
            f"3. Use strong action verbs (Led, Built, Automated, Reduced, Improved)\n"
            f"4. Keep it to 2 pages max\n"
            f"5. Maintain ATS-friendly plain text format (no tables, no graphics)\n\n"
            f"ORIGINAL CV:\n{cv_text[:3000]}\n\n"
            f"TARGET JD:\n{job_description[:1500]}\n\n"
            f"Return ONLY the rewritten CV text."
        )
        tailored_content = self._llm.complete(prompt, max_tokens=2000)

        # Score the new version
        new_analysis = self.analyze_cv(tailored_content, job_description)

        changes_prompt = (
            f"Compare original CV (ATS={analysis.ats_score}) with tailored CV (ATS={new_analysis.ats_score}). "
            f"List 5 key changes made. Format as bullet points."
        )
        changes_raw = self._llm.complete(changes_prompt, max_tokens=300)
        changes = [
            line.strip().lstrip("•-* ")
            for line in changes_raw.splitlines()
            if line.strip()
        ]

        return TailoredCV(
            job_id=job_id or uuid4(),
            original_cv_path="",
            tailored_content=tailored_content,
            ats_score_before=analysis.ats_score,
            ats_score_after=new_analysis.ats_score,
            changes_summary=changes[:5],
        )

    # ------------------------------------------------------------------
    # LinkedIn Optimization
    # ------------------------------------------------------------------

    def optimize_linkedin_summary(self, cv_text: str) -> List[str]:
        """
        Generate LinkedIn profile optimization suggestions.
        LinkedIn is the primary recruiter Sourcing tool — visibility here
        directly impacts the number of inbound opportunities.

        Args:
            cv_text: CV text used as context

        Returns:
            List of actionable LinkedIn improvement suggestions
        """
        self.logger.info("Generating LinkedIn optimization suggestions")
        prompt = (
            f"Based on this CV, provide 6 specific suggestions to optimize the "
            f"LinkedIn profile for maximum recruiter discoverability in Israeli hi-tech. "
            f"Focus on: headline, about section, skills, Open To Work settings, "
            f"keyword placement, and engagement tactics.\n\n"
            f"CV summary:\n{cv_text[:1500]}\n\n"
            f"Format as a numbered list."
        )
        raw = self._llm.complete(prompt, max_tokens=500)
        return [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in raw.splitlines()
            if line.strip() and re.match(r"^\d+", line.strip())
        ]

    def load_cv_from_file(self, path: str) -> str:
        """
        Read CV text from a file (supports .txt and .md).
        PDF support can be added with pdfminer.six if needed.

        Args:
            path: Filesystem path to the CV file

        Returns:
            Raw text content of the CV

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CV file not found: {path}")
        if p.suffix.lower() not in {".txt", ".md"}:
            raise ValueError(f"Unsupported CV format: {p.suffix}. Use .txt or .md")
        return p.read_text(encoding="utf-8")
