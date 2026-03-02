"""
Coach Agent — Interview Preparation & Debriefing.

Responsibilities:
1. Research a company before an interview (product, culture, funding, competitors)
2. Generate likely interview questions tailored to the role
3. Build STAR example templates based on candidate background
4. Debrief post-interview or post-rejection → extract lessons
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from core.logger import logger
from core.models import InterviewPrep, PipelineEntry
from tools.llm_tool import LLMTool

from .base_agent import BaseAgent


class CoachAgent(BaseAgent):
    """
    Prepares the candidate for interviews and extracts learnings from rejections.

    Args:
        llm_tool: LLMTool instance (injected for testability)
    """

    def __init__(self, llm_tool: Optional[LLMTool] = None) -> None:
        super().__init__(name="CoachAgent")
        self._llm = llm_tool or LLMTool(
            system_prompt=(
                "You are a world-class interview coach specializing in Israeli hi-tech companies. "
                "You combine deep technical knowledge with behavioral interview expertise."
            )
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_entry: Optional[PipelineEntry] = None,
        company_name: str = "",
        role_title: str = "",
        job_description: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build a complete interview preparation package.

        Args:
            pipeline_entry: PipelineEntry (preferred) — used for company/role context
            company_name:   Direct company name (used if no pipeline_entry)
            role_title:     Direct role title
            job_description: Raw JD text for question generation

        Returns:
            {
                "status": "ok",
                "prep": InterviewPrep,
                "summary": str
            }
        """
        # Resolve company and role from pipeline entry or direct args
        company = (pipeline_entry.company_name if pipeline_entry else company_name) or "Unknown Company"
        role = (pipeline_entry.role_title if pipeline_entry else role_title) or "Unknown Role"
        entry_id = pipeline_entry.id if pipeline_entry else UUID(int=0)

        self.logger.info(f"Preparing interview for {company} — {role}")

        prep = self.prepare(
            company_name=company,
            role_title=role,
            job_description=job_description,
            pipeline_entry_id=entry_id,
        )

        summary = (
            f"Interview prep ready for {company}. "
            f"{len(prep.likely_questions)} questions generated. "
            f"{len(prep.star_examples)} STAR examples prepared."
        )
        self.logger.success(summary)
        return {"status": "ok", "prep": prep, "summary": summary}

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        company_name: str,
        role_title: str,
        job_description: str = "",
        pipeline_entry_id: Optional[UUID] = None,
    ) -> InterviewPrep:
        """
        Generate a complete interview preparation package.

        Components:
        - Company research brief
        - 10 likely interview questions (technical + behavioral)
        - 3 STAR example templates
        - 5 smart questions to ask the interviewer
        - Red flags to watch for

        Args:
            company_name:       Target company
            role_title:         Target role
            job_description:    Raw JD text
            pipeline_entry_id:  Link back to pipeline entry

        Returns:
            InterviewPrep domain object
        """
        from uuid import uuid4
        company_research = self._research_company(company_name)
        questions = self._generate_questions(company_name, role_title, job_description)
        star_examples = self._generate_star_examples(role_title)
        questions_to_ask = self._generate_questions_to_ask(company_name, role_title)
        red_flags = self._identify_red_flags(company_name, job_description)

        return InterviewPrep(
            pipeline_entry_id=pipeline_entry_id or uuid4(),
            company_research=company_research,
            likely_questions=questions,
            star_examples=star_examples,
            questions_to_ask=questions_to_ask,
            red_flags=red_flags,
        )

    def _research_company(self, company_name: str) -> str:
        """
        Generate a company research brief covering:
        - Product and business model
        - Recent news / funding
        - Culture and values
        - Competitors

        Args:
            company_name: Company to research

        Returns:
            Structured research brief as a string
        """
        prompt = (
            f"Create a concise company research brief (max 300 words) for a job interview at {company_name}.\n"
            f"Structure:\n"
            f"1. What they do (product/service, customers)\n"
            f"2. Business model\n"
            f"3. Recent news or funding (if known)\n"
            f"4. Company culture signals\n"
            f"5. Main competitors\n"
            f"6. Why this company stands out\n\n"
            f"Be factual. If unknown, say 'research needed' for that section."
        )
        return self._llm.complete(prompt, max_tokens=500)

    def _generate_questions(
        self, company_name: str, role_title: str, job_description: str
    ) -> List[str]:
        """
        Generate 10 likely interview questions (mix of technical and behavioral).

        Args:
            company_name:    Target company
            role_title:      Target role title
            job_description: Raw JD text

        Returns:
            List of question strings
        """
        prompt = (
            f"Generate 10 likely interview questions for this role:\n"
            f"Company: {company_name}\n"
            f"Role: {role_title}\n"
            f"{'JD snippet: ' + job_description[:500] if job_description else ''}\n\n"
            f"Include: 4 technical questions, 3 behavioral (STAR), 2 situational, 1 culture-fit.\n"
            f"Format as a numbered list. Questions only — no answers."
        )
        raw = self._llm.complete(prompt, max_tokens=500)
        return self._parse_numbered_list(raw)

    def _generate_star_examples(self, role_title: str) -> List[str]:
        """
        Generate 3 STAR story templates tailored to the role.
        Templates use placeholders that the candidate fills with real experiences.

        Args:
            role_title: The target role

        Returns:
            List of STAR template strings
        """
        prompt = (
            f"Generate 3 STAR (Situation, Task, Action, Result) story templates "
            f"for a {role_title} interview. "
            f"Each template should have clear [PLACEHOLDER] sections the candidate fills in. "
            f"Focus on: technical achievement, leadership/collaboration, and handling a crisis. "
            f"Format as: STORY 1: ... STORY 2: ... STORY 3: ..."
        )
        raw = self._llm.complete(prompt, max_tokens=800)
        # Split on "STORY N:" pattern
        import re
        parts = re.split(r"STORY\s+\d+:", raw, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()][:3]

    def _generate_questions_to_ask(
        self, company_name: str, role_title: str
    ) -> List[str]:
        """
        Generate 5 smart questions for the candidate to ask the interviewer.
        Asking good questions signals preparation and genuine interest.

        Args:
            company_name: Target company
            role_title:   Target role

        Returns:
            List of question strings
        """
        prompt = (
            f"Generate 5 smart, specific questions a {role_title} candidate should ask "
            f"the interviewer at {company_name}. "
            f"Questions should demonstrate research and genuine curiosity. "
            f"Avoid generic questions like 'What does success look like?'. "
            f"Format as a numbered list."
        )
        raw = self._llm.complete(prompt, max_tokens=300)
        return self._parse_numbered_list(raw)

    def _identify_red_flags(
        self, company_name: str, job_description: str
    ) -> List[str]:
        """
        Identify potential red flags to watch for during the interview.
        Protects candidate from making a bad career move.

        Args:
            company_name:    Company name
            job_description: Raw JD text

        Returns:
            List of red flag strings
        """
        prompt = (
            f"Identify up to 5 potential red flags to watch for when interviewing at {company_name}.\n"
            f"{'JD: ' + job_description[:500] if job_description else ''}\n\n"
            f"Consider: vague role scope, high turnover signals, unrealistic expectations, "
            f"equity/compensation red flags, culture mismatch signals. "
            f"Be specific and practical. Format as a numbered list."
        )
        raw = self._llm.complete(prompt, max_tokens=300)
        return self._parse_numbered_list(raw)

    # ------------------------------------------------------------------
    # Post-interview debrief
    # ------------------------------------------------------------------

    def debrief(
        self,
        outcome: str,
        interview_notes: str,
        rejection_reason: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Analyze interview notes post-interview to extract lessons learned.
        Applies regardless of outcome (rejection or advance).

        Args:
            outcome:          "rejected" | "advanced" | "offer"
            interview_notes:  Candidate's notes from the interview
            rejection_reason: Feedback from recruiter (if provided)

        Returns:
            Dictionary with "strengths", "areas_to_improve", "action_plan"
        """
        self.logger.info(f"Running post-interview debrief — outcome: {outcome}")

        prompt = (
            f"Analyze this interview performance and provide coaching feedback.\n\n"
            f"Outcome: {outcome}\n"
            f"{'Rejection reason: ' + rejection_reason if rejection_reason else ''}\n\n"
            f"Interview notes:\n{interview_notes[:1500]}\n\n"
            f"Provide:\n"
            f"STRENGTHS: (3 things that went well)\n"
            f"AREAS TO IMPROVE: (3 specific things to work on)\n"
            f"ACTION PLAN: (3 concrete steps for the next interview)\n\n"
            f"Be direct and constructive."
        )
        raw = self._llm.complete(prompt, max_tokens=500)

        # Parse sections
        import re
        sections = {}
        for section in ["STRENGTHS", "AREAS TO IMPROVE", "ACTION PLAN"]:
            pattern = rf"{section}:\s*(.*?)(?=(?:STRENGTHS|AREAS TO IMPROVE|ACTION PLAN):|$)"
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            sections[section.lower().replace(" ", "_")] = match.group(1).strip() if match else ""

        return sections

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        """Parse a numbered list from LLM output into a clean Python list."""
        import re
        lines = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip() and re.match(r"^\d+", line.strip())
        ]
        return lines
