"""
Outreach Agent — Cold Message Generator & Follow-Up Tracker.

Responsibilities:
1. Find hiring managers and recruiters at target companies via LinkedIn search
2. Generate personalized cold outreach messages (LinkedIn DM / email)
3. Track sent messages and schedule follow-ups
4. Enforce daily outreach limits to avoid spam flags
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from config.settings import settings
from core.database import OutreachMessageORM, get_db
from core.logger import logger
from core.models import Contact, HiddenJob, OutreachMessage
from tools.llm_tool import LLMTool

from .base_agent import BaseAgent


# Message templates — the LLM personalizes these, not replaces them
_LINKEDIN_MESSAGE_TEMPLATE = """
Hi {contact_name},

I noticed {company_name} {funding_note}is doing impressive work in {domain}.
I'm a {candidate_role} with {experience_years}+ years of experience building QA organizations from scratch,
leading quality architecture, and driving release governance across FinTech, Deep-Tech, and Enterprise environments.

{personal_hook}

Would you be open to a 15-minute conversation?

Best,
{candidate_name}
""".strip()

_EMAIL_TEMPLATE = """
Subject: {role} Opening — {candidate_name}

Hi {contact_name},

{personal_hook}

I'm reaching out because {company_name} {funding_note}stands out in the Israeli tech scene.
As a {candidate_role} with {experience_years}+ years of experience — including building QA departments
from zero, implementing CI/CD quality gates, and reducing production escape rates by 35% —
I believe I could bring real impact to your engineering organization.

I'd love to schedule a quick 15-minute call to explore if there's a fit.

Best regards,
{candidate_name}
""".strip()


class OutreachAgent(BaseAgent):
    """
    Manages the cold-outreach pipeline: finds contacts, drafts messages,
    tracks sends, and schedules follow-ups.

    Args:
        llm_tool: LLMTool instance (injected for testability)
    """

    def __init__(self, llm_tool: Optional[LLMTool] = None) -> None:
        super().__init__(name="OutreachAgent")
        self._llm = llm_tool or LLMTool(
            system_prompt=(
                "You are an expert at writing highly personalized, concise, and compelling "
                "job search outreach messages for the Israeli hi-tech market. "
                "Messages must be warm, professional, and never generic."
            )
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        hot_jobs: Optional[List[HiddenJob]] = None,
        contacts: Optional[List[Contact]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate and queue outreach messages for the top hot jobs.

        Args:
            hot_jobs: List of HiddenJob from Intel Agent (sorted by hot_score)
            contacts: Pre-found contacts; if None, uses mock contacts for demo

        Returns:
            {
                "status": "ok",
                "messages_drafted": int,
                "messages_ready": List[OutreachMessage],
                "follow_ups_due": List[OutreachMessage],
                "summary": str
            }
        """
        jobs = hot_jobs or []
        if not jobs:
            self.logger.warning("No hot jobs provided — outreach drafting skipped")
            follow_ups = self._get_follow_ups_due()
            return {
                "status": "ok",
                "messages_drafted": 0,
                "messages_ready": [],
                "follow_ups_due": follow_ups,
                "summary": f"No hot jobs to outreach for. {len(follow_ups)} follow-ups due.",
            }

        # Check today's send limit
        already_sent_today = self._count_sent_today()
        remaining_budget = settings.outreach_daily_limit - already_sent_today

        if remaining_budget <= 0:
            self.logger.warning(
                f"Daily outreach limit reached ({settings.outreach_daily_limit}). Skipping."
            )
            return {
                "status": "ok",
                "messages_drafted": 0,
                "messages_ready": [],
                "follow_ups_due": self._get_follow_ups_due(),
                "summary": f"Daily limit of {settings.outreach_daily_limit} reached.",
            }

        # Draft messages for hot jobs (up to daily limit)
        drafted: List[OutreachMessage] = []
        for job in jobs[:remaining_budget]:
            contact = self._find_or_create_contact(job, contacts)
            msg = self.draft_message(job=job, contact=contact)
            self._persist_message(msg, contact)
            drafted.append(msg)

        # Check follow-ups due
        follow_ups = self._get_follow_ups_due()

        summary = (
            f"Outreach cycle complete. "
            f"Drafted {len(drafted)} messages. "
            f"{len(follow_ups)} follow-ups due today."
        )
        self.logger.success(summary)
        return {
            "status": "ok",
            "messages_drafted": len(drafted),
            "messages_ready": drafted,
            "follow_ups_due": follow_ups,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Message drafting
    # ------------------------------------------------------------------

    def draft_message(
        self,
        job: HiddenJob,
        contact: Contact,
        channel: str = "linkedin",
    ) -> OutreachMessage:
        """
        Generate a personalized outreach message for a specific job and contact.

        The LLM personalizes the message using:
        - Company context (funding, domain, product)
        - Contact's role (recruiter vs hiring manager)
        - Candidate's background

        Args:
            job:     The target HiddenJob opportunity
            contact: The Contact to message
            channel: "linkedin" or "email"

        Returns:
            OutreachMessage ready to send
        """
        self.logger.info(
            f"Drafting {channel} message for {contact.company_name} → {contact.full_name or 'Unknown'}"
        )

        funding_note = "recently raised capital and " if job.funding_linked else ""
        template = _LINKEDIN_MESSAGE_TEMPLATE if channel == "linkedin" else _EMAIL_TEMPLATE

        # Build personalization context for LLM
        context = {
            "contact_name": contact.full_name or "there",
            "company_name": contact.company_name,
            "funding_note": funding_note,
            "domain": self._infer_domain(job),
            "candidate_role": settings.candidate_role,
            "role": settings.candidate_role,          # alias used by email template
            "experience_years": settings.candidate_experience_years,
            "candidate_name": settings.candidate_name,
            "personal_hook": "",  # Filled by LLM below
        }

        # Generate a personalized hook via LLM
        hook_prompt = (
            f"Write ONE sentence (max 20 words) as a personal hook for a cold outreach message. "
            f"Company: {contact.company_name}. Job: {job.role_title}. "
            f"Candidate role: {settings.candidate_role}. "
            f"Context: {'recently raised funding' if job.funding_linked else 'hiring now'}. "
            f"Be specific and genuine. Return ONLY the sentence."
        )
        context["personal_hook"] = self._llm.complete(hook_prompt, max_tokens=60).strip()

        # Fill template
        raw_body = template.format(**context)

        # LLM polish — improve tone and conciseness
        polish_prompt = (
            f"Polish this outreach message. Make it warmer, more concise (max 100 words), "
            f"and specific to the {'LinkedIn DM' if channel == 'linkedin' else 'email'} format. "
            f"Do NOT change the structure. Return ONLY the improved message.\n\n{raw_body}"
        )
        body = self._llm.complete(polish_prompt, max_tokens=300).strip() or raw_body

        # Truncate LinkedIn messages to 300 chars (platform limit)
        if channel == "linkedin" and len(body) > 300:
            body = body[:297] + "..."

        return OutreachMessage(
            contact_id=contact.id,
            job_id=job.id,
            subject=f"{settings.candidate_role} — {settings.candidate_name}" if channel == "email" else None,
            body=body,
            channel=channel,
            follow_up_due=datetime.utcnow() + timedelta(days=settings.follow_up_days),
            status="DRAFT",
        )

    def draft_follow_up(self, original_message: OutreachMessage, contact: Contact) -> OutreachMessage:
        """
        Generate a follow-up message for a previously sent, unanswered outreach.

        Args:
            original_message: The original OutreachMessage that was sent
            contact:          The Contact who didn't respond

        Returns:
            New OutreachMessage as a follow-up
        """
        prompt = (
            f"Write a short, friendly follow-up message (max 60 words) for this context:\n"
            f"- Original message was sent {settings.follow_up_days} days ago\n"
            f"- No response yet\n"
            f"- Contact: {contact.full_name or 'the recipient'} at {contact.company_name}\n"
            f"- Channel: {original_message.channel}\n"
            f"- Original message:\n{original_message.body}\n\n"
            f"Keep it warm, not pushy. Return ONLY the follow-up text."
        )
        body = self._llm.complete(prompt, max_tokens=150).strip()

        return OutreachMessage(
            contact_id=contact.id,
            job_id=original_message.job_id,
            body=body,
            channel=original_message.channel,
            follow_up_due=datetime.utcnow() + timedelta(days=settings.follow_up_days * 2),
            status="DRAFT",
        )

    # ------------------------------------------------------------------
    # Contact resolution
    # ------------------------------------------------------------------

    def _find_or_create_contact(
        self, job: HiddenJob, provided_contacts: Optional[List[Contact]]
    ) -> Contact:
        """
        Find a matching contact for a job from the provided list,
        or create a placeholder contact for the company.

        In production this would call LinkedIn API / Apollo.io.

        Args:
            job:               The target job
            provided_contacts: List of pre-resolved contacts

        Returns:
            Best matching Contact
        """
        if provided_contacts:
            # Find first contact matching the company
            for contact in provided_contacts:
                if contact.company_name.lower() == job.company_name.lower():
                    return contact

        # Fallback — create a generic company contact
        return Contact(
            company_name=job.company_name,
            full_name=None,
            title="Hiring Manager",
            is_hiring_manager=True,
        )

    @staticmethod
    def _infer_domain(job: HiddenJob) -> str:
        """Infer the company's domain from job description keywords."""
        text = f"{job.role_title} {job.description_snippet or ''}".lower()
        domain_map = {
            "fintech": ["fintech", "payment", "banking", "finance"],
            "cybersecurity": ["security", "cyber", "soc", "infosec"],
            "devtools": ["developer", "devops", "platform", "infrastructure"],
            "healthtech": ["health", "medical", "clinical", "pharma"],
            "saas": ["saas", "b2b", "enterprise", "platform"],
        }
        for domain, keywords in domain_map.items():
            if any(kw in text for kw in keywords):
                return domain
        return "tech"

    # ------------------------------------------------------------------
    # Persistence & Tracking
    # ------------------------------------------------------------------

    def _persist_message(self, msg: OutreachMessage, contact: Contact) -> None:
        """Save a drafted message to the database."""
        with get_db() as db:
            orm = OutreachMessageORM(
                id=str(msg.id),
                contact_name=contact.full_name,
                contact_company=contact.company_name,
                contact_linkedin=contact.linkedin_url,
                contact_email=contact.email,
                job_id=str(msg.job_id) if msg.job_id else None,
                subject=msg.subject,
                body=msg.body,
                channel=msg.channel,
                follow_up_due=msg.follow_up_due,
                status=msg.status,
            )
            db.add(orm)

    def _count_sent_today(self) -> int:
        """Count messages with status SENT that were sent today."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with get_db() as db:
            return (
                db.query(OutreachMessageORM)
                .filter(
                    OutreachMessageORM.status == "SENT",
                    OutreachMessageORM.sent_at >= today_start,
                )
                .count()
            )

    def _get_follow_ups_due(self) -> List[OutreachMessage]:
        """Retrieve messages where follow_up_due has passed and status is SENT."""
        now = datetime.utcnow()
        with get_db() as db:
            rows = (
                db.query(OutreachMessageORM)
                .filter(
                    OutreachMessageORM.status == "SENT",
                    OutreachMessageORM.follow_up_due <= now,
                )
                .all()
            )
        return [
            OutreachMessage(
                contact_id=uuid4(),  # Placeholder — production would join Contact table
                body=r.body or "",
                channel=r.channel or "linkedin",
                status=r.status or "SENT",
            )
            for r in rows
        ]

    def mark_as_sent(self, message_id: str) -> None:
        """
        Update a message's status to SENT after it has been dispatched.

        Args:
            message_id: UUID string of the OutreachMessage
        """
        with get_db() as db:
            msg = db.query(OutreachMessageORM).filter_by(id=message_id).first()
            if msg:
                msg.status = "SENT"
                msg.sent_at = datetime.utcnow()
                self.logger.info(f"Message {message_id} marked as SENT")

    def mark_as_responded(self, message_id: str) -> None:
        """
        Update a message's status to RESPONDED.

        Args:
            message_id: UUID string of the OutreachMessage
        """
        with get_db() as db:
            msg = db.query(OutreachMessageORM).filter_by(id=message_id).first()
            if msg:
                msg.status = "RESPONDED"
                msg.responded_at = datetime.utcnow()
                self.logger.info(f"Message {message_id} marked as RESPONDED")
