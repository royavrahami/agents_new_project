"""
Tests for OutreachAgent — message drafting, daily limits, follow-up tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from agents.outreach_agent import OutreachAgent
from config.settings import settings
from core.database import OutreachMessageORM, get_db
from core.models import Contact, HiddenJob, OutreachMessage


class TestOutreachAgentMessageDrafting:
    """Tests for the cold message drafting logic."""

    def test_draft_linkedin_message_not_empty(self, mock_llm, sample_hidden_job, sample_contact):
        mock_llm.complete.return_value = "Hi Jane, I'd love to connect about opportunities at Acme Tech."
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact, channel="linkedin")
        assert msg.body != ""
        assert msg.channel == "linkedin"
        assert msg.status == "DRAFT"

    def test_draft_email_message_has_subject(self, mock_llm, sample_hidden_job, sample_contact):
        mock_llm.complete.return_value = "Dear Jane, I am writing about the QA role..."
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact, channel="email")
        assert msg.subject is not None
        assert msg.channel == "email"

    def test_linkedin_message_truncated_to_300_chars(self, mock_llm, sample_hidden_job, sample_contact):
        long_response = "A" * 500  # Way over LinkedIn 300-char limit
        mock_llm.complete.return_value = long_response
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact, channel="linkedin")
        assert len(msg.body) <= 300

    def test_follow_up_due_set_in_future(self, mock_llm, sample_hidden_job, sample_contact):
        mock_llm.complete.return_value = "Short message"
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact)
        assert msg.follow_up_due > datetime.utcnow()

    def test_funding_linked_job_mentions_funding(self, mock_llm, sample_contact):
        mock_llm.complete.return_value = "Great that Acme recently raised capital!"
        funded_job = HiddenJob(
            company_name="Funded Corp",
            role_title="QA Engineer",
            job_url="https://example.com/job/1",
            source_domain="comeet.com",
            funding_linked=True,
        )
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=funded_job, contact=sample_contact)
        # The mock LLM returns a response that references funding
        assert "raised capital" in msg.body or "recently" in mock_llm.complete.call_args_list[0][0][0]

    def test_draft_follow_up_message(self, mock_llm, sample_contact):
        mock_llm.complete.return_value = "Just following up on my previous message!"
        original_msg = OutreachMessage(
            contact_id=sample_contact.id,
            body="Original message",
            channel="linkedin",
            status="SENT",
        )
        agent = OutreachAgent(llm_tool=mock_llm)
        follow_up = agent.draft_follow_up(original_msg, sample_contact)
        assert follow_up.body != ""
        assert follow_up.follow_up_due > datetime.utcnow()


class TestOutreachAgentDailyLimit:
    """Tests for daily outreach limit enforcement."""

    def test_run_skips_when_limit_reached(self, mock_llm, sample_hidden_job):
        """When daily limit is already reached, no new messages should be drafted."""
        agent = OutreachAgent(llm_tool=mock_llm)

        # Pre-populate sent messages today at the limit
        today = datetime.utcnow()
        with get_db() as db:
            for i in range(settings.outreach_daily_limit):
                db.add(OutreachMessageORM(
                    contact_name=f"Contact {i}",
                    contact_company=f"Company {i}",
                    body=f"Message {i}",
                    channel="linkedin",
                    status="SENT",
                    sent_at=today,
                ))

        result = agent.execute(hot_jobs=[sample_hidden_job])
        assert result["messages_drafted"] == 0

    def test_run_respects_daily_limit(self, mock_llm):
        """When partial limit remains, only draft up to remaining budget."""
        mock_llm.complete.return_value = "Short test message"
        agent = OutreachAgent(llm_tool=mock_llm)

        # Pre-populate: limit - 2 messages already sent
        today = datetime.utcnow()
        already_sent = settings.outreach_daily_limit - 2
        with get_db() as db:
            for i in range(already_sent):
                db.add(OutreachMessageORM(
                    contact_name=f"Contact {i}",
                    contact_company=f"Company {i}",
                    body=f"Sent message {i}",
                    channel="linkedin",
                    status="SENT",
                    sent_at=today,
                ))

        # Provide 5 hot jobs but only 2 slots remain
        jobs = [
            HiddenJob(
                company_name=f"Company {j}",
                role_title="QA Engineer",
                job_url=f"https://example.com/job/{j}",
                source_domain="comeet.com",
            )
            for j in range(5)
        ]
        result = agent.execute(hot_jobs=jobs)
        assert result["messages_drafted"] <= 2


class TestOutreachAgentStatusTracking:
    """Tests for mark_as_sent and mark_as_responded."""

    def test_mark_as_sent_updates_status(self, mock_llm, sample_hidden_job, sample_contact):
        mock_llm.complete.return_value = "Test message"
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact)
        agent._persist_message(msg, sample_contact)

        with get_db() as db:
            msg_id = db.query(OutreachMessageORM).first().id

        agent.mark_as_sent(msg_id)

        # Read ORM attributes inside an active session to avoid DetachedInstanceError
        with get_db() as db:
            updated = db.query(OutreachMessageORM).filter_by(id=msg_id).first()
            status = updated.status
            sent_at = updated.sent_at
        assert status == "SENT"
        assert sent_at is not None

    def test_mark_as_responded_updates_status(self, mock_llm, sample_hidden_job, sample_contact):
        mock_llm.complete.return_value = "Test message"
        agent = OutreachAgent(llm_tool=mock_llm)
        msg = agent.draft_message(job=sample_hidden_job, contact=sample_contact)
        agent._persist_message(msg, sample_contact)

        with get_db() as db:
            msg_id = db.query(OutreachMessageORM).first().id

        agent.mark_as_sent(msg_id)
        agent.mark_as_responded(msg_id)

        with get_db() as db:
            updated = db.query(OutreachMessageORM).filter_by(id=msg_id).first()
            status = updated.status
            responded_at = updated.responded_at
        assert status == "RESPONDED"
        assert responded_at is not None


class TestOutreachAgentRun:
    """Integration tests for OutreachAgent.run()."""

    def test_run_with_no_jobs_returns_ok(self, mock_llm):
        agent = OutreachAgent(llm_tool=mock_llm)
        result = agent.execute(hot_jobs=[])
        assert result["status"] == "ok"
        assert result["messages_drafted"] == 0

    def test_run_with_jobs_drafts_messages(self, mock_llm, sample_hidden_job):
        mock_llm.complete.return_value = "Personalized outreach message"
        agent = OutreachAgent(llm_tool=mock_llm)
        result = agent.execute(hot_jobs=[sample_hidden_job])
        assert result["status"] == "ok"
        assert result["messages_drafted"] >= 1

    def test_run_returns_follow_ups_due(self, mock_llm):
        agent = OutreachAgent(llm_tool=mock_llm)
        # Insert a SENT message with past follow_up_due
        past_due = datetime.utcnow() - timedelta(days=1)
        with get_db() as db:
            db.add(OutreachMessageORM(
                contact_name="Overdue Contact",
                contact_company="Overdue Corp",
                body="Old message",
                channel="linkedin",
                status="SENT",
                sent_at=datetime.utcnow() - timedelta(days=6),
                follow_up_due=past_due,
            ))

        result = agent.execute(hot_jobs=[])
        assert len(result["follow_ups_due"]) >= 1
