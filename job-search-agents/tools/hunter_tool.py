"""
Hunter.io Email Discovery Tool.

Integrates with Hunter.io API to find email addresses and contacts at companies.
Used by OutreachAgent to discover recruiter emails automatically.

Responsibilities:
1. Find emails for specific people (first name, last name, company domain)
2. Find all emails at a company domain
3. Search for people by title and company (recruiters, hiring managers)
4. Validate email addresses
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import httpx
from config.settings import settings
from core.logger import logger


class HunterTool:
    """
    Hunter.io email discovery API client.

    Hunter.io provides:
    - Email finder: find emails by name + domain
    - Domain search: find all emails at a domain
    - Email verifier: validate if email exists
    - Lead lists: search by job title, company

    Args:
        api_key: Hunter.io API key (from settings if not provided)
        timeout: HTTP timeout in seconds
        rate_limit_delay: Seconds to wait between requests (free tier: 1 req/sec)
    """

    API_ENDPOINT = "https://api.hunter.io/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
        rate_limit_delay: float = 0.1,  # 10 requests per second (conservative)
    ) -> None:
        self.api_key = api_key or settings.hunter_api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.logger = logger
        self._last_request_time = 0

    def find_email(
        self,
        first_name: str,
        last_name: str,
        domain: str,
        company: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find email address for a specific person.

        Args:
            first_name: Person's first name
            last_name: Person's last name
            domain: Company domain (e.g., "google.com")
            company: Optional company name for context

        Returns:
            {
                "email": str,
                "confidence": float (0-100),
                "sources": int,
                "verified": bool,
            } or None if not found
        """
        if not self.api_key:
            self.logger.warning("Hunter.io API key not configured")
            return None

        try:
            params = {
                "domain": domain,
                "first_name": first_name,
                "last_name": last_name,
            }
            if company:
                params["company"] = company

            response = self._make_request(f"{self.API_ENDPOINT}/email-finder", params=params)

            if response.get("data"):
                email_data = response["data"]
                return {
                    "email": email_data.get("email"),
                    "confidence": email_data.get("confidence", 0),
                    "sources": email_data.get("sources", 0),
                    "verified": email_data.get("verification", {}).get("status") == "valid",
                    "position": email_data.get("position"),
                    "seniority": email_data.get("seniority"),
                    "linkedin_url": email_data.get("linkedin_url"),
                    "twitter_url": email_data.get("twitter_url"),
                    "phone_number": email_data.get("phone_number"),
                }

            return None
        except Exception as e:
            self.logger.error(f"Error finding email: {e}")
            return None

    def search_domain_emails(
        self,
        domain: str,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all emails at a company domain with optional filtering.

        Args:
            domain: Company domain (e.g., "google.com")
            limit: Maximum emails to return (1-100)
            filters: Optional filters:
                - position: Job title to filter by (e.g., "recruiter")
                - seniority: "junior", "mid", "senior", "executive"
                - department: "sales", "hr", "engineering", etc.

        Returns:
            List of contact dictionaries
        """
        if not self.api_key:
            return []

        try:
            limit = min(limit, 100)  # Hunter.io max per request

            params = {
                "domain": domain,
                "limit": limit,
            }

            if filters:
                if "position" in filters:
                    params["position"] = filters["position"]
                if "seniority" in filters:
                    params["seniority"] = filters["seniority"]
                if "department" in filters:
                    params["department"] = filters["department"]

            response = self._make_request(f"{self.API_ENDPOINT}/domain-search", params=params)

            if response.get("data"):
                emails = []
                for person in response["data"].get("emails", []):
                    emails.append({
                        "email": person.get("email"),
                        "first_name": person.get("first_name"),
                        "last_name": person.get("last_name"),
                        "position": person.get("position"),
                        "seniority": person.get("seniority"),
                        "department": person.get("department"),
                        "confidence": person.get("confidence", 0),
                        "verified": person.get("verification", {}).get("status") == "valid",
                        "linkedin_url": person.get("linkedin_url"),
                        "phone_number": person.get("phone_number"),
                    })

                return emails

            return []
        except Exception as e:
            self.logger.error(f"Error searching domain: {e}")
            return []

    def search_recruiters(
        self,
        domain: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find recruiters at a company (convenience method).

        Args:
            domain: Company domain
            limit: Maximum recruiters to return

        Returns:
            List of recruiter contacts sorted by seniority
        """
        emails = self.search_domain_emails(
            domain=domain,
            limit=limit,
            filters={"position": "recruiter"},
        )

        # If no results with "recruiter" filter, broaden search to HR dept
        if not emails:
            emails = self.search_domain_emails(
                domain=domain,
                limit=limit,
                filters={"department": "hr"},
            )

        # Sort by seniority (executive > senior > mid > junior)
        seniority_order = {"executive": 0, "senior": 1, "mid": 2, "junior": 3}
        emails.sort(
            key=lambda x: (
                seniority_order.get(x.get("seniority", "").lower(), 4),
                -x.get("confidence", 0),  # Higher confidence first
            )
        )

        return emails[:limit]

    def search_hiring_managers(
        self,
        domain: str,
        department: str = "engineering",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find hiring managers at a company in a specific department.

        Args:
            domain: Company domain
            department: Department name (engineering, product, etc.)
            limit: Maximum results

        Returns:
            List of hiring manager contacts
        """
        if not self.api_key:
            return []

        try:
            # Search for manager/director titles in the department
            emails = self.search_domain_emails(
                domain=domain,
                limit=limit * 2,
                filters={"department": department},
            )

            # Filter for manager/director/head titles
            manager_keywords = ["manager", "director", "head", "lead", "principal", "architect"]
            managers = []

            for person in emails:
                position = (person.get("position") or "").lower()
                if any(keyword in position for keyword in manager_keywords):
                    managers.append(person)

            return managers[:limit]
        except Exception as e:
            self.logger.error(f"Error searching hiring managers: {e}")
            return []

    def verify_email(self, email: str) -> Dict[str, Any]:
        """
        Verify if an email address is valid and currently in use.

        Args:
            email: Email address to verify

        Returns:
            {
                "email": str,
                "status": "valid" | "invalid" | "accept_all" | "unknown",
                "score": float (0-100),
                "smtp_check": bool,
                "smtp_error": str or None,
            }
        """
        if not self.api_key:
            return {"email": email, "status": "unknown"}

        try:
            params = {"email": email}
            response = self._make_request(f"{self.API_ENDPOINT}/email-verifier", params=params)

            if response.get("data"):
                verification = response["data"]
                return {
                    "email": email,
                    "status": verification.get("status"),
                    "score": verification.get("score", 0),
                    "smtp_check": verification.get("smtp_check", False),
                    "smtp_error": verification.get("smtp_error"),
                    "gibberish": verification.get("gibberish", False),
                    "disposable": verification.get("disposable", False),
                    "webmail": verification.get("webmail", False),
                }

            return {"email": email, "status": "unknown"}
        except Exception as e:
            self.logger.error(f"Error verifying email: {e}")
            return {"email": email, "status": "unknown"}

    def search_leads(
        self,
        domain: str,
        job_titles: Optional[List[str]] = None,
        seniority_levels: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for leads at a company by job title and seniority.

        This is a higher-level search combining multiple filters.

        Args:
            domain: Company domain
            job_titles: List of job titles to search for (e.g., ["recruiter", "talent"])
            seniority_levels: List of seniority levels (e.g., ["senior", "executive"])
            limit: Maximum results

        Returns:
            List of contacts matching criteria
        """
        if not self.api_key:
            return []

        # Default search criteria
        if not job_titles:
            job_titles = ["recruiter", "hr", "talent", "hiring"]
        if not seniority_levels:
            seniority_levels = ["executive", "senior", "mid"]

        all_contacts = []

        try:
            # Search by each job title
            for title in job_titles:
                emails = self.search_domain_emails(
                    domain=domain,
                    limit=min(limit, 100),
                    filters={"position": title},
                )

                # Filter by seniority if specified
                if seniority_levels:
                    emails = [
                        e for e in emails
                        if e.get("seniority", "").lower() in seniority_levels
                    ]

                all_contacts.extend(emails)

            # Deduplicate and sort by relevance
            seen_emails = set()
            unique_contacts = []

            for contact in sorted(
                all_contacts,
                key=lambda x: (
                    -x.get("confidence", 0),
                    {"executive": 0, "senior": 1, "mid": 2}.get(
                        x.get("seniority", "").lower(), 3
                    ),
                ),
            ):
                email = contact.get("email")
                if email and email not in seen_emails:
                    seen_emails.add(email)
                    unique_contacts.append(contact)

            return unique_contacts[:limit]
        except Exception as e:
            self.logger.error(f"Error searching leads: {e}")
            return []

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to Hunter.io API with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        try:
            if params is None:
                params = {}

            params["domain"] = params.get("domain", "")
            params["api_key"] = self.api_key

            with httpx.Client(timeout=timeout or self.timeout) as client:
                response = client.get(url, params=params)
                self._last_request_time = time.time()

                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error calling Hunter.io: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error making request to Hunter.io: {e}")
            raise

    @staticmethod
    def extract_domain_from_email(email: str) -> str:
        """Extract domain from email address."""
        if "@" in email:
            return email.split("@")[1]
        return ""

    @staticmethod
    def extract_domain_from_url(url: str) -> str:
        """Extract domain from URL."""
        url = url.replace("http://", "").replace("https://", "").replace("www.", "")
        return url.split("/")[0]


# Singleton instance
hunter = HunterTool()
