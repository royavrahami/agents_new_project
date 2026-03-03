"""
Comeet Job Board API Integration.

Direct integration with Comeet API for job discovery.
Comeet is the primary Israeli recruiting platform for tech jobs.

Responsibilities:
1. Search jobs by keywords and location
2. Get detailed job information
3. Get company profiles and information
4. Track job posting dates
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from config.settings import settings
from core.logger import logger


class ComeetTool:
    """
    Comeet job board API client.

    Provides access to Comeet's job listings and company data.
    Comeet is Israel's largest recruiting platform for tech roles.

    Args:
        api_key: Comeet API key (from settings if not provided)
        timeout: HTTP timeout in seconds
    """

    API_ENDPOINT = "https://api.comeet.co/api/v4"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10) -> None:
        self.api_key = api_key or settings.comeet_api_key
        self.timeout = timeout
        self.logger = logger

    def search_jobs(
        self,
        keywords: Optional[List[str]] = None,
        location: Optional[str] = None,
        company_ids: Optional[List[str]] = None,
        job_types: Optional[List[str]] = None,
        seniority: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for jobs on Comeet.

        Args:
            keywords: Job keywords to search (e.g., ["QA", "Manager"])
            location: Location filter (e.g., "Tel Aviv", "Israel")
            company_ids: Optional list of company IDs to filter
            job_types: Job types (e.g., ["Full Time", "Part Time"])
            seniority: Seniority levels (e.g., ["Senior", "Manager"])
            limit: Maximum jobs to return

        Returns:
            List of job dictionaries with basic info
        """
        if not self.api_key:
            self.logger.warning("Comeet API key not configured, returning empty results")
            return []

        try:
            # Build search query
            query_parts = []
            if keywords:
                query_parts.extend(keywords)
            if location:
                query_parts.append(location)

            query = " ".join(query_parts) if query_parts else "*"

            params = {
                "q": query,
                "limit": min(limit, 200),
            }

            # Add optional filters
            if company_ids:
                params["company_ids"] = ",".join(company_ids)
            if job_types:
                params["job_types"] = ",".join(job_types)
            if seniority:
                params["seniority"] = ",".join(seniority)

            # Make request
            response = self._make_request("/jobs/search", params=params)

            if response.get("jobs"):
                jobs = []
                for job in response["jobs"]:
                    job_dict = {
                        "id": job.get("id"),
                        "title": job.get("title"),
                        "company_id": job.get("company", {}).get("id"),
                        "company_name": job.get("company", {}).get("name"),
                        "location": job.get("location", {}).get("name"),
                        "description": job.get("description", "")[:500],  # First 500 chars
                        "url": job.get("url"),
                        "posted_date": job.get("created_at"),
                        "job_type": job.get("employment_type"),
                        "seniority": job.get("seniority_level"),
                        "keywords": self._extract_keywords_from_job(job),
                    }

                    # Calculate freshness score (newer = higher score)
                    posted = self._parse_date(job.get("created_at"))
                    days_old = (datetime.now() - posted).days if posted else 999
                    job_dict["days_posted"] = max(0, days_old)
                    job_dict["freshness_score"] = max(0.0, 1.0 - (days_old / 30.0))

                    jobs.append(job_dict)

                return jobs

            return []
        except Exception as e:
            self.logger.error(f"Error searching jobs: {e}")
            return []

    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific job.

        Args:
            job_id: Comeet job ID

        Returns:
            Detailed job dictionary or None
        """
        if not self.api_key:
            return None

        try:
            response = self._make_request(f"/jobs/{job_id}")

            if response.get("job"):
                job = response["job"]
                return {
                    "id": job.get("id"),
                    "title": job.get("title"),
                    "company_id": job.get("company", {}).get("id"),
                    "company_name": job.get("company", {}).get("name"),
                    "company_logo": job.get("company", {}).get("logo_url"),
                    "location": job.get("location", {}).get("name"),
                    "description": job.get("description"),
                    "requirements": job.get("requirements"),
                    "benefits": job.get("benefits"),
                    "salary_min": job.get("salary", {}).get("min"),
                    "salary_max": job.get("salary", {}).get("max"),
                    "currency": job.get("salary", {}).get("currency"),
                    "job_type": job.get("employment_type"),
                    "seniority": job.get("seniority_level"),
                    "department": job.get("department"),
                    "url": job.get("url"),
                    "posted_date": job.get("created_at"),
                    "updated_date": job.get("updated_at"),
                    "contact_email": job.get("contact", {}).get("email"),
                    "contact_name": job.get("contact", {}).get("name"),
                }

            return None
        except Exception as e:
            self.logger.error(f"Error getting job details: {e}")
            return None

    def get_company_profile(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile and information.

        Args:
            company_id: Comeet company ID

        Returns:
            Company profile dictionary or None
        """
        if not self.api_key:
            return None

        try:
            response = self._make_request(f"/companies/{company_id}")

            if response.get("company"):
                company = response["company"]
                return {
                    "id": company.get("id"),
                    "name": company.get("name"),
                    "logo_url": company.get("logo_url"),
                    "website": company.get("website"),
                    "description": company.get("description"),
                    "industry": company.get("industry"),
                    "size": company.get("company_size"),
                    "location": company.get("headquarters", {}).get("name"),
                    "founded_year": company.get("founded_year"),
                    "open_positions": len(company.get("jobs", [])),
                }

            return None
        except Exception as e:
            self.logger.error(f"Error getting company profile: {e}")
            return None

    def search_companies(
        self,
        keywords: Optional[List[str]] = None,
        location: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for companies on Comeet.

        Args:
            keywords: Company name keywords
            location: Company location
            limit: Maximum results

        Returns:
            List of company dictionaries
        """
        if not self.api_key:
            return []

        try:
            query_parts = []
            if keywords:
                query_parts.extend(keywords)
            if location:
                query_parts.append(location)

            query = " ".join(query_parts) if query_parts else "*"

            params = {
                "q": query,
                "limit": min(limit, 200),
            }

            response = self._make_request("/companies/search", params=params)

            if response.get("companies"):
                companies = []
                for company in response["companies"]:
                    companies.append({
                        "id": company.get("id"),
                        "name": company.get("name"),
                        "logo_url": company.get("logo_url"),
                        "industry": company.get("industry"),
                        "size": company.get("company_size"),
                        "open_positions": len(company.get("jobs", [])),
                    })

                return companies

            return []
        except Exception as e:
            self.logger.error(f"Error searching companies: {e}")
            return []

    def _extract_keywords_from_job(self, job: Dict[str, Any]) -> List[str]:
        """Extract keywords from job title and description."""
        keywords = []

        title = job.get("title", "").lower()
        description = job.get("description", "").lower()

        # Common tech keywords
        tech_keywords = [
            "python", "javascript", "java", "c++", "go", "rust",
            "react", "vue", "angular", "node.js",
            "aws", "azure", "gcp", "kubernetes", "docker",
            "sql", "mongodb", "postgresql",
            "api", "rest", "graphql",
            "qa", "qe", "test", "automation",
            "devops", "ci/cd", "jenkins", "github",
            "agile", "scrum", "kanban",
        ]

        for keyword in tech_keywords:
            if keyword in title or keyword in description:
                if keyword not in keywords:
                    keywords.append(keyword)

        # Extract role level
        role_keywords = ["senior", "junior", "manager", "lead", "director", "principal", "architect"]
        for keyword in role_keywords:
            if keyword in title.lower():
                keywords.append(keyword)
                break

        return keywords[:10]

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string to datetime."""
        if not date_str:
            return None

        try:
            # Handle ISO 8601 format
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Comeet API.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            timeout: Request timeout

        Returns:
            JSON response
        """
        try:
            if params is None:
                params = {}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self.API_ENDPOINT}{endpoint}"

            with httpx.Client(timeout=timeout or self.timeout) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error calling Comeet API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error making request to Comeet API: {e}")
            raise

    @staticmethod
    def extract_domain_from_url(url: str) -> str:
        """Extract domain from job URL."""
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1) if match else ""


# Singleton instance
comeet = ComeetTool()
