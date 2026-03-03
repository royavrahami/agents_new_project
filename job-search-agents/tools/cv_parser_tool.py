"""
CV Parser Tool — Structured CV Analysis.

Integrates with Affinda CV parsing API for real, structured CV analysis.
Falls back to LLM-based analysis if API unavailable.

Responsibilities:
1. Parse CV into structured sections (contact, summary, experience, education, skills)
2. Extract skills with confidence scores
3. Detect ATS formatting issues
4. Provide detailed parsing success metrics
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx
from config.settings import settings
from core.logger import logger


class CVParserTool:
    """
    Real CV parsing via Affinda API.
    Falls back to regex-based parsing if API unavailable.

    Args:
        api_key: Affinda API key (from settings if not provided)
        timeout: HTTP timeout in seconds
    """

    API_ENDPOINT = "https://api.affinda.com/v3/documents/parse"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30) -> None:
        self.api_key = api_key or settings.affinda_api_key
        self.timeout = timeout
        self.logger = logger

    def parse_cv(self, cv_text: str) -> Dict[str, Any]:
        """
        Parse CV text using Affinda API or fallback to regex parsing.

        Args:
            cv_text: Raw CV content (text or PDF-as-text)

        Returns:
            {
                "success": bool,
                "parsing_method": "affinda" | "fallback",
                "sections": {
                    "contact": {...},
                    "summary": str,
                    "experience": [...],
                    "education": [...],
                    "skills": [...],
                    "certifications": [...]
                },
                "skills_extracted": [{"name": str, "confidence": float}],
                "formatting_issues": [str],
                "parsing_success_rate": float (0-100),
                "error": str (if failed)
            }
        """
        if self.api_key:
            return self._parse_with_affinda(cv_text)
        else:
            self.logger.warning("Affinda API key not configured, using fallback regex parsing")
            return self._parse_with_regex(cv_text)

    def _parse_with_affinda(self, cv_text: str) -> Dict[str, Any]:
        """
        Call Affinda API for professional CV parsing.

        Affinda returns structured data with:
        - Sections (contact, employment, education, skills, certifications)
        - Extracted fields with confidence scores
        - Quality metrics
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "text/plain",
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.API_ENDPOINT,
                    content=cv_text,
                    headers=headers,
                )
                response.raise_for_status()

            data = response.json()

            # Extract and structure Affinda response
            sections = self._extract_sections_from_affinda(data)
            skills = self._extract_skills_from_affinda(data)
            formatting_issues = self._detect_formatting_issues_from_affinda(data)
            parsing_rate = self._calculate_affinda_success_rate(data)

            return {
                "success": True,
                "parsing_method": "affinda",
                "sections": sections,
                "skills_extracted": skills,
                "formatting_issues": formatting_issues,
                "parsing_success_rate": parsing_rate,
                "raw_affinda_data": data,
            }
        except httpx.HTTPError as e:
            self.logger.error(f"Affinda API error: {e}")
            # Fall back to regex parsing
            return self._parse_with_regex(cv_text)
        except Exception as e:
            self.logger.error(f"Unexpected error in Affinda parsing: {e}")
            return self._parse_with_regex(cv_text)

    def _parse_with_regex(self, cv_text: str) -> Dict[str, Any]:
        """
        Fallback regex-based CV parsing when Affinda API unavailable.
        Extracts basic structure and skills.
        """
        try:
            sections = {
                "contact": self._extract_contact_info(cv_text),
                "summary": self._extract_summary(cv_text),
                "experience": self._extract_experience(cv_text),
                "education": self._extract_education(cv_text),
                "skills": self._extract_skills_section(cv_text),
                "certifications": self._extract_certifications(cv_text),
            }

            skills = self._extract_skills_from_text(cv_text)
            formatting_issues = self._detect_ats_killers(cv_text)
            parsing_rate = self._estimate_parsing_success_rate(sections)

            return {
                "success": True,
                "parsing_method": "fallback",
                "sections": sections,
                "skills_extracted": skills,
                "formatting_issues": formatting_issues,
                "parsing_success_rate": parsing_rate,
            }
        except Exception as e:
            self.logger.error(f"Error in fallback parsing: {e}")
            return {
                "success": False,
                "parsing_method": "fallback",
                "error": str(e),
                "sections": {},
                "skills_extracted": [],
                "formatting_issues": [],
                "parsing_success_rate": 0.0,
            }

    # ----------- Affinda Response Extraction ----------

    def _extract_sections_from_affinda(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sections from Affinda response."""
        sections = {}

        # Contact info
        if "data" in data and "personal_details" in data["data"]:
            sections["contact"] = {
                "name": data["data"]["personal_details"].get("name"),
                "email": data["data"]["personal_details"].get("email"),
                "phone": data["data"]["personal_details"].get("phone_number"),
                "location": data["data"]["personal_details"].get("location"),
            }

        # Summary/objective
        if "data" in data and "summary" in data["data"]:
            sections["summary"] = data["data"].get("summary", "")

        # Employment history
        sections["experience"] = []
        if "data" in data and "employment_history" in data["data"]:
            for job in data["data"]["employment_history"]:
                sections["experience"].append({
                    "company": job.get("company"),
                    "title": job.get("job_title"),
                    "start_date": job.get("start_date"),
                    "end_date": job.get("end_date"),
                    "description": job.get("description"),
                })

        # Education
        sections["education"] = []
        if "data" in data and "education" in data["data"]:
            for edu in data["data"]["education"]:
                sections["education"].append({
                    "institution": edu.get("organization"),
                    "degree": edu.get("qualification"),
                    "field": edu.get("field_of_study"),
                    "graduation_date": edu.get("end_date"),
                })

        # Skills from Affinda
        sections["skills"] = []
        if "data" in data and "skills" in data["data"]:
            for skill in data["data"]["skills"]:
                sections["skills"].append({
                    "name": skill.get("name"),
                    "confidence": skill.get("confidence", 0.8),
                })

        # Certifications
        sections["certifications"] = []
        if "data" in data and "certifications" in data["data"]:
            for cert in data["data"]["certifications"]:
                sections["certifications"].append({
                    "name": cert.get("name"),
                    "issued_by": cert.get("issuing_organization"),
                    "date": cert.get("date"),
                })

        return sections

    def _extract_skills_from_affinda(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract skills with confidence scores from Affinda."""
        skills = []
        if "data" in data and "skills" in data["data"]:
            for skill in data["data"]["skills"]:
                skills.append({
                    "name": skill.get("name", ""),
                    "confidence": float(skill.get("confidence", 0.8)),
                })
        return skills

    def _detect_formatting_issues_from_affinda(self, data: Dict[str, Any]) -> List[str]:
        """Detect ATS formatting issues reported by Affinda."""
        issues = []

        # Check if Affinda detected parsing issues
        if "metadata" in data:
            quality_score = data["metadata"].get("quality_score", 100)
            if quality_score < 70:
                issues.append(f"Low parsing quality ({quality_score}%): CV may have formatting issues")

        # Common issues from API response
        if "data" in data:
            # If sections are empty, that's an issue
            if not data["data"].get("personal_details", {}).get("name"):
                issues.append("No name found in contact section")
            if not data["data"].get("employment_history"):
                issues.append("No employment history found")

        return issues

    def _calculate_affinda_success_rate(self, data: Dict[str, Any]) -> float:
        """Calculate parsing success rate from Affinda metadata."""
        if "metadata" in data:
            return float(data["metadata"].get("quality_score", 0)) / 100.0
        return 0.5

    # ----------- Regex-based Fallback Parsing ----------

    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from CV text."""
        contact = {
            "name": None,
            "email": None,
            "phone": None,
            "location": None,
        }

        # Email regex
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if email_match:
            contact["email"] = email_match.group(0)

        # Phone regex (various formats)
        phone_match = re.search(
            r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|"
            r"\+\d{1,3}[-.\s]?\d{1,14}",
            text
        )
        if phone_match:
            contact["phone"] = phone_match.group(0)

        return contact

    def _extract_summary(self, text: str) -> str:
        """Extract professional summary section."""
        # Look for common summary headers
        patterns = [
            r"(?:Professional\s+)?Summary.*?(?=\n(?:Experience|Education|Skills|Certification)|\Z)",
            r"(?:Executive\s+)?Profile.*?(?=\n(?:Experience|Education|Skills|Certification)|\Z)",
            r"Objective.*?(?=\n(?:Experience|Education|Skills|Certification)|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()[:500]

        return ""

    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience entries."""
        jobs = []

        # Pattern: Company, Title, Dates
        experience_section = re.search(
            r"(?:Experience|Employment).*?(?=\n(?:Education|Skills|Certification)|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not experience_section:
            return jobs

        section_text = experience_section.group(0)

        # Split by common patterns (company names often on their own line)
        entries = re.split(r"\n(?=[A-Z][a-zA-Z\s]+(?:\n|\s+(?:20\d{2}|19\d{2})))", section_text)

        for entry in entries[:10]:  # Limit to 10 jobs
            if len(entry.strip()) < 20:
                continue

            jobs.append({
                "company": "",
                "title": "",
                "start_date": "",
                "end_date": "",
                "description": entry[:200],
            })

        return jobs[:5]

    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education entries."""
        education = []

        # Pattern: Degree, University, Date
        education_section = re.search(
            r"(?:Education|Academic).*?(?=\n(?:Skills|Certification|Experience)|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not education_section:
            return education

        section_text = education_section.group(0)

        # Look for degree patterns (Bachelor, Master, PhD, etc.)
        degrees = re.findall(
            r"(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|M\.A\.|B\.A\.).*?(?:\n|$)",
            section_text,
            re.IGNORECASE
        )

        for degree in degrees[:5]:
            education.append({
                "institution": "",
                "degree": degree.strip(),
                "field": "",
                "graduation_date": "",
            })

        return education

    def _extract_skills_section(self, text: str) -> List[str]:
        """Extract skills list from dedicated section."""
        skills_section = re.search(
            r"(?:Technical\s+)?Skills.*?(?=\n(?:Experience|Education|Certification|Projects)|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not skills_section:
            return []

        section_text = skills_section.group(0)

        # Split by common delimiters
        skills = re.split(r"[,•\n]+", section_text)
        skills = [s.strip() for s in skills if len(s.strip()) > 0 and len(s.strip()) < 50]

        return skills[:20]

    def _extract_certifications(self, text: str) -> List[Dict[str, str]]:
        """Extract certifications."""
        certs = []

        certs_section = re.search(
            r"(?:Certification|Certified|License).*?(?=\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not certs_section:
            return certs

        section_text = certs_section.group(0)

        # Common certifications
        cert_patterns = [
            r"(?:AWS|Azure|GCP)[^,\n]*",
            r"ISTQB[^,\n]*",
            r"SCRUM[^,\n]*",
            r"PMP[^,\n]*",
            r"Six\s+Sigma[^,\n]*",
        ]

        for pattern in cert_patterns:
            matches = re.findall(pattern, section_text, re.IGNORECASE)
            for match in matches[:3]:
                certs.append({
                    "name": match.strip(),
                    "issued_by": "",
                    "date": "",
                })

        return certs[:5]

    def _extract_skills_from_text(self, text: str) -> List[Dict[str, float]]:
        """Extract skills throughout CV text with confidence scores."""
        skills = []

        # Common technical skills
        technical_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
            "AWS", "Azure", "GCP", "Kubernetes", "Docker",
            "React", "Vue", "Angular", "Node.js",
            "Git", "GitHub", "GitLab", "CI/CD", "Jenkins", "GitHub Actions",
            "Linux", "Windows", "macOS",
            "Agile", "Scrum", "Kanban",
            "Pytest", "Playwright", "Selenium", "JUnit", "TestNG",
            "ISTQB", "CTAL", "CTFL",
            "API", "REST", "GraphQL", "Microservices",
            "Machine Learning", "AI", "Deep Learning", "TensorFlow", "PyTorch",
        ]

        for skill in technical_skills:
            if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
                # Confidence based on number of mentions
                mention_count = len(re.findall(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE))
                confidence = min(0.95, 0.7 + (mention_count * 0.1))

                if not any(s["name"].lower() == skill.lower() for s in skills):
                    skills.append({
                        "name": skill,
                        "confidence": round(confidence, 2),
                    })

        return skills

    def _detect_ats_killers(self, text: str) -> List[str]:
        """Detect ATS-disqualifying formatting issues."""
        issues = []

        # HTML tags
        if re.search(r"<[^>]+>", text):
            issues.append("Contains HTML tags (will fail most ATS systems)")

        # Images
        if re.search(r"\.(png|jpg|jpeg|gif)", text, re.IGNORECASE):
            issues.append("References image files (ATS cannot process images)")

        # Non-ASCII sequences
        if re.search(r"[^\x00-\x7F]{3,}", text):
            issues.append("Contains non-ASCII characters (may cause ATS parsing issues)")

        # Tables
        if re.search(r"(\|[^\|]*\|)|(\+[-+]*\+)", text):
            issues.append("Contains table formatting (ATS may not parse correctly)")

        # Multiple spaces (often from poor PDF conversion)
        if re.search(r"  {2,}", text):
            issues.append("Multiple consecutive spaces detected (possible PDF conversion artifact)")

        return issues

    def _estimate_parsing_success_rate(self, sections: Dict[str, Any]) -> float:
        """Estimate parsing success based on sections found."""
        success_count = 0
        max_sections = 6

        if sections.get("contact", {}).get("email"):
            success_count += 1
        if sections.get("summary") and len(sections["summary"]) > 20:
            success_count += 1
        if sections.get("experience") and len(sections["experience"]) > 0:
            success_count += 1
        if sections.get("education") and len(sections["education"]) > 0:
            success_count += 1
        if sections.get("skills") and len(sections["skills"]) > 0:
            success_count += 1
        if sections.get("certifications") and len(sections["certifications"]) > 0:
            success_count += 1

        return round((success_count / max_sections) * 100, 1)


# Singleton instance
cv_parser = CVParserTool()
