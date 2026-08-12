

import os
import json
import requests
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from pypdf import PdfReader


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# ============================================================
# 2. GEMMA MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. JOB SEARCH TOOL
# ============================================================

@tool
def job_search(role: str) -> str:
    """
    Search current jobs and internships using a public jobs API.
    """

    try:

        url = "https://remotive.com/api/remote-jobs"

        response = requests.get(
            url,
            params={
                "search": role
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        jobs = []

        for job in data.get("jobs", [])[:10]:

            jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "location": job.get("candidate_required_location"),
                "job_type": job.get("job_type"),
                "url": job.get("url"),
                "description": job.get("description", "")[:500]
            })

        return json.dumps(
            {
                "role": role,
                "jobs": jobs
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": f"Job search failed: {str(e)}"
        })


# ============================================================
# 4. SKILL GAP TOOL
# ============================================================

@tool
def skill_gap_analysis(
    resume_text: str,
    role: str
) -> str:
    """
    Compare the student's resume with the target role.
    """

    prompt = f"""
You are an AI career advisor.

Analyze the student's resume for the target role.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text}

Provide:

1. Existing skills
2. Matching skills
3. Missing skills
4. Skills that need improvement
5. Technologies to learn
6. Priority:
   High / Medium / Low
7. Short learning roadmap

Keep the recommendations practical for a college student
or fresher.
"""

    try:

        response = llm.invoke(prompt)

        return str(response.content)

    except Exception as e:

        return f"Skill gap analysis failed: {str(e)}"


# ============================================================
# 5. PROJECT IDEA TOOL
# ============================================================

@tool
def project_idea_lookup(
    role: str,
    skills: str
) -> str:
    """
    Find project ideas using the public GitHub API.
    """

    try:

        query = f"{role} {skills}"

        url = "https://api.github.com/search/repositories"

        response = requests.get(
            url,
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10
            },
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "AI-Career-Agent"
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        projects = []

        for repo in data.get("items", []):

            projects.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count"),
                "url": repo.get("html_url")
            })

        return json.dumps(
            {
                "role": role,
                "projects": projects
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": f"Project lookup failed: {str(e)}"
        })


# ============================================================
# 6. GITHUB CHECK TOOL
# ============================================================

@tool
def github_check(github_id: str) -> str:
    """
    Check a student's public GitHub profile and repositories.
    """

    username = github_id.strip()

    if "github.com/" in username:

        username = username.split(
            "github.com/"
        )[1].strip("/")

    if username.startswith("@"):

        username = username[1:]

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AI-Career-Agent"
    }

    try:

        # ----------------------------------------------------
        # GitHub profile
        # ----------------------------------------------------

        user_url = (
            f"https://api.github.com/users/{username}"
        )

        user_response = requests.get(
            user_url,
            headers=headers,
            timeout=20
        )

        if user_response.status_code == 404:

            return json.dumps({
                "error": "GitHub user not found"
            })

        user_response.raise_for_status()

        user = user_response.json()

        # ----------------------------------------------------
        # GitHub repositories
        # ----------------------------------------------------

        repo_url = (
            f"https://api.github.com/users/"
            f"{username}/repos"
        )

        repo_response = requests.get(
            repo_url,
            headers=headers,
            params={
                "sort": "updated",
                "per_page": 10
            },
            timeout=20
        )

        repo_response.raise_for_status()

        repositories = repo_response.json()

        repos = []

        for repo in repositories:

            repos.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "updated": repo.get("updated_at"),
                "url": repo.get("html_url")
            })

        result = {

            "username": user.get("login"),

            "name": user.get("name"),

            "bio": user.get("bio"),

            "public_repositories":
                user.get("public_repos"),

            "followers":
                user.get("followers"),

            "following":
                user.get("following"),

            "profile":
                user.get("html_url"),

            "recent_repositories":
                repos
        }

        return json.dumps(
            result,
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": f"GitHub check failed: {str(e)}"
        })


# ============================================================
# 7. TOOLS
# ============================================================

tools = [
    job_search,
    skill_gap_analysis,
    project_idea_lookup,
    github_check
]


# ============================================================
# 8. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""
You are an AI Career Advisor Agent.

The student provides:

- Resume PDF
- Target role
- GitHub ID

You have four tools:

1. Job Search
2. Skill Gap Analysis
3. Project Idea Lookup
4. GitHub Check

Use the appropriate tools to analyze the student's profile.

For a complete analysis, use all four tools when possible.

After receiving the tool results, create a FINAL SYNTHESIS.

The final answer must contain:

1. Career suitability
2. Matching skills
3. Skill gaps
4. Recommended skills to learn
5. Job opportunities
6. Recommended projects
7. GitHub analysis
8. 30/60/90 day roadmap
9. Final recommendation

Do not invent information.

Clearly explain the results in simple language suitable
for a college student.

If a tool does not return results, clearly mention that.
"""
)


# ============================================================
# 9. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    resume_text: str = Field(
        description="Student resume text"
    )

    role: str = Field(
        description="Target job role"
    )

    github_id: str = Field(
        description="GitHub username or profile URL"
    )


# ============================================================
# 10. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        resume_text = x["resume_text"]
        role = x["role"]
        github_id = x["github_id"]

    else:

        resume_text = x.resume_text
        role = x.role
        github_id = x.github_id

    message = f"""
Student Career Analysis

TARGET ROLE:
{role}

GITHUB ID:
{github_id}

RESUME:
{resume_text}

Perform a complete career analysis.

Use the available tools for:

- Job search
- Skill gap analysis
- Project recommendations
- GitHub analysis

Then provide a final synthesis.
"""

    return {
        "messages": [
            ("user", message)
        ]
    }


# ============================================================
# 11. EXTRACT FINAL RESPONSE
# ============================================================

def extract_text_response(output):

    if not isinstance(output, dict):

        return str(output)

    messages = output.get("messages")

    if messages:

        last = messages[-1]

        content = getattr(
            last,
            "content",
            last
        )

        return str(content)

    return str(output)


# ============================================================
# 12. AGENT CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(format_for_agent)

    | agent

    | RunnableLambda(extract_text_response)

).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 13. FASTAPI
# ============================================================

app = FastAPI(
    title="AI Career Advisor Agent",
    description="AI Career Agent using Gemma 4 and FastAPI"
)


# ============================================================
# 14. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 15. LANGSERVE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 16. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    import io

    pdf = PdfReader(
        io.BytesIO(file_bytes)
    )

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# ============================================================
# 17. PDF ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze-pdf")
async def analyze_pdf(

    resume: UploadFile = File(...),

    role: str = Form(...),

    github_id: str = Form(...)

):

    if not resume.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_bytes = await resume.read()

    try:

        resume_text = extract_pdf_text(
            file_bytes
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"PDF reading failed: {str(e)}"
        )

    if not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF."
        )

    result = await formatted_agent_chain.ainvoke({

        "resume_text": resume_text,

        "role": role,

        "github_id": github_id

    })

    return {

        "status": "success",

        "role": role,

        "github_id": github_id,

        "analysis": result

    }


# ============================================================
# 18. HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "AI Career Advisor Agent is running",

        "model":
            "gemma-4-31b-it",

        "tools": [
            "job_search",
            "skill_gap_analysis",
            "project_idea_lookup",
            "github_check"
        ]

    }


# ============================================================
# 19. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
