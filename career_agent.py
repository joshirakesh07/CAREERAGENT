import os
import json
import requests
import uvicorn

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from pypdf import PdfReader


# ============================================================
# 1. LOAD API KEY
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
    Search current jobs related to the target role.
    """

    try:

        url = "https://remotive.com/api/remote-jobs"

        response = requests.get(
            url,
            params={"search": role},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        jobs = []

        keywords = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml engineer",
            "machine learning engineer",
            "deep learning",
            "data scientist",
            "computer vision",
            "nlp",
            "generative ai",
            "llm"
        ]

        for job in data.get("jobs", []):

            title = (
                job.get("title") or ""
            ).lower()

            if any(
                word in title
                for word in keywords
            ):

                jobs.append({
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "location": job.get(
                        "candidate_required_location"
                    ),
                    "job_type": job.get("job_type"),
                    "url": job.get("url")
                })

            if len(jobs) >= 10:
                break

        return json.dumps(
            {
                "role": role,
                "jobs": jobs
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
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
    Analyze the student's skill gaps for the target role.
    """

    prompt = f"""
You are an AI/ML career advisor.

Target role:
{role}

Student resume:
{resume_text}

Analyze:

1. Existing skills
2. Matching skills
3. Missing skills
4. Skills to improve
5. Technologies to learn
6. Priority of skills
7. Learning roadmap

Keep the answer practical for a college student.
"""

    try:

        response = llm.invoke(prompt)

        return get_text(response)

    except Exception as e:

        return "Skill gap analysis failed: " + str(e)


# ============================================================
# 5. PROJECT TOOL
# ============================================================

@tool
def project_idea_lookup(
    role: str,
    skills: str
) -> str:
    """
    Find GitHub projects related to the target role.
    """

    try:

        url = (
            "https://api.github.com/"
            "search/repositories"
        )

        query = f"{role} {skills}"

        response = requests.get(
            url,
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10
            },
            headers={
                "Accept":
                    "application/vnd.github+json",
                "User-Agent":
                    "AI-Career-Agent"
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        projects = []

        for repo in data.get(
            "items",
            []
        ):

            projects.append({
                "name": repo.get("name"),
                "description": repo.get(
                    "description"
                ),
                "language": repo.get(
                    "language"
                ),
                "stars": repo.get(
                    "stargazers_count"
                ),
                "url": repo.get(
                    "html_url"
                )
            })

        return json.dumps(
            {
                "projects": projects
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
        })


# ============================================================
# 6. GITHUB TOOL
# ============================================================

@tool
def github_check(github_id: str) -> str:
    """
    Check public GitHub profile and repositories.
    """

    username = github_id.strip()

    if "github.com/" in username:

        username = username.split(
            "github.com/",
            1
        )[1]

    username = username.strip("/")

    if username.startswith("@"):

        username = username[1:]

    username = username.replace(
        " ",
        ""
    )

    headers = {
        "Accept":
            "application/vnd.github+json",
        "User-Agent":
            "AI-Career-Agent"
    }

    try:

        user_url = (
            f"https://api.github.com/"
            f"users/{username}"
        )

        user_response = requests.get(
            user_url,
            headers=headers,
            timeout=20
        )

        if user_response.status_code == 404:

            return json.dumps({
                "error":
                    f"GitHub user '{username}' "
                    f"not found."
            })

        user_response.raise_for_status()

        user = user_response.json()

        repo_url = (
            f"https://api.github.com/"
            f"users/{username}/repos"
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
                "description": repo.get(
                    "description"
                ),
                "language": repo.get(
                    "language"
                ),
                "stars": repo.get(
                    "stargazers_count"
                ),
                "forks": repo.get(
                    "forks_count"
                ),
                "updated": repo.get(
                    "updated_at"
                ),
                "url": repo.get(
                    "html_url"
                )
            })

        return json.dumps(
            {
                "username":
                    user.get("login"),

                "name":
                    user.get("name"),

                "bio":
                    user.get("bio"),

                "public_repositories":
                    user.get("public_repos"),

                "followers":
                    user.get("followers"),

                "profile":
                    user.get("html_url"),

                "repositories":
                    repos
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
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
# 8. AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""
You are an AI Career Advisor.

The user provides:

- Resume
- Target role
- GitHub username

Use the available tools when useful:

1. Job Search
2. Skill Gap Analysis
3. Project Ideas
4. GitHub Check

After using the tools, return the information needed
for a final career analysis.

Do not invent information.

The final answer will be synthesized separately.
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
        description="Target role"
    )

    github_id: str = Field(
        description="GitHub username"
    )


# ============================================================
# 10. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        resume = x.get(
            "resume_text",
            ""
        )

        role = x.get(
            "role",
            ""
        )

        github = x.get(
            "github_id",
            ""
        )

    else:

        resume = x.resume_text
        role = x.role
        github = x.github_id

    return {
        "messages": [
            (
                "user",
                f"""
Student Career Analysis

TARGET ROLE:
{role}

GITHUB:
{github}

RESUME:
{resume}

Analyze this student.

Use the available tools for:

- Job search
- Skill gap
- Projects
- GitHub

Then provide the information needed for
a final career analysis.
"""
            )
        ]
    }


# ============================================================
# 11. GET TEXT
# ============================================================

def get_text(message):

    if message is None:
        return ""

    content = getattr(
        message,
        "content",
        None
    )

    if content is None:
        return ""

    if isinstance(
        content,
        str
    ):

        return content.strip()

    if isinstance(
        content,
        list
    ):

        result = []

        for item in content:

            if isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    result.append(
                        item.get(
                            "text",
                            ""
                        )
                    )

            elif isinstance(
                item,
                str
            ):

                result.append(item)

        return "\n".join(
            result
        ).strip()

    return str(content).strip()


# ============================================================
# 12. FINAL SYNTHESIS
# ============================================================

def final_synthesis(
    agent_output
):

    if not isinstance(
        agent_output,
        dict
    ):

        return str(
            agent_output
        )

    messages = agent_output.get(
        "messages",
        []
    )

    # --------------------------------------------------------
    # Collect all useful messages
    # --------------------------------------------------------

    collected = []

    for message in messages:

        text = get_text(message)

        if text:

            message_type = (
                message.__class__.__name__
            )

            collected.append(
                f"{message_type}:\n{text}"
            )

    # --------------------------------------------------------
    # Get tool calls/results
    # --------------------------------------------------------

    tool_information = []

    for message in messages:

        message_type = (
            message.__class__.__name__
        )

        if (
            "ToolMessage"
            in message_type
        ):

            text = get_text(message)

            if text:

                tool_information.append(
                    text
                )

    # --------------------------------------------------------
    # If there are no tool results, use all messages
    # --------------------------------------------------------

    if tool_information:

        information = "\n\n".join(
            tool_information
        )

    else:

        information = "\n\n".join(
            collected
        )

    # --------------------------------------------------------
    # Ask Gemma for FINAL ANSWER
    # --------------------------------------------------------

    prompt = f"""
You are the final AI Career Advisor.

Create a complete final answer using the
information below.

INFORMATION:
{information}

Return ONLY the final career analysis.

Use this structure:

# AI Career Analysis

## 1. Career Suitability

## 2. Matching Skills

## 3. Skill Gaps

## 4. Recommended Skills to Learn

## 5. Job Opportunities

## 6. Recommended Projects

## 7. GitHub Analysis

## 8. 30/60/90 Day Roadmap

## 9. Final Recommendation

Be clear and practical.

Do not invent information.

If a section has no available information,
say "No information available."
"""

    try:

        response = llm.invoke(
            prompt
        )

        result = get_text(
            response
        )

        if result:

            return result

    except Exception as e:

        return (
            "Final synthesis error: "
            + str(e)
            + "\n\n"
            + information
        )

    return (
        "No final response was generated.\n\n"
        + information
    )


# ============================================================
# 13. AGENT CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        final_synthesis
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 14. FASTAPI
# ============================================================

app = FastAPI(
    title="AI Career Advisor Agent",
    version="1.0.0"
)


# ============================================================
# 15. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# 16. LANGSERVE
# ============================================================

add_routes(
    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 17. PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes
):

    import io

    pdf = PdfReader(
        io.BytesIO(
            file_bytes
        )
    )

    text = ""

    for page in pdf.pages:

        page_text = (
            page.extract_text()
        )

        if page_text:

            text += (
                page_text
                + "\n"
            )

    return text


# ============================================================
# 18. PDF API
# ============================================================

@app.post(
    "/analyze-pdf"
)
async def analyze_pdf(

    resume: UploadFile = File(...),

    role: str = Form(...),

    github_id: str = Form(...)

):

    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_bytes = await resume.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="PDF is empty."
        )

    try:

        resume_text = extract_pdf_text(
            file_bytes
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    if not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail="Could not extract PDF text."
        )

    result = await formatted_agent_chain.ainvoke({

        "resume_text":
            resume_text,

        "role":
            role,

        "github_id":
            github_id

    })

    return {

        "status":
            "success",

        "role":
            role,

        "github_id":
            github_id,

        "analysis":
            result

    }


# ============================================================
# 19. HOME
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
# 20. RUN
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
