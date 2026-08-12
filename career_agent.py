

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
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# ============================================================
# 2. GEMMA 4 - AGENT CORE
# ============================================================

gemma = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    thinking_level="minimal"
)


# ============================================================
# 3. GEMINI FLASH - FINAL SYNTHESIS
# ============================================================

final_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 4. JOB SEARCH TOOL
# ============================================================

@tool
def job_search(role: str) -> str:
    """
    Search for jobs related to the student's target role.
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

        all_jobs = data.get(
            "jobs",
            []
        )

        keywords = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml engineer",
            "machine learning engineer",
            "deep learning",
            "data scientist",
            "data science",
            "computer vision",
            "nlp",
            "generative ai",
            "llm"
        ]

        jobs = []

        for job in all_jobs:

            title = (
                job.get("title")
                or ""
            ).lower()

            if any(
                keyword in title
                for keyword in keywords
            ):

                jobs.append({

                    "title":
                        job.get("title"),

                    "company":
                        job.get(
                            "company_name"
                        ),

                    "location":
                        job.get(
                            "candidate_required_location"
                        ),

                    "job_type":
                        job.get(
                            "job_type"
                        ),

                    "url":
                        job.get(
                            "url"
                        )

                })

            if len(jobs) >= 10:
                break

        return json.dumps(
            {
                "target_role":
                    role,

                "jobs_found":
                    len(jobs),

                "jobs":
                    jobs

            },
            indent=2
        )

    except Exception as e:

        return json.dumps({

            "error":
                f"Job search failed: {str(e)}"

        })


# ============================================================
# 5. SKILL GAP TOOL
# ============================================================

@tool
def skill_gap_analysis(
    resume_text: str,
    role: str
) -> str:
    """
    Compare the resume with the target role.
    """

    prompt = f"""
You are an expert AI/ML career advisor.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text}

Analyze the student's preparation for the target role.

Give:

1. Existing skills
2. Matching skills
3. Missing skills
4. Skills to improve
5. Technologies to learn
6. Priority of each skill
7. Practical learning roadmap

Focus on a college student / fresher.

Do not invent information.
"""

    try:

        response = final_llm.invoke(
            prompt
        )

        return extract_text(
            response
        )

    except Exception as e:

        return (
            "Skill gap analysis failed: "
            + str(e)
        )


# ============================================================
# 6. PROJECT IDEA TOOL
# ============================================================

@tool
def project_idea_lookup(
    role: str,
    skills: str
) -> str:
    """
    Find relevant GitHub projects for the target role.
    """

    try:

        url = (
            "https://api.github.com/"
            "search/repositories"
        )

        query = (
            f"{role} {skills}"
        )

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

                "name":
                    repo.get(
                        "name"
                    ),

                "description":
                    repo.get(
                        "description"
                    ),

                "language":
                    repo.get(
                        "language"
                    ),

                "stars":
                    repo.get(
                        "stargazers_count"
                    ),

                "url":
                    repo.get(
                        "html_url"
                    )

            })

        return json.dumps(

            {
                "target_role":
                    role,

                "projects":
                    projects

            },

            indent=2

        )

    except Exception as e:

        return json.dumps({

            "error":
                f"Project search failed: {str(e)}"

        })


# ============================================================
# 7. GITHUB CHECK TOOL
# ============================================================

@tool
def github_check(
    github_id: str
) -> str:
    """
    Check a public GitHub profile and repositories.
    """

    username = github_id.strip()

    # If user enters full URL
    if "github.com/" in username:

        username = username.split(
            "github.com/",
            1
        )[1]

    username = username.strip("/")

    # Remove @
    if username.startswith("@"):

        username = username[1:]

    # Remove accidental spaces
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

        # ----------------------------------------------------
        # Profile
        # ----------------------------------------------------

        profile_url = (
            f"https://api.github.com/"
            f"users/{username}"
        )

        profile_response = requests.get(

            profile_url,

            headers=headers,

            timeout=20

        )

        if profile_response.status_code == 404:

            return json.dumps({

                "error":
                    f"GitHub user '{username}' "
                    f"was not found.",

                "github_id_used":
                    username

            })

        profile_response.raise_for_status()

        profile = profile_response.json()


        # ----------------------------------------------------
        # Repositories
        # ----------------------------------------------------

        repos_url = (

            f"https://api.github.com/"
            f"users/{username}/repos"

        )

        repos_response = requests.get(

            repos_url,

            headers=headers,

            params={

                "sort":
                    "updated",

                "per_page":
                    10

            },

            timeout=20

        )

        repos_response.raise_for_status()

        repositories = (
            repos_response.json()
        )

        repos = []

        for repo in repositories:

            repos.append({

                "name":
                    repo.get(
                        "name"
                    ),

                "description":
                    repo.get(
                        "description"
                    ),

                "language":
                    repo.get(
                        "language"
                    ),

                "stars":
                    repo.get(
                        "stargazers_count"
                    ),

                "forks":
                    repo.get(
                        "forks_count"
                    ),

                "updated":
                    repo.get(
                        "updated_at"
                    ),

                "url":
                    repo.get(
                        "html_url"
                    )

            })


        return json.dumps(

            {

                "username":
                    profile.get(
                        "login"
                    ),

                "name":
                    profile.get(
                        "name"
                    ),

                "bio":
                    profile.get(
                        "bio"
                    ),

                "public_repositories":
                    profile.get(
                        "public_repos"
                    ),

                "followers":
                    profile.get(
                        "followers"
                    ),

                "following":
                    profile.get(
                        "following"
                    ),

                "profile":
                    profile.get(
                        "html_url"
                    ),

                "repositories":
                    repos

            },

            indent=2

        )

    except Exception as e:

        return json.dumps({

            "error":
                f"GitHub check failed: {str(e)}"

        })


# ============================================================
# 8. TOOLS
# ============================================================

tools = [

    job_search,

    skill_gap_analysis,

    project_idea_lookup,

    github_check

]


# ============================================================
# 9. GEMMA AGENT
# ============================================================

agent = create_agent(

    model=gemma,

    tools=tools,

    system_prompt="""
You are an AI Career Advisor Agent.

Your task is to analyze a student's:

- Resume
- Target role
- GitHub username

You have four tools:

1. job_search
2. skill_gap_analysis
3. project_idea_lookup
4. github_check

Use the tools to collect information.

For a complete career analysis, use all four tools.

IMPORTANT:

Do not invent information.

Use the GitHub username exactly as provided.

Use job search results only when returned by the tool.

After tool calls, stop and return the collected
information. A separate model will create the final
career analysis.
"""

)


# ============================================================
# 10. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    resume_text: str = Field(
        description="Student resume text"
    )

    role: str = Field(
        description="Target career role"
    )

    github_id: str = Field(
        description="GitHub username"
    )


# ============================================================
# 11. TEXT EXTRACTION
# ============================================================

def extract_text(
    message
):

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

        parts = []

        for item in content:

            if isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:

                        parts.append(
                            text
                        )

            elif isinstance(
                item,
                str
            ):

                parts.append(
                    item
                )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ============================================================
# 12. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(
        x,
        dict
    ):

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


    message = f"""
Student Career Analysis

TARGET ROLE:
{role}

GITHUB ID:
{github}

RESUME:
{resume}

Use all available tools:

1. Job Search
2. Skill Gap Analysis
3. Project Idea Lookup
4. GitHub Check

Collect the tool results.
"""

    return {

        "messages": [

            (
                "user",
                message
            )

        ]

    }


# ============================================================
# 13. EXTRACT TOOL RESULTS
# ============================================================

def collect_tool_results(
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

    results = []

    for message in messages:

        message_type = getattr(
            message,
            "type",
            ""
        )

        class_name = (
            message.__class__.__name__
        )

        # ToolMessage
        if (
            message_type == "tool"
            or
            "ToolMessage"
            in class_name
        ):

            text = extract_text(
                message
            )

            if text:

                results.append(
                    text
                )

    # --------------------------------------------------------
    # Fallback: collect all useful content
    # --------------------------------------------------------

    if not results:

        for message in messages:

            text = extract_text(
                message
            )

            if text:

                results.append(
                    text
                )

    if not results:

        return "No tool results were returned."

    return "\n\n".join(
        results
    )


# ============================================================
# 14. FINAL SYNTHESIS USING GEMINI FLASH
# ============================================================

def create_final_answer(
    agent_output
):

    tool_results = (
        collect_tool_results(
            agent_output
        )
    )


    prompt = f"""
You are the final AI Career Advisor.

Create a complete career analysis using
the tool results below.

TOOL RESULTS:
{tool_results}

Return ONLY the final answer.

Use exactly this structure:

# AI Career Analysis

## 1. Career Suitability

Explain whether the student is suitable
for the target role.

## 2. Matching Skills

List the student's skills that match
the target role.

## 3. Skill Gaps

List important missing or weak skills.

## 4. Recommended Skills to Learn

Give practical skills and technologies.

## 5. Job Opportunities

Summarize relevant jobs returned by
the job search tool.

## 6. Recommended Projects

Recommend projects based on the
student's skills and target role.

## 7. GitHub Analysis

Analyze the GitHub information returned
by the GitHub tool.

## 8. 30/60/90 Day Roadmap

30 Days:
...

60 Days:
...

90 Days:
...

## 9. Final Recommendation

Give a clear recommendation.

IMPORTANT:

Do not invent GitHub information.

Do not invent job information.

If information is unavailable,
say "No information available."

Keep the answer suitable for a
B.Tech student / fresher.
"""


    try:

        response = final_llm.invoke(
            prompt
        )

        result = extract_text(
            response
        )

        if result:

            return result

    except Exception as e:

        return (
            "Final synthesis failed: "
            + str(e)
            + "\n\nTool Results:\n"
            + tool_results
        )


    return (
        "Final synthesis returned no text.\n\n"
        + tool_results
    )


# ============================================================
# 15. COMPLETE AGENT CHAIN
# ============================================================

def run_career_agent(
    x
):

    # --------------------------------------------------------
    # Format input
    # --------------------------------------------------------

    formatted = (
        format_for_agent(x)
    )


    # --------------------------------------------------------
    # Run Gemma agent
    # --------------------------------------------------------

    try:

        agent_output = (
            agent.invoke(
                formatted
            )
        )

    except Exception as e:

        return (
            "Agent execution failed: "
            + str(e)
        )


    # --------------------------------------------------------
    # Final synthesis
    # --------------------------------------------------------

    return create_final_answer(
        agent_output
    )


formatted_agent_chain = (

    RunnableLambda(
        run_career_agent
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 16. FASTAPI
# ============================================================

app = FastAPI(

    title=
        "AI Career Advisor Agent",

    description=
        "Gemma 4 Agent with Career Analysis Tools",

    version="2.0.0"

)


# ============================================================
# 17. CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# 18. LANGSERVE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"

)


# ============================================================
# 19. PDF EXTRACTION
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
# 20. PDF ENDPOINT
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

            detail=
                "Only PDF files are allowed."

        )


    file_bytes = (
        await resume.read()
    )


    if not file_bytes:

        raise HTTPException(

            status_code=400,

            detail="PDF is empty."

        )


    try:

        resume_text = (
            extract_pdf_text(
                file_bytes
            )
        )

    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=
                f"PDF extraction failed: {str(e)}"

        )


    if not resume_text.strip():

        raise HTTPException(

            status_code=400,

            detail=
                "Could not extract text from PDF."

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
# 21. HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "AI Career Advisor Agent is running",

        "agent_model":
            "gemma-4-31b-it",

        "final_model":
            "gemini-2.5-flash",

        "tools": [

            "job_search",

            "skill_gap_analysis",

            "project_idea_lookup",

            "github_check"

        ]

    }


# ============================================================
# 22. RUN SERVER
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
