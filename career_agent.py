

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

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain.agents import create_agent

from pydantic import BaseModel, Field

from pypdf import PdfReader


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set"
    )


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
    Search for jobs related to the student's target role.
    Uses the public Remotive jobs API.
    """

    try:

        # Search using the role
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

        # Keywords for AI/ML jobs
        ai_keywords = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml",
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
                job.get("title") or ""
            ).lower()

            description = (
                job.get("description") or ""
            ).lower()

            # Check whether job is relevant
            relevant = any(
                keyword in title
                or keyword in description
                for keyword in ai_keywords
            )

            if relevant:

                jobs.append({

                    "title":
                        job.get("title"),

                    "company":
                        job.get("company_name"),

                    "location":
                        job.get(
                            "candidate_required_location"
                        ),

                    "job_type":
                        job.get("job_type"),

                    "url":
                        job.get("url"),

                    "description":
                        job.get(
                            "description",
                            ""
                        )[:600]
                })

            if len(jobs) >= 10:
                break

        # If filtering returned nothing,
        # return first few search results
        if not jobs:

            for job in all_jobs[:10]:

                jobs.append({

                    "title":
                        job.get("title"),

                    "company":
                        job.get("company_name"),

                    "location":
                        job.get(
                            "candidate_required_location"
                        ),

                    "job_type":
                        job.get("job_type"),

                    "url":
                        job.get("url")
                })

        return json.dumps(
            {
                "target_role": role,
                "jobs_found": len(jobs),
                "jobs": jobs
            },
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "error":
                f"Job search failed: {str(e)}"
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
    Compare student's resume skills with
    the requirements of the target role.
    """

    prompt = f"""
You are an expert AI/ML career advisor.

Analyze the student's resume for the target role.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text}

Provide a concise but useful analysis with:

1. Existing skills
2. Matching skills
3. Missing skills
4. Skills that need improvement
5. Technologies to learn
6. Priority:
   High / Medium / Low
7. Recommended learning roadmap

Focus on practical skills for a college student
or fresher.

Do not invent information about the student.
"""

    try:

        response = llm.invoke(prompt)

        return extract_llm_text(
            response
        )

    except Exception as e:

        return (
            "Skill gap analysis failed: "
            + str(e)
        )


# ============================================================
# 5. PROJECT IDEA TOOL
# ============================================================

@tool
def project_idea_lookup(
    role: str,
    skills: str
) -> str:
    """
    Find relevant GitHub projects for the
    student's target role and skills.
    """

    try:

        query = (
            f"{role} {skills}"
        )

        url = (
            "https://api.github.com/"
            "search/repositories"
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
                    repo.get("name"),

                "description":
                    repo.get("description"),

                "language":
                    repo.get("language"),

                "stars":
                    repo.get(
                        "stargazers_count"
                    ),

                "forks":
                    repo.get(
                        "forks_count"
                    ),

                "url":
                    repo.get("html_url")
            })

        return json.dumps(

            {
                "target_role":
                    role,

                "projects_found":
                    len(projects),

                "projects":
                    projects
            },

            indent=2
        )

    except Exception as e:

        return json.dumps({

            "error":
                f"Project lookup failed: {str(e)}"

        })


# ============================================================
# 6. GITHUB CHECK TOOL
# ============================================================

@tool
def github_check(github_id: str) -> str:
    """
    Check a student's public GitHub profile,
    repositories and activity.
    """

    # --------------------------------------------------------
    # Clean GitHub username
    # --------------------------------------------------------

    username = github_id.strip()

    if "github.com/" in username:

        username = username.split(
            "github.com/",
            1
        )[1]

    username = username.strip("/")

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
        # GitHub Profile
        # ----------------------------------------------------

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
                    f"was not found.",

                "suggestion":
                    "Enter the exact GitHub username."

            })

        user_response.raise_for_status()

        user = user_response.json()

        # ----------------------------------------------------
        # Repositories
        # ----------------------------------------------------

        repo_url = (

            f"https://api.github.com/"
            f"users/{username}/repos"

        )

        repo_response = requests.get(

            repo_url,

            headers=headers,

            params={

                "sort":
                    "updated",

                "per_page":
                    10

            },

            timeout=20
        )

        repo_response.raise_for_status()

        repositories = repo_response.json()

        repos = []

        for repo in repositories:

            repos.append({

                "name":
                    repo.get("name"),

                "description":
                    repo.get("description"),

                "language":
                    repo.get("language"),

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

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

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

            "error":
                f"GitHub check failed: {str(e)}"

        })


# ============================================================
# 7. REGISTER TOOLS
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

- Resume
- Target role
- GitHub username

Your available tools are:

1. job_search
2. skill_gap_analysis
3. project_idea_lookup
4. github_check

IMPORTANT:

Use the tools to collect real information.

For a complete career analysis, use all four tools.

After the tools have returned their results,
you MUST provide a final career analysis.

The final response MUST contain:

1. Career suitability
2. Matching skills
3. Skill gaps
4. Recommended skills to learn
5. Job opportunities
6. Recommended projects
7. GitHub analysis
8. 30/60/90 day roadmap
9. Final recommendation

Do not invent GitHub information.

Do not invent job listings.

If a tool fails, mention that it failed.

Always finish with a clear final answer
for the student.
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
# 10. EXTRACT TEXT FROM GEMINI
# ============================================================

def extract_llm_text(response):

    if response is None:

        return ""

    content = getattr(
        response,
        "content",
        None
    )

    if content is None:

        return ""

    # Normal string
    if isinstance(
        content,
        str
    ):

        return content.strip()

    # Gemini content list
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
# 11. FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(x):

    if isinstance(
        x,
        dict
    ):

        resume_text = x.get(
            "resume_text",
            ""
        )

        role = x.get(
            "role",
            ""
        )

        github_id = x.get(
            "github_id",
            ""
        )

    else:

        resume_text = (
            x.resume_text
        )

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

Use these tools:

1. Job search
2. Skill gap analysis
3. Project recommendations
4. GitHub analysis

After using the tools, provide a final
career recommendation.
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
# 12. FINAL RESPONSE EXTRACTION
# ============================================================

def extract_text_response(
    agent_output
):

    if agent_output is None:

        return "No response generated."

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

    if not messages:

        return "No response generated."


    # --------------------------------------------------------
    # First look for a normal final AI response
    # --------------------------------------------------------

    for message in reversed(
        messages
    ):

        content = getattr(
            message,
            "content",
            None
        )

        if isinstance(
            content,
            str
        ):

            if content.strip():

                # Ignore raw tool-call-only messages
                if not getattr(
                    message,
                    "tool_calls",
                    None
                ):

                    return content.strip()


        if isinstance(
            content,
            list
        ):

            text_parts = []

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
                            text_parts.append(
                                text
                            )

            final_text = "\n".join(
                text_parts
            ).strip()

            if final_text:

                return final_text


    # --------------------------------------------------------
    # No final text found
    # Return tool information for fallback synthesis
    # --------------------------------------------------------

    tool_results = []

    for message in messages:

        content = getattr(
            message,
            "content",
            None
        )

        message_type = (
            message.__class__.__name__
        )

        if (
            "ToolMessage"
            in message_type
        ):

            if content:

                tool_results.append(
                    str(content)
                )

    if tool_results:

        return create_final_synthesis(
            tool_results
        )


    return (
        "The agent completed but "
        "did not return a final response."
    )


# ============================================================
# 13. FINAL SYNTHESIS FALLBACK
# ============================================================

def create_final_synthesis(
    tool_results
):

    combined_results = "\n\n".join(
        tool_results
    )

    prompt = f"""
You are the final career advisor.

The tools have returned the following information:

{combined_results}

Create the FINAL career analysis.

Use this exact structure:

# Career Analysis

## 1. Career Suitability

## 2. Matching Skills

## 3. Skill Gaps

## 4. Recommended Skills

## 5. Job Opportunities

## 6. Recommended Projects

## 7. GitHub Analysis

## 8. 30/60/90 Day Roadmap

## 9. Final Recommendation

Use only information present in the
tool results and student information.

Do not invent facts.

Keep the answer clear and useful for a student.
"""

    try:

        response = llm.invoke(
            prompt
        )

        text = extract_llm_text(
            response
        )

        if text:

            return text

    except Exception as e:

        return (
            "Final synthesis failed: "
            + str(e)
            + "\n\nTool results:\n"
            + combined_results
        )

    return (
        "Final synthesis could not be generated."
        "\n\nTool results:\n"
        + combined_results
    )


# ============================================================
# 14. AGENT CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 15. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=
        "AI Career Advisor Agent",

    description=
        "AI Career Agent using Gemma 4, "
        "LangChain and FastAPI",

    version="1.0.0"

)


# ============================================================
# 16. CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# 17. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"

)


# ============================================================
# 18. PDF TEXT EXTRACTION
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
# 19. PDF ANALYSIS ENDPOINT
# ============================================================

@app.post(
    "/analyze-pdf"
)
async def analyze_pdf(

    resume: UploadFile = File(...),

    role: str = Form(...),

    github_id: str = Form(...)

):

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(

            status_code=400,

            detail=
                "Only PDF files are allowed."

        )


    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    file_bytes = await resume.read()

    if not file_bytes:

        raise HTTPException(

            status_code=400,

            detail="Uploaded PDF is empty."

        )


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

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
                f"PDF reading failed: {str(e)}"

        )


    if not resume_text.strip():

        raise HTTPException(

            status_code=400,

            detail=
                "Could not extract text "
                "from the PDF."

        )


    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    result = (
        await formatted_agent_chain.ainvoke(

            {

                "resume_text":
                    resume_text,

                "role":
                    role,

                "github_id":
                    github_id

            }

        )
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

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
# 20. HOME ROUTE
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
# 21. RUN SERVER
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
