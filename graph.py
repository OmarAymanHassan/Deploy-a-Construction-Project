import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Construction"

from langgraph.graph import START, END ,StateGraph
from typing import TypedDict,Optional,Dict,Any,Literal,List
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field,BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch



gemini_model =ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=1 , api_key=os.getenv("gemini_api_key"))


## StateAgent

class ConstructionInput(TypedDict):
    extraction_info: str # Extraaction from the description
    output:Optional[str] # the result of extraction
    company_info:Optional[str] # the company info from tavily
    company_name:str 
    all_company_content:str # all the content gathered by tavily
    summarized_content:str # after summarization all the tavily content
    links:Optional[list[str]]
    final_extraction_with_score:str



# First Node Class

class ScopeCoverage(BaseModel):
    #included:list[Literal["HVC","Electrical","Interior"]]
    included:list[str]
    excluded:list[str]
    sub_contracted: list[str]

class CostDetails(BaseModel):
    target_budget_usd:str
    overrun_percentage:int
    hard_stop_usd:str


class TimelineDetails(BaseModel):
    target_completion_months:int
    acceptable_extension_weeks:int
    delay_penalties:str


    
class ExtractionInfo(BaseModel):
    Company_name: str = Field(description="Company name")
    cost: CostDetails = Field(description="Cost details")
    # Dict[key,value] --> i want keys in str format, and values in `Any` format
    timeline: TimelineDetails = Field(description="Project timeline")
    prior_similar_projects_count:int = Field(description="Number of prior similar projects")
    scope_coverage:ScopeCoverage = Field(description="The project coverage materials, choose only HVC, Electrical or Interior")
    legal_and_compliance:str =Field(description="Legal and Compliances that included in the project")



# First Node

def extract_construction_info(state: ConstructionInput):
    
    prompt = "\n".join([
        "You are an expert construction project manager and requirements analyst.",
        "Your task is to EXTRACT structured data from the input.",
        "",
        "STRICT RULES (MANDATORY):",
        "- You MUST return VALID JSON only.",
        "- Do NOT include explanations, summaries, or text outside JSON.",
        "- Every field must match the required type exactly.",
        "- If information is missing or unclear, return an empty object {} or null.",
        "",
        "DATA TYPE RULES:",
        "- cost MUST be a JSON object (dictionary), never a string.",
        "- timeline MUST be a JSON object (dictionary), never a string.",
        "- The keys inside cost and timeline may vary depending on the input.",
        "- Values inside cost and timeline may be numbers, strings, or nested objects.",
        "",
        "EXAMPLES:",
        "If the input says: '$1.2M target, $1.35M hard stop'",
        "Return:",
        '{ "cost": { "target": 1200000, "hard_stop": 1350000 } }',
        "",
        "If the input provides cost in free text and exact values are unclear:",
        '{ "cost": { "raw": "original text here" } }',
        "",
        "Your Responsibilities:",
        "1- Extract structured requirements",
        "2- Identify constraints and priorities",
        "3- Convert budget and timeline information into JSON objects",
        "4- in ScopeCoverage, summarize each sentence to be only either one of those [HVC,Interior,Electrical]"
        "",
        "INPUT:",
        f"{state['extraction_info']}",
        "",
        "OUTPUT JSON ONLY:"
    ])

    response = gemini_model.with_structured_output(ExtractionInfo).invoke(prompt)
    print("Gemini Response:", response)

#    return {"output": response}
    return {"output": response}




# 2nd Node class

class SearchCompanyInfo(BaseModel):
    company_name: str = Field(description="Company name")
    cost: str = Field(description="Cost details")
    timeline: str = Field(description="Project timeline")
    number_of_same_projects: int = Field(description="Number of prior similar projects")
    is_relevant: int = Field(description="How strong this company Is relevant to the Project, range from 0 to 10" ,ge=0 , le=10)


# Tavily

tavily = TavilySearch(max_results=5,search_depth="basic" , api_key=os.getenv("TAVILY_API_KEY"),include_raw_content="markdown",include_image_descriptions=False,include_images=False)


# 2nd Node 

def search_company_info(state: ConstructionInput) -> SearchCompanyInfo:
    company_name = state["company_name"]
    search_result = tavily.invoke(company_name)
    # Process search_result to extract required fields
    search_contents = [item['content'] for item in search_result['results']]
    search_score = [item['score'] for item in search_result['results']]

    prompt = "\n".join([
        "You are an expert construction project analyst.",
        "Based on the following search results, extract the company's average cost, average timeline, number of similar projects, and relevance to the given project.",
        "DONT EVER ASSUME ANYTHING, IF YOU DONT FIND THE INFO , RETURN NULL or 0",
        "",
        "## SEARCH RESULTS:",
        f"{search_contents}",
        "## SEARCH SCORES For each result:",
        f"{search_score}",
        "",
        "OUTPUT FORMAT (JSON):",
        "{",
        '  "company_name": str,',
        '  "avg_cost": str,',
        '  "avg_timeline": str,',
        '  "number_of_same_projects": int,',
        '  "is_relevant": bool',
        "}"
    ])

    response = gemini_model.with_structured_output(SearchCompanyInfo).invoke(prompt)
    print("Gemini Company Info Response:", response)
    return {"company_info": response}


# 3rd Node

def search_company_info(state: ConstructionInput):
    company_name = state["company_name"]
    search_result = tavily.invoke(company_name)
    # Process search_result to extract required fields
    search_raw_contents = [item['raw_content'] for item in search_result['results']]
    search_url = [item["url"] for item in search_result["results"]]
    print(f"Links provided : \n\n{search_url}")
    
    return {"all_company_content": search_raw_contents , "links": search_url}




def summarized_company_content(state:ConstructionInput):
    all_content = state['all_company_content']
    prompt = "\n".join([
        "You are an expert construction project analyst.",
        "You have collected multiple pieces of information about a company.",
        "Your task: Summarize all the content in detail, without skipping any important information.",
        "Specifically, include:",
        "1. How many projects the company has completed related to the current project, with details if available.",
        "2. Any red flags in the company's history or performance or timeline alignment.",
        "3. Whether the company aligns with the protocols and requirements in the project details.",
        "4. Any negative reports, bad news, or reputational concerns.",
        "5. Any other observations that could impact the project decision.",
        "Write a concise but complete plain text summary, combining all information from the collected content.",
        "Be explicit and do not leave out anything important.",
        "### Input:",
        f"{all_content}",
        "### Output:"
    ])

    response = gemini_model.invoke((prompt))
    print(f"Summarized Content : {response.content}")
    print("--- Summary Node is Done Successfully ---")
    return {"summarized_content":response.content}


# 4th node class
class KeySignals(BaseModel):
    us_commercial_experience: bool = Field(description="Has US commercial experience")
    project_scale_alignment: str = Field(description="Alignment of project scale: Low/Medium/High")
    recent_negative_news: bool = Field(description="Any recent negative news or bad reputation")

class Confidence(BaseModel):
    overall_confidence:float = Field(description="Overall confidence in the entire extraction")
    explanation:str=Field(description="on what basis you put this confidence")

class ExternalCompanyInsights(BaseModel):
    company_name:str=Field(description="Company Name")
    sources: List[str] = Field(description="List of URLs or references")
    external_company_insights: KeySignals = Field(description="Structured insights per company")
    overall_confidence: Confidence = Field(description="Overall confidence in the entire extraction")



# 4th node
def summary_extractor_evaluator(state:ConstructionInput):


    """prompt = "\n".join([
        "You are an expert construction project analyst.",
        "You are given:",
        "1) Project requirements and protocols",
        "2) External information about companies (history, completed projects, reputation, news, public signals)",
        "",
        "Your task is to evaluate how well each company aligns with the project requirements by comparing the two sources.",
        "",
        "Evaluation instructions:",
        "- Compare each company's past projects, construction scope, and technical experience against the project requirements.",
        "- Identify alignment, partial alignment, gaps, inconsistencies, or red flags.",
        "- Consider reputation signals such as negative news, disputes, failures, or regulatory issues if present.",
        "- Base all judgments strictly and only on the provided content.",
        "",
        "Confidence interpretation rules (VERY IMPORTANT):",
        "- You MUST compute an overall_confidence score between 0 and 1.",
        "- The confidence score MUST be justified using an explicit explanation field.",
        "- The confidence score should be based on the following factors combined:",
        "  1) Strength of prior experience: relevance and similarity of past projects to the current project.",
        "  2) Evidence quality: clarity, consistency, and credibility of the provided information and sources.",
        "  3) Reputation signals: presence or absence of negative news, disputes, failures, or red flags.",
        "  4) Alignment completeness: how fully the company aligns with the stated project protocols and scale.",
        "",
        "Scoring guidance (implicit, do not output these rules):",
        "- High confidence (close to 1): strong relevant experience, clear alignment, reputable track record, no major red flags.",
        "- Medium confidence (~0.4–0.7): partial alignment, limited evidence, mixed or unclear signals.",
        "- Low confidence (close to 0): weak or irrelevant experience, poor evidence quality, or significant negative signals.",
        "",
        "For each company, extract structured insights in JSON with the following fields:",
        "- company_name",
        "- sources: URLs or references used",
        "- external_company_insights:",
        "    - us_commercial_experience: true if evidence exists, otherwise false",
        "    - project_scale_alignment: 'Low', 'Medium', or 'High' based on comparison with project requirements",
        "    - recent_negative_news: true if any bad news, disputes, or reputational risks are present",
        "- overall_confidence:",
        "    - overall_confidence: float (0–1)",
        "    - explanation: a concise explanation describing EXACTLY why this confidence score was assigned, referencing experience, reputation, evidence strength, and alignment",
        "",
        "IMPORTANT RULES:",
        "- Output VALID JSON ONLY.",
        "- Do NOT invent information.",
        "- If evidence is missing or weak, reflect that directly in both the confidence score and its explanation.",
        "",
        "## Project Requirements and Protocols:",
        f"{state['extraction_info']}",
        "",
        "## External Company Information:",
        f"{state['summarized_content']}",
        "",
        "## Sources:",
        f"{state['links']}"
    ])
    """

    prompt = "\n".join([
    "You are an expert construction project evaluator.",
    "You must act as a STRICT, LOGICAL scoring engine — not a subjective reviewer.",
    "",
    "You are given:",
    "1) Project requirements and protocols",
    "2) External information about companies (projects, scale, geography, reputation, news)",
    "",
    "Your task is to EVALUATE each company using an EXPLICIT, STEP-BY-STEP scoring process.",
    "",
    "IMPORTANT: You MUST calculate the score first, then explain it.",
    "",
    "====================",
    "MANDATORY SCORING FRAMEWORK",
    "====================",
    "",
    "You MUST compute the overall confidence score as a weighted sum of the following components:",
    "",
    "1) Relevant Project Experience (weight = 0.40)",
    "- Score based on similarity of completed projects to the current project scope.",
    "- Larger or more complex projects COUNT POSITIVELY for smaller or mid-scale projects.",
    "- NEVER penalize a company for having handled larger-scale projects.",
    "",
    "2) Capability vs Scale Fit (weight = 0.25)",
    "- High score if company demonstrates capability >= required scale.",
    "- Medium score if capability is unclear but plausible.",
    "- Low score ONLY if company has ONLY handled significantly smaller projects.",
    "",
    "3) Evidence Quality (weight = 0.20)",
    "- High if information is clear, consistent, and sourced.",
    "- Medium if partial or indirect.",
    "- Low if vague, promotional, or weak.",
    "",
    "4) Reputation Impact (weight = 0.15 — CONDITIONAL)",
    "- FIRST classify any negative news:",
    "  a) Operational / Legal / Strategic impact",
    "  b) Contextually relevant to this project or geography",
    "",
    "- IF negative news is NOT operationally or strategically relevant → score = 1.0 (NO penalty).",
    "- IF relevant and impactful → apply a proportional reduction.",
    "",
    "====================",
    "STEP-BY-STEP PROCESS (MANDATORY)",
    "====================",
    "",
    "For EACH company you MUST:",
    "1) Assign a score between 0 and 1 for EACH component.",
    "2) Multiply each component by its weight.",
    "3) Sum the weighted values to compute overall_confidence.",
    "4) Explain EXACTLY why each component received its score.",
    "",
    "====================",
    "OUTPUT FORMAT (STRICT)",
    "====================",
    "",
    "Return VALID JSON ONLY with this structure:",
    "",
    "{",
    "  company_name: string,",
    "  sources: list,",
    "  scoring_breakdown: {",
    "    experience_score: float,",
    "    scale_fit_score: float,",
    "    evidence_quality_score: float,",
    "    reputation_impact_score: float",
    "  },",
    "  weighted_calculation: {",
    "    experience_weighted: float,",
    "    scale_fit_weighted: float,",
    "    evidence_quality_weighted: float,",
    "    reputation_impact_weighted: float",
    "  },",
    "  overall_confidence: float,",
    "  explanation: string",
    "}",
    "",
    "IMPORTANT RULES:",
    "- Do NOT invent information.",
    "- Do NOT apply penalties without explaining WHY the factor is relevant.",
    "- Larger-scale experience MUST be treated as positive capability evidence.",
    "- If evidence is missing, explicitly state uncertainty.",
    "",
    "## Project Requirements:",
    f"{state['extraction_info']}",
    "",
    "## External Company Information:",
    f"{state['summarized_content']}",
    "",
    "## Sources:",
    f"{state['links']}"
])



    response = gemini_model.with_structured_output(ExternalCompanyInsights).invoke(prompt)

    return {"final_extraction_with_score": response}


    
#Graph

graph = StateGraph(ConstructionInput)
graph.add_node("Extract Construction Info",extract_construction_info)
graph.add_node("Search Company Info",search_company_info)
graph.add_node("summary_company_info" , summarized_company_content)
graph.add_node("company_evaluation" ,summary_extractor_evaluator)

graph.add_edge(START, "Extract Construction Info")
graph.add_edge(START, "Search Company Info")
graph.add_edge("Search Company Info" , "summary_company_info")
graph.add_edge("summary_company_info","company_evaluation")
graph.add_edge("Extract Construction Info","summary_company_info")
graph.add_edge("company_evaluation", END)
graph = graph.compile()

