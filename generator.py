import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

SYSTEM_PROMPT = """"""

load_dotenv()

def format_chunks(docs, metas):
    formatted = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        block=f"""
[Rule {i+1}]
Title:    {meta['title']}
Section:  {meta['section']}
Severity: {meta['severity']}

{doc}
""".strip()
        formatted.append(block)

    return "\n\n---\n\n".join(formatted)

def direct_answer(docs, metas):
    doc = docs[0]
    meta = metas[0]
    return {
        "answer": doc,
        "source": meta["title"],
        "rule_ids":  [meta["rule_id"]],
        "method": "direct",       
        "used_llm": False,
        "caveat": "Please verify this with your academic advisor."  if meta["severity"] == "high" else None
    }
    

def needs_llm(docs, metas, rrf_scores):
    if len(docs)==0:
        return False

    top_score = rrf_scores[0] if rrf_scores else 0

    if len(docs) == 1 and top_score>0.9:
        return False

    if len(docs)>1:
        return True
    
    if top_score<0.60:
        return True

    return False


def call_llm(question, docs, metas):
    client  = genai.Client()
    context = format_chunks(docs, metas)

    user_message = f"""
    Here are the relevant university rules:

    {context}

    Student's question: {question}
    """

    # response = client.messages.create(
    #     model      = "claude-3-5-haiku-20241022",
    #     max_tokens = 512,
    #     system     = SYSTEM_PROMPT,
    #     messages   = [{"role": "user", "content": user_message}]
    # )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=512,
        )
    )

    answer    = response.text
    rule_ids  = [m["rule_id"]  for m in metas]
    titles    = [m["title"]    for m in metas]
    severities= [m["severity"] for m in metas]

    return {
        "answer":   answer,
        "source":   titles,
        "rule_ids": rule_ids,
        "method":   "llm",
        "used_llm": True,
        "caveat":   "Please verify this with your academic advisor."
                    if "high" in severities else None
    }

def generate(question, docs, metas, rrf_scores):
    if not docs:
        return {
            "answer":   "I couldn't find a rule covering this. Please contact the Registrar's office.",
            "source":   None,
            "rule_ids": [],
            "method":   "no_result",
            "used_llm": False,
            "caveat":   None
        }

    if needs_llm(docs, metas, rrf_scores):
        return call_llm(question, docs, metas)
    else:
        return direct_answer(docs, metas)