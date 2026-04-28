import json
from google.genai import types
from config import client, MODEL

def clean_facts_for_agents(facts: dict) -> dict:
    
    if not isinstance(facts, dict):
        return facts
 
    cleaned = dict(facts)  # shallow copy
    raw_features = facts.get("csv_features") or {}
 
    cleaned_features = {}
    for key, value in raw_features.items():
        if isinstance(value, str):
            if value and value.lower() not in ("unknown", "other", "n/a", ""):
                cleaned_features[key] = value
            continue
        if value is None:
            continue
        try:
            if float(value) == 0:
                continue
        except (TypeError, ValueError):
            continue
        cleaned_features[key] = value
 
    cleaned["csv_features"] = cleaned_features
    
    all_numeric = ["revenue_million", "revenue_growth_rate", "burn_rate_million",
                   "runway_months", "funding_rounds", "team_size",
                   "founder_experience_years", "has_technical_cofounder",
                   "product_traction_users", "customer_growth_rate",
                   "enterprise_customers", "market_size_billion"]
    cleaned["_features_not_in_deck"] = [
        f for f in all_numeric if f not in cleaned_features
    ]
    return cleaned
 
 
def _call_agent(prompt: str, schema: dict) -> dict:
    """Generic text-in, JSON-out Gemini call."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1
            ),
        )
        return response.parsed
    except Exception as e:
        if "429" in str(e):
            return {"error": "Quota exceeded. Wait ~60 seconds."}
        return {"error": str(e)}

bull_schema = {
    "type": "OBJECT",
    "properties": {
        "green_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["green_flags"],
}


def run_bull(deck_facts: dict, benchmark: dict = None) -> dict:
    print("Bull analyzing...")
    cleaned = clean_facts_for_agents(deck_facts)

    # {f"Prioritize flags backed by ABOVE-benchmark metrics." if benchmark else ""}
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK CONTEXT:\n{json.dumps(benchmark, indent=2)}"
    prompt = f"""
    You are "The Bull" — an optimistic VC analyst.
    Identify 4 GREEN FLAGS. Each: short (4-10 words), specific.

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {bench_block}

    Return JSON with a "green_flags" array of 4 items.
    """
    return _call_agent(prompt, bull_schema)


bear_schema = {
    "type": "OBJECT",
    "properties": {
        "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["red_flags"],
}


def run_bear(deck_facts: dict, benchmark: dict = None) -> dict:
    print("Bear analyzing...")
    cleaned = clean_facts_for_agents(deck_facts)
    # {f"Prioritize flags backed by BELOW-benchmark metrics." if benchmark else ""}
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK CONTEXT:\n{json.dumps(benchmark, indent=2)}"
    prompt = f"""
    You are "The Bear" — a skeptical VC analyst.
    Identify 4 RED FLAGS. Look for: high CAC, unproven monetization,
    crowded market, weak moat, team gaps. Each: short (4-10 words), specific.

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {bench_block}

    Return JSON with a "red_flags" array with 4 items.
    """
    return _call_agent(prompt, bear_schema)


summarizer_schema = {
    "type": "OBJECT",
    "properties": {
        "recommendation": {"type": "STRING"},   # GO | NO-GO | HOLD
        "risk_level":     {"type": "STRING"},   # Low | Medium | High
        "memo":           {"type": "STRING"},   # 2-3 sentences
    },
    "required": ["recommendation", "risk_level", "memo"],
}


def run_summarizer(deck_facts: dict, 
                   bull: dict, 
                   bear: dict,
                   benchmark: dict = None, 
                   ml_result: dict = None,
                   vitality: dict = None,
                   reconciliation: dict = None,
                   quadrant: dict = None) -> dict:
    
    print("Summarizer writing memo...")
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK:\n{json.dumps(benchmark, indent=2)}"

   
    pinned_risk = vitality.get("risk_level") if vitality else None
    risk_constraint = (
        f'\n\nCONSTRAINT: risk_level MUST be exactly "{pinned_risk}". '
        f'This is the data-driven verdict from the Vitality engine '
        f'(ML + peer benchmark + agent balance). Do not override it. '
        f'Your job is to write the memo and pick GO/NO-GO/HOLD — '
        f'risk_level is fixed.'
        if pinned_risk else ""
    )

    ml_block = f"""
    ML SIGNAL:
    - success_probability: {ml_result.get("success_probability")}
    - confidence: {ml_result.get("confidence")}
    """ if ml_result else ""

    vitality_block = f"""
    VITALITY:
    - score: {vitality.get("score")}
    - risk_level: {vitality.get("risk_level")}
    """ if vitality else ""

    quadrant_block = f"""
    QUADRANT:
    - position: {quadrant.get("quadrant")}
    """ if quadrant else ""

    recon_block = f"""
    RECONCILIATION:
    - ML vs agent agreement: {reconciliation.get("agreement")}
    """ if reconciliation else ""

    prompt = f"""
    You are a senior VC partner writing the final investment decision.
    Your job is NOT to summarize, your job is to decide.

    -------------------------
    INPUTS
    -------------------------

    STARTUP FACTS:
    {json.dumps(deck_facts, indent=2)}

    BULL CASE (strengths):
    {json.dumps(bull, indent=2)}

    BEAR CASE (risks):
    {json.dumps(bear, indent=2)}
    {bench_block}
    {risk_constraint}

    {ml_block}
    {vitality_block}
    {quadrant_block}
    {recon_block}



    ------------------------------
    DECISION RULES
    ------------------------------
    1. Weigh strengths vs risks.
    2. Classify risks as:
        - mitigable (acceptable)
        - fundamental (deal-breaker)
    3. Your recommendation must align with your reasoning:
        - GO if strengths clearly outweigh risks and no fundamental issues.
        - NO-GO if fundamental risks outweigh strengths.
        - HOLD if it's a close call or if there are strong mitigations for the risks.
    4. Constraints:
        - A NO-GO cannot be based on benchmark metrics if ML signal and vitality score are strong.
        - If ML and Agent reasoning conflict, favor HOLD and call out the disagreement in the memo.
    
    ----------------
    SIGNAL INTEGRATION
    ----------------
    - ML sucess probability is a strong signal.
        - >70% is a strong positive.
        - <30% is a strong negative.
        - If contradicting ML, be cautious about a GO recommendation.
    - Vitality Score:
        - High vitality should push towards GO.
        - Low vitality should push towards NO-GO.
    - Quadrant Position:
        - High risk + Low return should push towards NO-GO.
        - High Return + High risk should be HOLD or selective GO. 

    --------------------
    OUTPUT
    --------------------

    Return JSON with:
      - recommendation: "GO" | "NO-GO" | "HOLD"
      - risk_level:     "Low" | "Medium" | "High"
      - memo:           2-3 sentences naming the key caveat

    The memo should clearly articulate the main reason for the recommendation, citing specific green and red flags, and how you weighed them. If there is a conflict between ML signal and agent reasoning, call it out in the memo.
    Be concise but specific — avoid generic statements.
    """
    result = _call_agent(prompt, summarizer_schema)

    if pinned_risk and isinstance(result, dict) and "risk_level" in result:
        result["risk_level"] = pinned_risk

    return result