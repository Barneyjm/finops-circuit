"""The questions and the gates. This file is the whole policy.

A FinOps review of LLM spend asks the same few things of every conversation: was this work for
the business, what kind, could a cheaper model have done it, and is it the same request other
people make. A model answers those with probabilities; the gates below turn them into the
actions a cost report recommends. Nothing here generates text.
"""

from __future__ import annotations

from decision_circuits import Circuit, Q, argmax

VALUE_TYPES = {
    "customer_facing": "Work a customer sees or pays for: replies to customers, product copy, content that ships",
    "internal_productivity": "Everyday work for the job: emails, documents, summaries, planning, spreadsheets",
    "engineering": "Building or fixing software, data or infrastructure: code, queries, configs, debugging",
    "research": "Learning or analysis for work: explaining a field, comparing options, digging into a question",
    "personal": "Not work: homework, hobbies, stories, role-play, chat for its own sake",
    "waste": "No real use: tests, empty or repeated prompts, attempts to break the model's rules, spam",
}

FUNCTIONS = {
    "sales_marketing": "Selling, advertising, positioning, outreach",
    "support": "Helping customers with a product or an order",
    "engineering": "Software, data, infrastructure",
    "finance_legal": "Money, contracts, compliance, tax",
    "operations_hr": "Running the business: hiring, logistics, policy, admin",
    "product_design": "Deciding what to build and how it looks",
}

COMPLEXITY = [
    "Trivial: a lookup, a greeting, a one-line rewrite",
    "Routine: a standard email, a short explanation, a simple function",
    "Involved: several steps, a long document, code that has to fit a codebase",
    "Hard: careful multi-step reasoning, subtle bugs, expert judgment where errors are costly",
]


def build_circuit(v2: bool = False) -> Circuit:
    """`v2` adds the questions only circuit v2 models answer (circuit-1.7b v2.0 and later): every
    business function the conversation touches, and the line that shows what it was for."""
    c = Circuit()

    # ---- what the model is asked, all in one request -------------------------------
    c.choice("value_type", "What was this conversation for? Pick the single best fit.", VALUE_TYPES)
    c.noul(
        "work",
        "Is the person using the assistant for their job or a business, rather than for themselves?",
        true="Doing work for an employer, clients or their own business",
        false="Personal use, school, entertainment, or no real purpose",
    )
    c.score("complexity", "How hard is what the assistant was asked to do?", COMPLEXITY)
    c.noul(
        "small_model_ok",
        "Could a small, cheap language model have produced answers just as good here?",
        true="Yes: common knowledge, simple writing or simple code",
        false="No: it needs strong reasoning, niche expertise or long careful output",
    )
    c.noul(
        "repeatable",
        "Is this a request many people would make in nearly the same words, so a template or a cached answer would serve it?",
        true="A common, standard request",
        false="Specific to this person's situation",
    )
    c.noul(
        "sensitive",
        "Does the conversation contain personal, confidential or regulated data (names with details, credentials, health, finances, internal documents)?",
        true="Contains such data",
        false="Nothing sensitive",
    )
    if v2:
        c.multi("functions", "Which business functions does this conversation serve? Mark every one that applies.", FUNCTIONS)
        c.locate("purpose", "Which line shows best what the person was trying to get done?", none="no line makes the purpose clear")

    # ---- what the code decides ------------------------------------------------------
    # The value bucket the spend is reported under; unsure means a person classifies it.
    c.gate("value", argmax("value_type", min_confidence=0.25), on_uncertain="escalate")
    # Business spend or not: the bucket's own number, checked by a second question.
    c.gate("business", (Q("work") & ~(Q("value_type")["personal"] | Q("value_type")["waste"])) >= 0.5, band=0.1, on_uncertain="escalate")
    # Recommend a cheaper model only when the work is not hard and a small model would do.
    c.gate("downgrade", (Q("small_model_ok") & ~Q("complexity")[3]) >= 0.65, band=0.1, on_uncertain="default", default=False)
    # Cache or template it when many people ask the same thing.
    c.gate("cache", Q("repeatable") >= 0.7, band=0.1, on_uncertain="default", default=False)
    # Spend with no business use: a policy question, not a model question.
    c.gate("policy", (Q("value_type")["waste"] | (Q("value_type")["personal"] & ~Q("work"))) >= 0.6, band=0.1, on_uncertain="default", default=False)
    # Sensitive data: never move it to another model or cache it without a person looking.
    c.gate("hold", Q("sensitive") >= 0.5, band=0.15, on_uncertain="default", default=True)
    return c
