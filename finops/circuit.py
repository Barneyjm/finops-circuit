"""The tags and the gates. This file is the whole policy.

Spend on LLM calls is allocated the way cloud spend is: by tags. Each conversation gets

    app          which program sent it (set in code, from the prompt template; see tags.py)
    workload     interactive (a person) or automated (a program)
    environment  production or dev/test
    task         what kind of work, then subtask: which kind of that work (two stages)
    domain       the business area it serves
    data_class   how sensitive what it carries is

A model answers the tag questions with probabilities; a tag the circuit is not sure of comes
out "untagged", as an unlabelled resource does in a cloud bill, so tag coverage is a number
the report can give. The same request asks what a cost review needs on top: is the work hard,
would a small model do, and is it a request many people make. Nothing here generates text.
"""

from __future__ import annotations

from decision_circuits import Circuit, Q, argmax

UNTAGGED = "untagged"

TASKS = {
    "code": "Writing, fixing, explaining or reviewing software, including queries and configs",
    "writing": "Drafting or editing prose for a reader: emails, posts, documents, copy",
    "summarization": "Condensing given text: documents, articles, meetings, threads",
    "translation": "Turning text from one language into another",
    "extraction": "Pulling specific items out of given text: keywords, entities, fields, tables",
    "classification": "Labelling given text: sentiment, topic, intent, moderation",
    "data_analysis": "Working with numbers or data: math, statistics, spreadsheets, cleaning data",
    "research": "Explaining a subject, comparing options, answering a factual question",
    "advice": "Guidance on a person's own situation: health, money, relationships, decisions",
    "creative": "Stories, role-play, poems, jokes, games",
    "chit_chat": "Greetings and small talk with no task",
    "other": "None of the above",
}

SUBTASKS = {
    "code": {
        "generate": "New code",
        "debug": "Fix code that fails",
        "explain": "Explain what code does",
        "review": "Improve or critique working code",
        "convert": "Port between languages or frameworks",
        "sql": "Database queries",
    },
    "writing": {
        "email_message": "An email or chat message",
        "marketing_copy": "Ads, product descriptions, landing pages",
        "social_post": "A post for a social platform",
        "document": "A report, essay, article or proposal",
        "rewrite": "Rephrase, shorten or correct existing text",
        "job_application": "A resume, cover letter or interview answer",
    },
    "summarization": {"document": "A long document or report", "article": "An article or web page", "conversation": "A meeting, call or chat", "list_notes": "Notes into action items or bullets"},
    "translation": {"general": "Everyday text", "technical": "Technical or legal text", "localization": "Product text adapted for a market"},
    "extraction": {
        "keywords": "Keywords or search terms",
        "entities": "Names, places, dates, amounts",
        "structured": "Fields into JSON or a table",
        "answer_span": "The part of a text that answers a question",
    },
    "classification": {"sentiment": "Positive, negative or neutral", "topic": "What it is about", "intent": "What the writer wants", "moderation": "Whether it breaks a rule"},
    "data_analysis": {
        "math_problem": "Solve a math problem",
        "statistics": "Statistics or probability",
        "spreadsheet": "Spreadsheet formulas or tables",
        "data_processing": "Clean, reshape or query a dataset",
    },
    "research": {"explanation": "Explain a concept or field", "comparison": "Compare options", "fact_lookup": "A specific fact", "recommendation": "Which option to choose"},
    "advice": {"health": "Physical or mental health", "finance": "Personal money", "relationships": "Family, friends, partners", "career": "Jobs and careers", "legal": "A person's legal situation"},
    "creative": {"story": "A story or scene", "roleplay": "Playing a character with the user", "poetry": "Poems or lyrics", "games_jokes": "Games, riddles, jokes"},
    "chit_chat": {"greeting": "Hello and how are you", "about_the_ai": "Questions about the assistant itself", "banter": "Casual conversation"},
    "other": {"other": "Anything else"},
}

DOMAINS = {
    "software": "Software, data and IT",
    "marketing_sales": "Marketing, advertising and sales",
    "customer_support": "Serving a company's customers",
    "finance": "Accounting, investing, banking",
    "legal": "Law, contracts, compliance",
    "hr_people": "Hiring, workplace, careers",
    "education": "Studying, teaching, homework",
    "health": "Medicine and wellbeing",
    "operations": "Running the business: planning, vendors, admin",
    "science_engineering": "Science and engineering outside software",
    "media_entertainment": "Fiction, games, media",
    "personal_life": "Everyday personal matters",
    "other": "None of these",
}

ENVIRONMENTS = {
    "production": "Doing a real job, by a person or by a program: the output is meant to be used, however templated",
    "dev_test": "The content itself is a test: 'test', placeholder or dummy input, the same prompt tried several ways, attempts to get round the model's rules",
}

WORKLOADS = {
    "interactive": "A person typing and reading the replies",
    "automated": "A program sending templated input: fixed instructions around swapped-in content, a required output format",
}

DATA_CLASSES = {
    "public": "Nothing that is not already public",
    "internal": "Ordinary business content not meant for outsiders",
    "confidential": "Business secrets, contracts, credentials, private documents",
    "regulated": "Personal data about people, health, payment or financial account details",
}

COMPLEXITY = [
    "Trivial: a lookup, a greeting, a one-line rewrite",
    "Routine: a standard email, a short explanation, a simple function",
    "Involved: several steps, a long document, code that has to fit a codebase",
    "Hard: careful multi-step reasoning, subtle bugs, expert judgment where errors are costly",
]

TAG_QUESTIONS = {"task": TASKS, "domain": DOMAINS, "environment": ENVIRONMENTS, "workload": WORKLOADS, "data_class": DATA_CLASSES}
MIN_TAG_CONFIDENCE = 0.2  # normalized: 1 - entropy / log(options); below it a tag is left untagged


def build_circuit(v2: bool = False) -> Circuit:
    """Stage one: every tag but the subtask, and the cost questions. `v2` adds `purpose` (locate:
    the line that shows what the conversation was for) for circuit v2 models."""
    c = Circuit()
    c.choice("task", "What kind of work was the assistant asked to do? Pick the single best fit.", TASKS)
    c.choice("domain", "Which area does this conversation serve?", DOMAINS)
    c.choice("environment", "Is this doing a real job, or is the content itself a test?", ENVIRONMENTS)
    c.choice("workload", "Was a person chatting, or a program sending templated input?", WORKLOADS)
    c.choice("data_class", "How sensitive is the most sensitive thing in this conversation?", DATA_CLASSES)
    c.noul("work", "Is this being done for a job or a business, rather than for the person themselves?", true="For work or a business", false="Personal, school or entertainment")
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
        false="Specific to this situation",
    )
    if v2:
        c.locate("purpose", "Which line shows best what the person was trying to get done?", none="no line makes the purpose clear")

    # ---- tags: the pick, or untagged when the circuit is not sure --------------------
    for key in TAG_QUESTIONS:
        c.gate(key, argmax(key, min_confidence=MIN_TAG_CONFIDENCE), on_uncertain="default", default=UNTAGGED)

    # ---- actions ----------------------------------------------------------------------
    # Business spend or not.
    c.gate("business", Q("work") >= 0.5, band=0.1, on_uncertain="escalate")
    # A cheaper model, only when the work is not hard and a small model would do.
    c.gate("downgrade", (Q("small_model_ok") & ~Q("complexity")[3]) >= 0.65, band=0.1, on_uncertain="default", default=False)
    # Cache or template it when many people ask the same thing.
    c.gate("cache", Q("repeatable") >= 0.7, band=0.1, on_uncertain="default", default=False)
    # Sensitive data: nothing moves models or gets cached without a person looking.
    c.gate("hold", (Q("data_class")["confidential"] | Q("data_class")["regulated"]) >= 0.5, band=0.15, on_uncertain="default", default=True)
    # Testing on the production bill.
    c.gate("dev_test", Q("environment")["dev_test"] >= 0.6, band=0.1, on_uncertain="default", default=False)
    return c


def build_subtask_circuit(task: str) -> Circuit:
    """Stage two, for a conversation whose task was tagged: which kind of that task."""
    c = Circuit()
    c.choice("subtask", f"This conversation is {task.replace('_', ' ')} work. Which kind?", SUBTASKS[task])
    c.gate("subtask", argmax("subtask", min_confidence=MIN_TAG_CONFIDENCE), on_uncertain="default", default=UNTAGGED)
    return c
