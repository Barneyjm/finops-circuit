# finops-circuit

Tag LLM spend the way cloud spend is tagged, then say what to change. Built on
[decision circuits](https://github.com/Barneyjm/decision-circuits).

Each conversation gets allocation tags, `app`, `workload`, `environment`, `task` and `subtask`,
`domain`, `data_class`, and a cost line from its tokens. A decision model answers the tag
questions with probabilities; plain code turns them into tags, a tag it is not sure of comes
out `untagged`, and the report groups spend by any tags the way a cloud bill is grouped.

```
$ finops analyze samples --backend fake

== 11_keyword_app.json  gpt-4o-2024-08-06, 1 replies, $0.0003  (44 in / 20 out)
   Provide only relevant keywords to facilitate an online search for the product below. Return a comma-separated
-> app=app-fb9384 workload=automated environment=production task=extraction subtask=keywords domain=marketing_sales data_class=public
   business=True saves $0.0003 | small model ok 0.95 | repeatable 0.90
   * downgrade to gpt-4o-mini
   * cache or template: a common request
   task         decided   task -> extraction p=0.85 conf=0.69 (min 0.2)
   subtask      decided   subtask -> keywords p=0.85 conf=0.58 (min 0.2)
```

## Run it

```bash
git clone https://github.com/Barneyjm/finops-circuit && cd finops-circuit
uv sync
cp .env.example .env                                   # one key for the backend you pick
uv run finops fetch --n 200                            # a WildChat-4.8M sample into data/
uv run finops report data/wildchat.jsonl --backend jev --by task,subtask
```

| | |
|---|---|
| `finops fetch --n N` | N real conversations from WildChat-4.8M into `data/wildchat.jsonl` |
| `finops analyze <file or dir>` | per conversation: tags, cost, actions, the gate traces |
| `finops report <file or dir> --by k1,k2` | spend grouped by tag keys, tag coverage, actions, savings |
| `finops diagram` | the circuit as Mermaid |

`--backend` picks the model, as in [call-center-circuit](https://github.com/Barneyjm/call-center-circuit):
`jev`, `circuits`, `local`, `semif`, `openai`, `anthropic`, or `fake` (hand-written answers in
`samples/`, no network).

## The tags

| tag | set by | values |
|---|---|---|
| `app` | code: the prompt template (below) | `app-<hex>`, or `adhoc` for people typing |
| `workload` | the circuit | `interactive`, `automated` |
| `environment` | the circuit | `production`, `dev_test` |
| `task` | the circuit, stage one | code, writing, summarization, translation, extraction, classification, data_analysis, research, advice, creative, chit_chat, other |
| `subtask` | the circuit, stage two | that task's kinds only: code → generate, debug, explain, review, convert, sql; writing → email_message, marketing_copy, social_post, document, rewrite, job_application; ... |
| `domain` | the circuit | software, marketing_sales, customer_support, finance, legal, hr_people, education, health, operations, science_engineering, media_entertainment, personal_life, other |
| `data_class` | the circuit | public, internal, confidential, regulated |

**Two stages.** The first request asks every tag question at once; a second asks only the
subtasks of the task the first one tagged, so "which kind of code work" is never asked of a
poem. A tag below the confidence floor is `untagged`, never guessed, and the report gives tag
coverage as the share of spend each tag covers.

**Apps come from code.** A program wraps each input in the same fixed instructions, so its
conversations open with the same words. The opening of the first message with numbers,
quotes, links and addresses blanked is the template key; a key three or more conversations
open with, going on to say different things, is one app. The same message sent many times
("hello! how are you today?") is a repeated request, not a program.

**Declared tags win.** A conversation that carries tags from its own metadata (`"tags":
{"environment": "dev_test", "app": "billing-service"}`, from an API key, a project, a header)
keeps them; the circuit fills the gaps, and `tag_source` says which is which. Environment
especially belongs in metadata: whether traffic is a test is rarely in its text.

## What the report recommends

Gates over the same answers (`finops/circuit.py`), actions in code (`finops/agent.py`):

| action | when | saves |
|---|---|---|
| policy | not business use | all of it |
| dev/test on a premium model | environment is dev_test and the model is not small | the difference to gpt-4o-mini |
| downgrade | a small model would do and the work is not hard | the difference to gpt-4o-mini |
| cache or template | many people make the same request | no dollar figure: depends on repeats |
| hold | confidential or regulated data | nothing moves models or gets cached without a person |
| trim context | six or more replies, resent history a third of the bill or more | |
| review | the task could not be tagged, or business use is unsure | |

Cost: a chat API is sent the whole conversation every turn, so a reply's input is everything
before it and a long conversation costs roughly the square of its length. Output tokens come
from the data; input tokens are estimated at four characters a token. `finops/pricing.py`
holds list prices per model; edit it for your contracts.

## One hundred real conversations

`finops report` on 100 WildChat conversations (8 models from gpt-3.5-turbo to o1-preview,
half not in English), tagged by Jev:

| | |
|---|---|
| tag coverage (share of spend) | task, subtask, domain, data_class 100%; workload 78%; environment 56% |
| apps found | 4, among them a paraphrasing tool, a SQL optimizer and a JSON translation job; 88 conversations adhoc |
| largest single line | one data-analysis conversation on o1-preview: 43% of all spend |
| business share of spend | 20% |
| actions | 71 policy, 18 cache, 6 downgrade, 8 dev/test on a premium model, 7 hold, 3 trim context, 11 review |

Environment is the weak tag, as it would be in any bill with no metadata: whether a
conversation is a test is rarely in what it says. Declare it where the traffic comes from and
the circuit only tags what it can read. circuit-1.7b v2.0 was never trained on this taxonomy
and leaves most of it untagged; Jev tags it confidently. Swapping the backend is one flag.

## Data

[WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (AI2, ODC-BY): real
conversations with ChatGPT models. `finops fetch` samples it through Hugging Face's datasets
server and keeps only the model, the language and the turns with their token counts; the
country, state, hashed IP and browser headers WildChat records are dropped before anything is
written. `data/` is not committed. The conversations in `samples/` are written for the tests
and carry hand-written answers; none come from WildChat.

## License

MIT. WildChat data is ODC-BY: attribute AI2 when you publish from it.
