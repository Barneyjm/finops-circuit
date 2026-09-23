# finops-circuit

An LLM FinOps agent built on [decision circuits](https://github.com/Barneyjm/decision-circuits).
Each conversation comes in with its model and tokens; one request asks a decision model six
typed questions about it; a circuit, plain code, turns the probabilities into a value bucket
and the actions a cost review recommends; the report adds it up.

```
$ finops analyze samples/01_support_reply.json --backend fake

== 01_support_reply.json  gpt-4-0613, 1 replies, $0.0053  (49 in / 64 out)
   A customer says their order #88231 arrived with a cracked screen and wants a replacement before Friday. Write
-> value=customer_facing business=True saves $0.0053
   value_type customer_facing 0.85 | work 0.95 | small model ok 0.90 | repeatable 0.80 | sensitive 0.15
   * downgrade to gpt-4o-mini
   * cache or template: a common request
   value      decided   value_type -> customer_facing p=0.85 conf=0.63 (min 0.25)
   business   decided   work p=0.95; gate _business_2 p=0.94; and under independence -> p=0.89
   downgrade  decided   small_model_ok p=0.90; gate _downgrade_3 p=0.93; and under independence -> p=0.84
```

The model classifies. It never prices anything and never writes the report: cost is tokens
times a price table, and every action is a gate over the model's probabilities.

## Run it

```bash
git clone https://github.com/Barneyjm/finops-circuit && cd finops-circuit
uv sync
cp .env.example .env                              # one key for the backend you pick
uv run finops fetch --n 200                       # a WildChat-4.8M sample into data/
uv run finops report data/wildchat.jsonl --backend jev
```

| | |
|---|---|
| `finops fetch --n N` | N real conversations from WildChat-4.8M into `data/wildchat.jsonl` |
| `finops analyze <file>` | one conversation or a directory: cost, value, actions, the gate traces |
| `finops report <file>` | the review: spend by value bucket, business share, actions, savings |
| `finops diagram` | the circuit as Mermaid |

`--json` prints full findings with the audit record. `--backend` picks the model, as in
[call-center-circuit](https://github.com/Barneyjm/call-center-circuit): `jev`, `circuits`,
`local`, `semif`, `openai`, `anthropic`, or `fake` (the hand-written answers in `samples/`,
no network).

## What the circuit asks

| question | type | used for |
|---|---|---|
| `value_type` | choice of 6 | the bucket spend is reported under: customer-facing, internal productivity, engineering, research, personal, waste |
| `work` | yes/no | business spend or not, checked against the bucket |
| `complexity` | score, 4 levels | never recommend a cheaper model for hard work |
| `small_model_ok` | yes/no | the downgrade candidate |
| `repeatable` | yes/no | a common request: cache it or template it |
| `sensitive` | yes/no | personal or confidential data: a person decides before anything moves |

With a circuit v2 model (`circuits`, `local`) two more go out: `functions` (multi: every
business function served) and `purpose` (locate: the line that shows what it was for).

The gates, in `finops/circuit.py`:

```python
c.gate("value", argmax("value_type", min_confidence=0.25), on_uncertain="escalate")
c.gate("business", (Q("work") & ~(Q("value_type")["personal"] | Q("value_type")["waste"])) >= 0.5, band=0.1, on_uncertain="escalate")
c.gate("downgrade", (Q("small_model_ok") & ~Q("complexity")[3]) >= 0.65, band=0.1, on_uncertain="default", default=False)
c.gate("cache", Q("repeatable") >= 0.7, band=0.1, on_uncertain="default", default=False)
c.gate("policy", (Q("value_type")["waste"] | (Q("value_type")["personal"] & ~Q("work"))) >= 0.6, band=0.1, on_uncertain="default", default=False)
c.gate("hold", Q("sensitive") >= 0.5, band=0.15, on_uncertain="default", default=True)
```

And the actions, in `finops/agent.py`: **policy** (spend with no business use; saves all of
it), **downgrade** to a small model (saves the price difference; never for hard or sensitive
work, never when the model is already small), **cache or template** (no dollar figure: it
depends on how often the request repeats), **hold** (sensitive data), **trim context** (six
or more replies where resending earlier turns is a third of the bill or more), **review**
(the circuit was not sure what it was for).

## Cost

A chat API is sent the whole conversation on every turn, so a reply's input is everything
before it; a long conversation costs roughly the square of its length. Output tokens come from
the data (WildChat records them per reply); input tokens are estimated at four characters a
token. `finops/pricing.py` holds list prices per model at release; edit it for your contracts.

## One hundred real conversations

`finops report` on 100 WildChat conversations (8 models, from gpt-3.5-turbo to o1-preview,
half of them not in English):

| | Jev (jev-1.13) | circuit-1.7b v2.0 |
|---|---|---|
| classified | 99 | 21 (79 to review) |
| business share of spend | 18% | 11% |
| largest bucket | personal, 51% of spend | |
| downgrade / policy / cache flagged | 21 / 38 / 20 | 0 / 0 / 0 |
| savings from the actions with a dollar figure | 56% of spend | |

Jev answers this question confidently and, read by hand, mostly right. circuit-1.7b v2.0 was
never trained on it: its answers are spread thin, the `value` gate sends them to review
rather than guess, and where the two disagree Jev is usually the better read. That is the
circuit doing its job (unsure goes to a person) and a clear training target for the open
model. Swapping the backend is one flag; the policy does not change.

## Data

[WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (AI2, ODC-BY):
real conversations with ChatGPT models. `finops fetch` samples it through Hugging Face's
datasets server and keeps only the model, the language and the turns with their token
counts. The country, state, hashed IP and browser headers WildChat records are dropped
before anything is written. `data/` is not committed. The ten conversations in `samples/`
are written for the tests and carry hand-written answers; none come from WildChat.

## License

MIT. WildChat data is ODC-BY: attribute AI2 when you publish from it.
