# Proposal: Human-in-the-Loop support for `ai:Agent`

- **Status:** Implemented (initial version)
- **Module:** `ballerina/ai`
- **Authors:** WSO2 AI team

## Summary

This proposal adds **Human-in-the-Loop (HITL)** support to `ai:Agent`. An agent can now
**pause** mid-workflow to request human input, approval, or validation, surface that request
to the caller, and later **resume** execution using the human's response. This makes
AI-driven processes controllable and suitable for sensitive enterprise use cases
(refund approvals, privileged operations, decisions that need human judgment).

## Motivation

Before this change, `ai:Agent.run()` ran the full reason→act loop synchronously to
completion. It stopped only on a final chat answer, an error, or `maxIter`. There was no way
to interrupt the loop, hand control to a human, and continue based on their reply. This
prevented the agent from being used in workflows that require a human checkpoint.

## Design

The approach is **interrupt-and-resume** (similar to LangGraph's `interrupt`):

1. A tool requests human input by **returning an `ai:HumanInput` value** from its body.
2. The agent detects this, **pauses**, persists the conversation, and surfaces an
   `ai:Interrupt` describing what the human must provide.
3. The caller obtains the human's response out-of-band (UI, email, queue, ticket, …) and
   calls **`Agent.resume(sessionId, humanResponse)`**, which injects the response as the
   paused tool's result and **continues** the loop.

### Why tool-driven (not LLM-driven)

A tool returns `HumanInput` to pause. The tool itself decides when a human is needed
(e.g. only for refunds above a threshold), so the behavior is deterministic and does not
rely on the model choosing to ask.

```ballerina
@ai:AgentTool
isolated function refund(decimal amount, string customer) returns ai:HumanInput|json {
    if amount > 1000d {
        return {
            message: string `Approve refund of $${amount} to ${customer}?`,
            responseSchema: {"type": "boolean"}
        };
    }
    return {approved: true}; // small refunds auto-approve
}
```

`ai:HumanInput`, `ai:Interrupt`, and `ai:HumanResponse` are **closed records of `anydata`**.
This is required because (a) the compiler plugin restricts `@ai:AgentTool` return types to
`anydata | http:Response | stream<…> | error`, and (b) memory persists messages via
`cloneReadOnly()`.

### The resume checkpoint: a "dangling tool call"

The agent loop already records each step as a paired assistant `toolCalls` message and a
function-result message. To pause, the agent persists the triggering assistant tool-call
message **without** its matching function-result — a "dangling tool call". That dangling
message *is* the checkpoint, stored in the agent's `Memory` against the `sessionId`.

On `resume`, the agent loads the conversation (which ends with the dangling tool call),
appends a function-result message carrying the human's response, and re-enters the loop
with no new user message. The model sees a valid `[…, assistant{toolCalls}, function{result}]`
sequence and continues.

### Return types

Ballerina's dependently-typed return (`run` infers `td` ∈ `{string, Trace}`) forbids adding
a third record member (`Interrupt`) to the union — record basic types would collide. So the
pause is surfaced as:

- **String mode:** an `ai:InterruptError` (a distinct `error`, disjoint from `string`)
  whose `interrupt` detail field carries the `ai:Interrupt`.
- **Trace mode:** the returned `ai:Trace` carries the `interrupt` field.

```ballerina
string|ai:Error result = agent.run("Refund order #4521", sessionId = "sess-1");
if result is ai:InterruptError {
    ai:Interrupt interrupt = result.detail().interrupt;
    boolean decision = askHuman(interrupt.message);              // out of band
    string|ai:Error final = agent.resume("sess-1", decision);
    // `final` is the answer, or another InterruptError if more approvals are needed
}
```

## API additions

| Symbol | Description |
| --- | --- |
| `HumanInput` (record) | Returned from a tool body to pause and request input. Fields: `message`, optional `responseSchema`, optional `metadata`. |
| `Interrupt` (record) | Surfaced when paused. Fields: `interruptId`, `sessionId`, `toolName`, `toolArguments`, `message`, optional `responseSchema`, optional `metadata`. |
| `HumanResponse` (type) | `anydata` — the human's reply passed to `resume`. |
| `Agent.resume(sessionId, response, context?, td?)` | Continues a paused execution. Returns the answer, an `InterruptError`/`Trace`-with-interrupt if it pauses again, or an error. |
| `InterruptError` (error) | Carries an `Interrupt` in string mode. |
| `NoPendingInterruptError` (error) | Returned by `resume` when the session has no pending interrupt. |
| `Trace.interrupt` (field) | Set when the turn paused for human input. |

## Implementation notes

- `HumanInput` flows through tool execution as a normal `anydata` tool result and is detected
  in `Executor.act` (`ballerina/agent-utils.bal`), which sets an interrupt flag and stops the
  loop without recording a step.
- The shared `executeLoop` helper drives both `run` and `resume`; on interrupt it persists the
  dangling checkpoint and returns an `ExecutionTrace` carrying the `Interrupt`.
- `ShortTermMemory` trimming is now pairing-aware: it never leaves an orphaned function
  (tool-result) message at the head of the window, which protects the checkpoint across
  pause/resume (`ballerina/short_term_memory.bal`).
- `Agent.run`/`resume` are Java external trampolines
  (`native/.../Agent.java`) that compute `withTrace` and dispatch to `runInternal`/`resumeInternal`.

## Limitations (initial version)

- **Memory required across the pause:** the checkpoint lives in the configured `Memory`.
  Resume must use the same agent's memory (durable if a persistent `Memory` implementation is
  supplied; in-process for the default `ShortTermMemory`). Stateless agents are supported —
  the checkpoint is retained instead of deleted while paused.
- **Fresh iteration budget on resume:** each `resume` is bounded by `maxIter` independently of
  the pre-pause segment.
- **One tool call per step:** the agent executes a single tool call per iteration, which aligns
  naturally with pausing on that call.

## Testing

- `ballerina/tests/hitl-test.bal` — interrupt surfaced from `run`, `resume` continuation,
  trace-mode interrupt, `NoPendingInterruptError`, and stateless-agent resume.
- Follow-up: a compiler-plugin test asserting a tool returning `ai:HumanInput|json` compiles
  (a closed `anydata` record satisfies the return-type rule).
