# Human-in-the-Loop agent

A minimal `ai:Agent` that pauses for human approval before a sensitive action and resumes
once the human responds. It exposes the standard `/chat` endpoint, so the **Agent Chat**
panel in the WSO2 VS Code extension renders the pause as an inline **Approve / Reject**
prompt and resumes the agent when you click.

## How it works

- `agent.bal` — the `refundPayment` tool returns an `ai:HumanInput` for refunds over `$1000`,
  which pauses the agent. The agent is created with a `memory` (required for HITL).
- `service.bal` — the chat service is a single call: `paymentsAgent.chat(request)`. That one
  method handles the whole flow — it starts a new turn or resumes the paused session, and when
  the agent pauses it returns the `interrupt` in `ai:ChatRespMessage` so the UI can prompt. (Under
  the hood it wraps `run`/`resume`; you don't need to track paused sessions yourself.)

HTTP contract consumed by the extension:

| Stage  | Request                          | Response                                                            |
| ------ | -------------------------------- | ------------------------------------------------------------------- |
| Ask    | `{sessionId, message:"refund $5000 to Alice"}` | `{message:"Approve a refund of $5000 to Alice?", interrupt:{...}}` |
| Resume | `{sessionId, message:"true"}`    | `{message:"Refund of $5000 to Alice has been processed."}`          |

## Prerequisites

Publish the HITL-enabled `ballerina/ai` build to your local repository:

```bash
cd ../../ballerina
bal pack
bal push --repository=local
```

## Run

1. Add your OpenAI key in `Config.toml`:

   ```toml
   openAiApiKey = "sk-..."
   ```

2. Start the service:

   ```bash
   bal run
   ```

   It listens on `http://localhost:9090/chat`.

## Try it

- **From VS Code:** open this project in the WSO2 extension and use **Try it** on the agent
  service to open the Agent Chat panel. Ask: *"Refund $5000 to Alice"*. The agent pauses and
  shows an **Approve / Reject** prompt; click one and the agent resumes.
- **From curl:**

  ```bash
  # Ask (pauses)
  curl -s localhost:9090/chat -H 'Content-Type: application/json' \
    -d '{"sessionId":"s1","message":"Refund $5000 to Alice"}'
  # -> {"message":"Approve a refund of $5000 to Alice?","interrupt":{...}}

  # Resume with the decision
  curl -s localhost:9090/chat -H 'Content-Type: application/json' \
    -d '{"sessionId":"s1","message":"true"}'
  # -> {"message":"Refund of $5000 to Alice has been processed."}
  ```

> Note: the BI server-side chat-service generator does not yet emit this run/resume handling
> automatically — this sample shows the pattern to add to a generated chat service.
