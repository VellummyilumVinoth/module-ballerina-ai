// Copyright (c) 2025 WSO2 LLC (http://www.wso2.com).
//
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/jballerina.java;
import ballerina/test;

type ApprovalParams record {|
    string action;
|};

// A tool that pauses the agent by returning a `HumanInput` value, requesting approval
// before performing a sensitive action.
isolated function requestApprovalMock(*ApprovalParams params) returns HumanInput|string {
    return {
        message: string `Approve action: ${params.action}?`,
        responseSchema: {"type": "boolean"},
        metadata: {"action": params.action}
    };
}

ToolConfig approvalTool = {
    name: "requestApproval",
    description: "Requests human approval before performing a sensitive action.",
    parameters: {
        properties: {
            params: {
                properties: {
                    action: {'type: "string", description: "The action to approve"}
                }
            }
        }
    },
    caller: requestApprovalMock
};

SystemPrompt approvalSystemPrompt = {
    role: "Approval Assistant",
    instructions: "Help the user, requesting human approval for sensitive actions."
};

// Mock model that requests approval via the tool on the first turn, then produces a final
// answer once the human's approval result (a function message) is present in the history.
public isolated client class HitlMockLLM {
    *ModelProvider;

    isolated remote function chat(ChatMessage[]|ChatUserMessage messages, ChatCompletionFunctions[] tools,
            string? stop) returns ChatAssistantMessage|LlmError {
        ChatMessage[] msgs = [];
        if messages is ChatUserMessage {
            msgs.push(messages);
        } else {
            msgs = messages;
        }
        foreach ChatMessage message in msgs {
            if message is ChatFunctionMessage {
                return {
                    role: ASSISTANT,
                    content: string `The action was approved (${message.content ?: ""}) and completed.`
                };
            }
        }
        return {
            role: ASSISTANT,
            toolCalls: [{name: "requestApproval", arguments: {params: {action: "delete account"}}, id: "call_approval_1"}]
        };
    }

    isolated remote function generate(Prompt prompt, typedesc<anydata> td = <>) returns td|Error = @java:Method {
        'class: "io.ballerina.lib.ai.MockGenerator"
    } external;
}

@test:Config
function testAgentInterruptsOnHumanInput() returns error? {
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new HitlMockLLM(),
        tools = [approvalTool],
        memory = check new ShortTermMemory()
    );
    string sessionId = "hitl-interrupt";
    string|Error result = agent.run("Please delete my account", sessionId);
    if result !is InterruptError {
        test:assertFail("Expected the agent to pause with an InterruptError");
    }
    Interrupt interrupt = result.detail().interrupt;
    test:assertEquals(interrupt.toolName, "requestApproval");
    test:assertEquals(interrupt.sessionId, sessionId);
    test:assertEquals(interrupt.message, "Approve action: delete account?");
    test:assertEquals(interrupt.responseSchema, {"type": "boolean"});
}

@test:Config
function testAgentResumeContinues() returns error? {
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new HitlMockLLM(),
        tools = [approvalTool],
        memory = check new ShortTermMemory()
    );
    string sessionId = "hitl-resume";
    string|Error result = agent.run("Please delete my account", sessionId);
    if result !is InterruptError {
        test:assertFail("Expected the agent to pause with an InterruptError");
    }

    string|Error resumed = agent.resume(sessionId, true);
    if resumed is Error {
        test:assertFail("Resume failed: " + resumed.message());
    }
    test:assertEquals(resumed, "The action was approved (true) and completed.");
}

@test:Config
function testAgentRunWithTraceCarriesInterrupt() returns error? {
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new HitlMockLLM(),
        tools = [approvalTool],
        memory = check new ShortTermMemory()
    );
    Trace|Error result = agent.run("Please delete my account", "hitl-trace");
    if result is Error {
        test:assertFail("Expected a Trace, got an error: " + result.message());
    }
    Interrupt? interrupt = result?.interrupt;
    if interrupt !is Interrupt {
        test:assertFail("Expected the trace to carry an interrupt");
    }
    test:assertEquals(interrupt.toolName, "requestApproval");
}

@test:Config
function testResumeWithoutPendingInterrupt() returns error? {
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new HitlMockLLM(),
        tools = [approvalTool],
        memory = check new ShortTermMemory()
    );
    string|Error resumed = agent.resume("session-with-no-pause", true);
    if resumed !is NoPendingInterruptError {
        test:assertFail("Expected a NoPendingInterruptError when no interrupt is pending");
    }
}

// A real `@AgentTool`-annotated function returning `HumanInput` — the annotation makes the
// compiler plugin validate the return type, confirming `HumanInput` (a closed `anydata`
// record) is an accepted tool return type and does not trigger diagnostic ERROR_104.
@AgentTool {description: "Approve a payment above the configured threshold."}
isolated function approvePayment(int amount) returns HumanInput|json {
    if amount > 1000 {
        HumanInput humanInput = {message: string `Approve payment of ${amount}?`, responseSchema: {"type": "boolean"}};
        return humanInput;
    }
    return {approved: true};
}

ToolConfig paymentTool = {
    name: "approvePayment",
    description: "Approve a payment above the configured threshold.",
    parameters: {
        properties: {
            amount: {'type: "integer", description: "The payment amount"}
        }
    },
    caller: approvePayment
};

public isolated client class PaymentMockLLM {
    *ModelProvider;

    isolated remote function chat(ChatMessage[]|ChatUserMessage messages, ChatCompletionFunctions[] tools,
            string? stop) returns ChatAssistantMessage|LlmError {
        ChatMessage[] msgs = [];
        if messages is ChatUserMessage {
            msgs.push(messages);
        } else {
            msgs = messages;
        }
        foreach ChatMessage message in msgs {
            if message is ChatFunctionMessage {
                return {role: ASSISTANT, content: string `Payment approved (${message.content ?: ""}).`};
            }
        }
        return {role: ASSISTANT, toolCalls: [{name: "approvePayment", arguments: {amount: 1500}, id: "pay_1"}]};
    }

    isolated remote function generate(Prompt prompt, typedesc<anydata> td = <>) returns td|Error = @java:Method {
        'class: "io.ballerina.lib.ai.MockGenerator"
    } external;
}

@test:Config
function testAnnotatedToolReturningHumanInputPauses() returns error? {
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new PaymentMockLLM(),
        tools = [paymentTool],
        memory = check new ShortTermMemory()
    );
    string sessionId = "hitl-payment";
    string|Error result = agent.run("Pay invoice #12", sessionId);
    if result !is InterruptError {
        test:assertFail("Expected the agent to pause with an InterruptError");
    }
    Interrupt interrupt = result.detail().interrupt;
    test:assertEquals(interrupt.toolName, "approvePayment");
    test:assertEquals(interrupt.message, "Approve payment of 1500?");

    string|Error resumed = agent.resume(sessionId, true);
    if resumed is Error {
        test:assertFail("Resume failed: " + resumed.message());
    }
    test:assertEquals(resumed, "Payment approved (true).");
}

@test:Config
function testStatelessAgentSupportsResume() returns error? {
    // A stateless agent (memory explicitly set to ()) must still be resumable: the paused
    // checkpoint is retained in the default in-memory store rather than deleted on pause.
    Agent agent = check new (
        systemPrompt = approvalSystemPrompt,
        model = new HitlMockLLM(),
        tools = [approvalTool],
        memory = ()
    );
    string sessionId = "hitl-stateless";
    string|Error result = agent.run("Please delete my account", sessionId);
    if result !is InterruptError {
        test:assertFail("Expected the stateless agent to pause with an InterruptError");
    }

    string|Error resumed = agent.resume(sessionId, true);
    if resumed is Error {
        test:assertFail("Resume failed for stateless agent: " + resumed.message());
    }
    test:assertEquals(resumed, "The action was approved (true) and completed.");
}
