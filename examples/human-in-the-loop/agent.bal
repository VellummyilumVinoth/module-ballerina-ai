// Copyright (c) 2025 WSO2 LLC (http://www.wso2.com).
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/ai;
import ballerinax/ai.openai;

configurable string openAiApiKey = ?;

// A Human-in-the-Loop tool. For sensitive actions (a large refund) it returns an
// `ai:HumanInput`, which pauses the agent and asks the user to approve. Smaller refunds
// are processed without interruption. On resume, the human's reply (e.g. "true"/"false")
// is fed back to the model as this tool's result.
@ai:AgentTool
isolated function refundPayment(int amount, string customer) returns ai:HumanInput|string {
    if amount > 1000 {
        ai:HumanInput approval = {
            message: string `Approve a refund of $${amount} to ${customer}?`,
            responseSchema: {"type": "boolean"}
        };
        return approval;
    }
    return string `Refund of $${amount} to ${customer} has been processed.`;
}

ai:SystemPrompt systemPrompt = {
    role: "Payments Assistant",
    instructions: string `You help operators process customer refunds.
When asked to refund a customer, call the refundPayment tool with the amount and customer name.
If the tool asks for approval, wait for the human's decision and then report the outcome clearly.`
};

final ai:ModelProvider openAiModel = check new openai:ModelProvider(openAiApiKey, modelType = openai:GPT_4O);

// Human-in-the-Loop requires a memory so the paused state can be persisted and resumed.
final ai:Agent paymentsAgent = check new (
    systemPrompt = systemPrompt,
    model = openAiModel,
    tools = [refundPayment],
    memory = check new ai:ShortTermMemory(check new ai:InMemoryShortTermMemoryStore(20)),
    verbose = true
);
