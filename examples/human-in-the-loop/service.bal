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
import ballerina/http;

// The chat service the VS Code Agent Chat panel talks to.
//
// `agent.chat(request)` handles the full Human-in-the-Loop flow in a single call: it starts a
// new turn or, if the session is awaiting a human response, resumes the paused agent. When the
// agent pauses, the returned `ai:ChatRespMessage` carries an `interrupt` so the UI can render an
// approval prompt; the user's reply (the next message) resumes the agent automatically.
service on new ai:Listener(9090) {
    resource function post chat(@http:Payload ai:ChatReqMessage request) returns ai:ChatRespMessage|error {
        return paymentsAgent.chat(request);
    }
}
