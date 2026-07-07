// Copyright (c) 2026 WSO2 LLC (http://www.wso2.com).
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

import ballerina/http;
import ballerina/lang.runtime;
import ballerina/test;

type CapturedMessage record {|
    string text;
    string sender;
    int count;
|};

// Shared state captured by the mock trigger service (a single guarded variable, as the handler runs
// on the listener strand). Ballerina permits only one restricted variable per `lock`.
isolated CapturedMessage capturedState = {text: "", sender: "", count: 0};

// A WhatsApp service that records inbound messages and sends no reply (so no outbound API call is made).
isolated service class MockWhatsAppService {
    *WhatsAppService;

    isolated remote function onMessages(TriggerMessage msg) returns TriggerReply|error? {
        lock {
            capturedState = {text: msg.text, sender: msg.sender, count: capturedState.count + 1};
        }
        return ();
    }

    isolated remote function onMessageTemplateStatusUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onMessageTemplateQualityUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onTemplateCategoryUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onPhoneNumberQualityUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onPhoneNumberNameUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onAccountUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onAccountReviewUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onBusinessCapabilityUpdate(WhatsAppEventNotification event) returns error? {
    }

    isolated remote function onSecurity(WhatsAppEventNotification event) returns error? {
    }
}

const int WA_TEST_PORT = 18190;

@test:Config {}
function testWhatsAppListenerHandshakeAndDispatch() returns error? {
    WhatsAppListener waListener = check new (WA_TEST_PORT,
        verifyToken = "test-verify-token", accessToken = "test-access-token", phoneNumberId = "PNID123");
    check waListener.attach(new MockWhatsAppService());
    check waListener.'start();

    http:Client callerClient = check new (string `http://localhost:${WA_TEST_PORT}`);

    // GET handshake: a matching verify token echoes the challenge with 200.
    http:Response challenge = check callerClient->get(
        "/?hub.mode=subscribe&hub.verify_token=test-verify-token&hub.challenge=challenge-123");
    test:assertEquals(challenge.statusCode, 200, "handshake should return 200");
    test:assertEquals(check challenge.getTextPayload(), "challenge-123", "challenge should be echoed");

    // GET handshake: a wrong verify token is rejected with 403.
    http:Response forbidden = check callerClient->get(
        "/?hub.mode=subscribe&hub.verify_token=wrong-token&hub.challenge=challenge-123");
    test:assertEquals(forbidden.statusCode, 403, "wrong verify token should be forbidden");

    // POST an inbound text message; it should reach onMessage as a normalized TriggerMessage.
    json inbound = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "PNID123"},
                            "contacts": [{"profile": {"name": "Alice"}, "wa_id": "15551234567"}],
                            "messages": [
                                {
                                    "from": "15551234567",
                                    "id": "wamid.TEST1",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": "Hello agent"}
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    };
    http:Response postResp = check callerClient->post("/", inbound);
    test:assertEquals(postResp.statusCode, 200, "webhook POST should be acknowledged with 200");

    // The listener acknowledges before dispatching, so allow the async dispatch to complete.
    runtime:sleep(2);

    lock {
        test:assertEquals(capturedState.count, 1, "onMessage should be invoked exactly once");
        test:assertEquals(capturedState.text, "Hello agent", "message text should be forwarded");
        test:assertEquals(capturedState.sender, "15551234567", "sender should be forwarded");
    }

    check waListener.gracefulStop();
}

// When no verify token is configured, the subscription handshake is disabled and echoes the
// challenge for any presented token.
@test:Config {}
function testWhatsAppListenerHandshakeWithoutVerifyToken() returns error? {
    WhatsAppListener waListener = check new (WA_TEST_PORT + 1,
        accessToken = "test-access-token", phoneNumberId = "PNID123");
    check waListener.attach(new MockWhatsAppService());
    check waListener.'start();

    http:Client callerClient = check new (string `http://localhost:${WA_TEST_PORT + 1}`);

    http:Response challenge = check callerClient->get(
        "/?hub.mode=subscribe&hub.verify_token=any-token&hub.challenge=challenge-456");
    test:assertEquals(challenge.statusCode, 200, "handshake should return 200 for any token");
    test:assertEquals(check challenge.getTextPayload(), "challenge-456", "challenge should be echoed");

    check waListener.gracefulStop();
}
