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
import ballerinax/whatsapp.business.cloud as wa;

# Configuration for the `WhatsAppListener`.
#
# + verifyToken - Token used to verify the Meta webhook subscription (the GET `hub.verify_token`
# handshake). When set, use the same value in the Meta App webhook configuration. Omit to accept any
# subscription token (handshake verification disabled; not recommended for production).
# + accessToken - WhatsApp Business Cloud API permanent access token, used to send reply messages.
# + phoneNumberId - WhatsApp Business phone number ID that replies are sent from.
# + appSecret - Meta app secret. When set, every inbound notification is authenticated via the
# `X-Hub-Signature-256` header (HMAC-SHA256 over the raw body). Omit to disable signature
# verification (not recommended for production).
# + apiVersion - Graph API version used for outbound sends. Defaults to `"v23.0"`.
@display {label: "WhatsApp Listener Configuration"}
public type WhatsAppListenerConfig record {|
    string? verifyToken = ();
    string accessToken;
    string phoneNumberId;
    string appSecret?;
    string apiVersion = "v23.0";
|};

# A WhatsApp Business Cloud trigger listener for AI agents. It wraps the
# `ballerinax/whatsapp.business.cloud` connector: inbound webhooks (Meta handshake,
# `X-Hub-Signature-256` verification and event parsing) are handled by the connector's `Listener`,
# and replies are sent through the connector's `Client`. Inbound text messages are delivered to the
# attached `TriggerService` (or, via `attachAgent`, straight to an `Agent`), and any reply it
# returns is sent back to the original sender.
#
# ```ballerina
# listener ai:WhatsAppListener waListener = new (
#     verifyToken = "my-verify-token", accessToken = "my-access-token", phoneNumberId = "1234567890");
#
# service ai:TriggerService on waListener {
#     remote function onMessage(ai:TriggerMessage msg) returns ai:TriggerReply|error? {
#         return {text: check myAgent.run(msg.text, msg.sender)};
#     }
# }
# ```
public class WhatsAppListener {
    private final wa:Listener waListener;
    private final wa:Client waClient;
    private final string phoneNumberId;
    private final string accessToken;
    private final string apiVersion;

    # Initializes the WhatsApp listener.
    #
    # + listenOn - Port number to bind a new HTTP listener to, or an existing `http:Listener`
    # + config - The listener configuration (tokens, phone number ID and optional app secret)
    # + return - An `error` if the listener or API client could not be initialized, otherwise `()`
    public function init(int|http:Listener listenOn = 8090, *WhatsAppListenerConfig config)
            returns Error? {
        self.phoneNumberId = config.phoneNumberId;
        self.accessToken = config.accessToken;
        self.apiVersion = config.apiVersion;

        string? verifyToken = config.verifyToken;
        string? appSecret = config?.appSecret;
        wa:Listener|error waListener = appSecret is string
            ? new (listenOn, verifyToken = verifyToken, appSecret = appSecret)
            : new (listenOn, verifyToken = verifyToken);
        if waListener is error {
            return error WhatsAppTriggerError("Failed to initialize the WhatsApp webhook listener", waListener);
        }
        self.waListener = waListener;

        wa:Client|error waClient = new ({auth: {token: config.accessToken}});
        if waClient is error {
            return error WhatsAppTriggerError("Failed to initialize the WhatsApp API client", waClient);
        }
        self.waClient = waClient;
    }

    # Attaches a `WhatsAppService` to this listener. Inbound messages are delivered to `onMessages`
    # (any returned `TriggerReply` is sent back to the sender), and the other WhatsApp webhook events
    # are delivered to their respective handlers.
    #
    # + whatsappService - The service that handles WhatsApp webhook events
    # + name - The path (or path segments) to attach on; defaults to the listener root
    # + return - An `error` if attachment fails, otherwise `()`
    public function attach(WhatsAppService whatsappService, string[]|string? name = ()) returns error? {
        wa:WhatsAppService bridge = new WhatsAppServiceBridge(
                whatsappService, self.waClient, self.apiVersion, self.phoneNumberId, self.accessToken);
        check self.waListener.attach(bridge, name);
    }

    # Attaches an `Agent` directly to this listener. Each inbound message is passed to `agent.run`,
    # using the sender as the session ID so every conversation keeps its own memory, and the
    # response is sent back as the reply.
    #
    # + agent - The agent that answers incoming messages
    # + name - The path (or path segments) to attach on; defaults to the listener root
    # + return - An `error` if attachment fails, otherwise `()`
    public function attachAgent(Agent agent, string[]|string? name = ()) returns error? {
        check self.attach(new AgentWhatsAppService(agent), name);
    }

    # Detaches a previously attached service.
    #
    # + whatsappService - The service to detach
    # + return - An `error` if detaching fails, otherwise `()`
    public function detach(WhatsAppService whatsappService) returns error? {
        // The underlying connector listener manages a single bridge service; nothing to do here.
    }

    # Starts the listener.
    #
    # + return - An `error` if the listener could not be started, otherwise `()`
    public function 'start() returns error? {
        return self.waListener.'start();
    }

    # Gracefully stops the listener, allowing in-flight requests to complete.
    #
    # + return - An `error` if the listener could not be stopped, otherwise `()`
    public function gracefulStop() returns error? {
        return self.waListener.gracefulStop();
    }

    # Immediately stops the listener.
    #
    # + return - An `error` if the listener could not be stopped, otherwise `()`
    public function immediateStop() returns error? {
        return self.waListener.immediateStop();
    }
}

// Bridges the connector's `WhatsAppService` events to the attached ai service. Inbound messages go
// to `onMessage`/`onMessages` (and any reply is sent back through the connector client); non-message
// events are routed by their `field` to the matching `WhatsAppService` handler.
service class WhatsAppServiceBridge {
    *wa:WhatsAppService;

    private final WhatsAppService userService;
    private final wa:Client waClient;
    private final string apiVersion;
    private final string phoneNumberId;
    private final string accessToken;

    function init(WhatsAppService userService, wa:Client waClient, string apiVersion,
            string phoneNumberId, string accessToken) {
        self.userService = userService;
        self.waClient = waClient;
        self.apiVersion = apiVersion;
        self.phoneNumberId = phoneNumberId;
        self.accessToken = accessToken;
    }

    remote function onMessageReceived(wa:MessageReceivedEvent event) returns error? {
        string? text = event.text;
        if text is () {
            // Non-text messages are not forwarded to the agent path.
            return;
        }
        TriggerMessage message = {
            sender: event.'from,
            messageId: event.messageId,
            text,
            timestamp: event.timestamp ?: "",
            channel: "whatsapp"
        };
        TriggerReply|error? reply = self.userService->onMessages(message);
        if reply is error {
            return reply;
        }
        if reply is TriggerReply {
            wa:TextMessage response = {
                messaging_product: "whatsapp",
                recipient_type: "individual",
                to: event.'from,
                'type: "text",
                text: {body: reply.text}
            };
            wa:SendMessageHeaders headers = {Authorization: string `Bearer ${self.accessToken}`};
            _ = check self.waClient->sendMessage(self.apiVersion, self.phoneNumberId, headers, response);
        }
    }

    remote function onMessageStatus(wa:MessageStatusEvent event) returns error? {
        // Delivery/read receipts are not routed to the agent path.
    }

    remote function onEvent(wa:WhatsAppEvent event) returns error? {
        WhatsAppEventNotification notification = {
            phoneNumberId: event.phoneNumberId,
            'field: event.'field,
            payload: event.value
        };
        match event.'field {
            "message_template_status_update" => {
                return self.userService->onMessageTemplateStatusUpdate(notification);
            }
            "message_template_quality_update" => {
                return self.userService->onMessageTemplateQualityUpdate(notification);
            }
            "template_category_update" => {
                return self.userService->onTemplateCategoryUpdate(notification);
            }
            "phone_number_quality_update" => {
                return self.userService->onPhoneNumberQualityUpdate(notification);
            }
            "phone_number_name_update" => {
                return self.userService->onPhoneNumberNameUpdate(notification);
            }
            "account_update" => {
                return self.userService->onAccountUpdate(notification);
            }
            "account_review_update" => {
                return self.userService->onAccountReviewUpdate(notification);
            }
            "business_capability_update" => {
                return self.userService->onBusinessCapabilityUpdate(notification);
            }
            "security" => {
                return self.userService->onSecurity(notification);
            }
        }
    }
}

// A `WhatsAppService` that routes inbound messages straight to an `Agent` (used by `attachAgent`).
// Non-message events are ignored.
service class AgentWhatsAppService {
    *WhatsAppService;

    private final Agent agent;

    function init(Agent agent) {
        self.agent = agent;
    }

    remote function onMessages(TriggerMessage msg) returns TriggerReply|error? {
        return {text: check self.agent.run(msg.text, msg.sender)};
    }

    remote function onMessageTemplateStatusUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onMessageTemplateQualityUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onTemplateCategoryUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onPhoneNumberQualityUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onPhoneNumberNameUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onAccountUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onAccountReviewUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onBusinessCapabilityUpdate(WhatsAppEventNotification event) returns error? {
    }

    remote function onSecurity(WhatsAppEventNotification event) returns error? {
    }
}
