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

# A normalized incoming message delivered by a `Channel` to a `TriggerService`.
#
# + sender - Identifier of the sender, used both to route replies and as the agent session ID.
# + messageId - Channel-specific unique identifier of the message.
# + text - Text content of the message.
# + timestamp - Optional timestamp of when the message was sent, as reported by the channel.
# + channel - Optional name of the originating channel (e.g. `"whatsapp"`).
public type TriggerMessage record {|
    string sender;
    string messageId;
    string text;
    string timestamp?;
    string channel?;
|};

# A normalized reply to be sent back to the sender through the originating `Channel`.
#
# + text - Text content of the reply.
public type TriggerReply record {|
    string text;
|};

# Defines the service interface for handling normalized incoming messages from a trigger listener
# such as `WhatsAppListener`.
public type TriggerService distinct service object {
    # Handles a normalized incoming message.
    #
    # + msg - The normalized incoming message.
    # + return - A `TriggerReply` to send back, `()` to send no reply, or an `error`.
    remote function onMessage(TriggerMessage msg) returns TriggerReply|error?;
};
