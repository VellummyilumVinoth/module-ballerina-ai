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

# A WhatsApp Business Cloud webhook notification for a non-message event (account, template,
# phone-number, security). Delivered to the corresponding `WhatsAppService` handler.
#
# + phoneNumberId - The business phone number ID the event relates to, if present.
# + 'field - The Meta webhook `field` discriminator identifying the event type.
# + payload - The `value` object of the change, as received.
public type WhatsAppEventNotification record {|
    string phoneNumberId;
    string 'field;
    json payload;
|};

# The WhatsApp Business Cloud trigger service. It exposes every WhatsApp Business Account webhook
# event as a separate handler (matching the Meta "Trigger On" event set). Attach an implementation
# to a `WhatsAppListener`; implement `onMessages` to answer inbound messages (typically with an
# `Agent`) and any of the other handlers to react to account/template/phone-number/security events.
public type WhatsAppService distinct service object {

    # Handles inbound WhatsApp messages (the `messages` field). Return a `TriggerReply` to reply to
    # the sender, `()` to send no reply, or an `error`.
    remote function onMessages(TriggerMessage msg) returns TriggerReply|error?;

    # Handles message template status updates (`message_template_status_update`).
    remote function onMessageTemplateStatusUpdate(WhatsAppEventNotification event) returns error?;

    # Handles message template quality updates (`message_template_quality_update`).
    remote function onMessageTemplateQualityUpdate(WhatsAppEventNotification event) returns error?;

    # Handles template category updates (`template_category_update`).
    remote function onTemplateCategoryUpdate(WhatsAppEventNotification event) returns error?;

    # Handles phone number quality updates (`phone_number_quality_update`).
    remote function onPhoneNumberQualityUpdate(WhatsAppEventNotification event) returns error?;

    # Handles phone number name updates (`phone_number_name_update`).
    remote function onPhoneNumberNameUpdate(WhatsAppEventNotification event) returns error?;

    # Handles WhatsApp Business Account updates (`account_update`).
    remote function onAccountUpdate(WhatsAppEventNotification event) returns error?;

    # Handles account review updates (`account_review_update`).
    remote function onAccountReviewUpdate(WhatsAppEventNotification event) returns error?;

    # Handles business capability updates (`business_capability_update`).
    remote function onBusinessCapabilityUpdate(WhatsAppEventNotification event) returns error?;

    # Handles security notifications (`security`).
    remote function onSecurity(WhatsAppEventNotification event) returns error?;
};
