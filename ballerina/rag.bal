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

import ai.wso2;

public type ChunkStratery isolated object {
    public isolated function chunk(string content) returns Document[]|Error;
};

public isolated class DocumentByLineSplitter {
    *ChunkStratery;

    public isolated function chunk(string content) returns Document[]|Error {
        return re `\n`.split(content).'map(line => {content: line});
    }
}

public isolated class Rag {
    private final ModelProvider model;
    private final VectorKnowledgeBase knowledgeBase;
    private final RagPromptBuilder promptBuilder;

    public isolated function init(ModelProvider? model = (),
            VectorKnowledgeBase? knowledgeBase = (),
            RagPromptBuilder promptBuilder = new DefaultRagPromptBuilder()) returns Error? {
        self.model = model ?: check getDefaultModelProvider();
        self.knowledgeBase = knowledgeBase ?: check getDefaultKnowledgeBase();
        self.promptBuilder = promptBuilder;
    }

    public isolated function query(string query) returns string|Error {
        DocumentMatch[] context = check self.knowledgeBase.getRetriever().retrieve(query);
        // later when we allow re-reankers we can use the score in the document match
        Prompt prompt = self.promptBuilder.build(context.'map(ctx => ctx.document), query);
        ChatMessage[] messages = self.mapPromptToChatMessages(prompt);
        ChatAssistantMessage response = check self.model->chat(messages, []);
        return response.content ?: error Error("Unable to obtain valid answer");
    }

    public isolated function ingest(Document[] documents) returns Error? {
        return self.knowledgeBase.index(documents);
    }

    private isolated function mapPromptToChatMessages(Prompt prompt) returns ChatMessage[] {
        string? systemPrompt = prompt?.systemPrompt;
        string? userPrompt = prompt?.userPrompt;
        ChatMessage[] messages = [];
        if systemPrompt is string {
            messages.push({role: SYSTEM, content: systemPrompt});
        }
        if userPrompt is string {
            messages.push({role: USER, content: userPrompt});
        }
        return messages;
    }
};

# Description.
#
# + systemPrompt - field description  
# + userPrompt - field description
public type Prompt record {|
    string systemPrompt?;
    string userPrompt;
|};

public type RagPromptBuilder isolated object {
    public isolated function build(Document[] context, string query) returns Prompt;
};

public isolated class DefaultRagPromptBuilder {
    *RagPromptBuilder;

    public isolated function build(Document[] context, string query) returns Prompt {
        // following is a sample implementation
        string systemPrompt = string `Answer the question based on the following provided context: `
            + string `<CONTEXT>${string:'join("\n", ...context.'map(doc => doc.content))}</CONTEXT>"""`;
        string userPrompt = "Question:\n" + query;
        return {systemPrompt, userPrompt};
    }
}

public isolated class VectorKnowledgeBase {
    private final VectorStore vectorStore;
    private final EmbeddingProvider embeddingModel;
    private final Retriever retriever;

    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel) {
        self.embeddingModel = embeddingModel;
        self.vectorStore = vectorStore;
        self.retriever = new (vectorStore, embeddingModel);
    }

    public isolated function index(Document[] documents) returns Error? {
        VectorEntry[] entries = [];
        foreach var document in documents {
            float[]|SparseVector|Embedding embedding = check self.embeddingModel->embed(document.content);
            VectorEntry entry = {embedding, document};
            // generate sparse vectors
            entries.push(entry);
        }
        check self.vectorStore.add(entries);
    }

    isolated function getRetriever() returns Retriever {
        return self.retriever;
    }
};

public type Document record {
    string content;
    map<anydata> metadata?;
};

public type VectorEntry record {|
    Vector|SparseVector|Embedding embedding;
    Document document;
|};

public type VectorMatch record {|
    *VectorEntry;
    float score;
|};

public type DocumentMatch record {|
    Document document;
    float score;
|};

public type VectorStore isolated object {
    public isolated function add(VectorEntry[] entries) returns Error?;
    public isolated function query(Vector|SparseVector|Embedding query) returns VectorMatch[]|Error;
};

public type Vector float[];

public type Embedding record {|
    float[] dense;
    SparseVector sparse;
|};

public type EmbeddingProvider isolated client object {
    isolated remote function embed(string document) returns Vector|SparseVector|Embedding|Error;
};

public isolated client class Wso2EmbeddingProvider {
    *EmbeddingProvider;
    private final wso2:Client embeddingClient;

    public isolated function init(*Wso2ModelProviderConfig config) returns Error? {
        wso2:Client|error embeddingClient = new (config = {auth: {token: config.accessToken}}, 
        serviceUrl = config.serviceUrl);
        if embeddingClient is error {
            return error Error("Failed to initialize Wso2ModelProvider", embeddingClient);
        }
        self.embeddingClient = embeddingClient;
    }

    isolated remote function embed(string document) returns Vector|SparseVector|Embedding|Error {
        wso2:EmbeddingRequest request = {input: document};
        wso2:EmbeddingResponse|error response = self.embeddingClient->/embeddings.post(request);
        if response is error {
            return error Error("Error generating embedding for provided document", response);
        }
        return response.data[0].embedding;
    }
}

public isolated class Retriever {
    private final VectorStore vectorStore;
    private final EmbeddingProvider embeddingModel;

    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel) {
        self.vectorStore = vectorStore;
        self.embeddingModel = embeddingModel;
    }

    public isolated function retrieve(string query) returns DocumentMatch[]|Error {
        Vector|SparseVector|Embedding embedding = check self.embeddingModel->embed(query);
        VectorMatch[] matches = check self.vectorStore.query(embedding);
        return from VectorMatch 'match in matches
            select {document: 'match.document, score: 'match.score};
    }
}

public type SparseVector record {
    int[] indices;
    Vector values;
};

public enum VectorStoreQueryMode {
    DENSE,
    SPARSE,
    HYBRID
};

public isolated class InMemoryVectorStore {
    *VectorStore;
    private final VectorEntry[] entries = [];
    private final int topK;

    public isolated function init(int topK = 3) {
        self.topK = topK;
    }

    public isolated function add(VectorEntry[] entries) returns Error? {
        foreach VectorEntry entry in entries {
            if entry.embedding !is Vector {
                return error Error("InMemoryVectorStore implementation only supports dense vectors");
            }
        }
        readonly & VectorEntry[] clonedEntries = entries.cloneReadOnly();
        lock {
            self.entries.push(...clonedEntries);
        }
    }

    public isolated function query(Vector|SparseVector|Embedding query) returns VectorMatch[]|Error {
        if query !is Vector {
            return error Error("InMemoryVectorStore implementation only supports dense vectors");
        }

        lock {
            VectorMatch[] results = from var entry in self.entries
                let float similarity = self.cosineSimilarity(query.clone(), <Vector>entry.embedding)
                limit self.topK
                select {document: entry.document, embedding: entry.embedding, score: similarity};
            return results.clone();
        }
    }

    isolated function cosineSimilarity(Vector a, Vector b) returns float {
        if a.length() != b.length() {
            return 0.0;
        }

        float dot = 0.0;
        float normA = 0.0;
        float normB = 0.0;

        foreach int i in 0 ..< a.length() {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        float denom = normA.sqrt() * normB.sqrt();
        return denom == 0.0 ? 0.0 : dot / denom;
    }
}

public type Wso2ModelProviderConfig record {|
    string serviceUrl;
    string accessToken;
|};

configurable Wso2ModelProviderConfig? wso2ModelProviderConfig = ();

public isolated client class Wso2ModelProvider {
    *ModelProvider;
    private final wso2:Client llmClient;

    public isolated function init(*Wso2ModelProviderConfig config) returns Error? {
        wso2:Client|error llmClient = new (config = {auth: {token: config.accessToken}}, serviceUrl = config.serviceUrl);
        if llmClient is error {
            return error Error("Failed to initialize Wso2ModelProvider", llmClient);
        }
        self.llmClient = llmClient;
    }

    isolated remote function chat(ChatMessage[] messages, ChatCompletionFunctions[] tools, string? stop = ())
    returns ChatAssistantMessage|LlmError {
        wso2:CreateChatCompletionRequest request = {stop, messages: self.mapToChatCompletionRequestMessage(messages)};
        if tools.length() > 0 {
            request.functions = tools;
        }
        wso2:CreateChatCompletionResponse|error response = self.llmClient->/chat/completions.post(request);
        if response is error {
            return error LlmConnectionError("Error while connecting to the model", response);
        }

        var choices = response.choices;
        if choices.length() == 0 {
            return error LlmInvalidResponseError("Empty response from the model when using function call API");
        }
        wso2:ChatCompletionResponseMessage? message = choices[0].message;
        ChatAssistantMessage chatAssistantMessage = {role: ASSISTANT, content: message?.content};
        wso2:ChatCompletionFunctionCall? functionCall = message?.functionCall;
        if functionCall is wso2:ChatCompletionFunctionCall {
            chatAssistantMessage.toolCalls = [check self.mapToFunctionCall(functionCall)];
        }
        return chatAssistantMessage;
    }

    private isolated function mapToChatCompletionRequestMessage(ChatMessage[] messages)
        returns wso2:ChatCompletionRequestMessage[] {
        wso2:ChatCompletionRequestMessage[] chatCompletionRequestMessages = [];
        foreach ChatMessage message in messages {
            if message is ChatAssistantMessage {
                wso2:ChatCompletionRequestMessage assistantMessage = {role: ASSISTANT};
                FunctionCall[]? toolCalls = message.toolCalls;
                if toolCalls is FunctionCall[] {
                    assistantMessage["function_call"] = {
                        name: toolCalls[0].name,
                        arguments: toolCalls[0].arguments.toJsonString()
                    };
                }
                if message?.content is string {
                    assistantMessage["content"] = message?.content;
                }
                chatCompletionRequestMessages.push(assistantMessage);
            } else {
                chatCompletionRequestMessages.push(message);
            }
        }
        return chatCompletionRequestMessages;
    }

    private isolated function mapToFunctionCall(wso2:ChatCompletionFunctionCall functionCall)
    returns FunctionCall|LlmError {
        do {
            json jsonArgs = check functionCall.arguments.fromJsonString();
            map<json>? arguments = check jsonArgs.cloneWithType();
            return {name: functionCall.name, arguments};
        } on fail error e {
            return error LlmError("Invalid or malformed arguments received in function call response.", e);
        }
    }
}

isolated function getDefaultModelProvider() returns Wso2ModelProvider|Error {
    Wso2ModelProviderConfig? config = wso2ModelProviderConfig;
    if config is () {
        return error Error("Set the WSO2 model provider config in toml file");
    }
    return new Wso2ModelProvider(config);
}

isolated function getDefaultKnowledgeBase() returns VectorKnowledgeBase|Error {
    Wso2ModelProviderConfig? config = wso2ModelProviderConfig;
    if config is () {
        return error Error("Set the WSO2 model provider config in toml file");
    }
    EmbeddingProvider|Error wso2EmbeddingProvider = new Wso2EmbeddingProvider(config);
    if wso2EmbeddingProvider is Error {
        return error Error("error creatating default vector konwledge base");
    }
    return new VectorKnowledgeBase(new InMemoryVectorStore(), wso2EmbeddingProvider);
}


