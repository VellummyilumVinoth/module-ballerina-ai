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

    public isolated function init(ModelProvider model = new Wso2ModelProvider(),
            VectorKnowledgeBase knowledgeBase = new VectorKnowledgeBase(new InMemoryVectorStore(), new Wso2EmbeddingProvider()),
            RagPromptBuilder promptBuilder = new DefaultRagPromptBuilder()) {
        self.model = model;
        self.knowledgeBase = knowledgeBase;
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

    isolated remote function embed(string document) returns Vector|SparseVector|Embedding|Error {
        return [];
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
            VectorMatch[] results = [];
            foreach var entry in self.entries {
                float similarity = self.cosineSimilarity(query.clone(), <Vector>entry.embedding);
                results.push({document: entry.document, embedding: entry.embedding, score: similarity});
            }
            var sorted = from var entry in results
                order by entry.score
                select entry;
            return sorted.clone();
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

public isolated client class Wso2ModelProvider {
    *ModelProvider;

    isolated remote function chat(ChatMessage[] messages, ChatCompletionFunctions[] tools, string? stop)
    returns ChatAssistantMessage|LlmError {
        return {role: ASSISTANT, content: "Dummy response"};
    }
}
