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

public isolated class QueryEngine {
    private final ModelProvider model;
    private final VectorIndex index;
    private final PromptBuilder promptBuilder;

    public isolated function init(ModelProvider model, VectorIndex vectorIndex,
            PromptBuilder promptBuilder = new DefaultPromptBuilder()) {
        self.model = model;
        self.index = vectorIndex;
        self.promptBuilder = promptBuilder;
    }

    public isolated function query(string query) returns string|Error {
        DocumentMatch[] context = check self.index.getRetriever().retrieve(query);
        // later when we allow re-reankers we can use the score in the document match
        Prompt prompt = self.promptBuilder.build(context.'map(ctx => ctx.document), query);
        ChatMessage[] messages = self.mapPromptToChatMessages(prompt);
        ChatAssistantMessage response = check self.model->chat(messages, []);
        return response.content ?: error Error("Unable to obtain valid answer");
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

public type Prompt record {|
    string systemPrompt?;
    string userPrompt?;
|};

public type PromptBuilder isolated object {
    public isolated function build(Document[] context, string query) returns Prompt;
};

public isolated class DefaultPromptBuilder {
    *PromptBuilder;

    public isolated function build(Document[] context, string query) returns Prompt {
        // following is a sample implementation
        string systemPrompt = string `Answer the question based on the following provided context: `
            + string `<CONTEXT>${string:'join("\n", ...context.'map(doc => doc.content))}</CONTEXT>"""`;
        string userPrompt = "Question:\n" + query;
        return {systemPrompt, userPrompt};
    }
}

public isolated class VectorIndex {
    // Need hybrid index or seperate vector store (one for dense and one for sparse)? then we need to change the init API
    // If we have seperate vector store we need to provide seperate embedding modesl for each vector store (one for dense one for sparse); 
    // How to assosiate vector store with embedding model?

    // Or do we need to compe up with hierarchy of vector indexes?
    // VectorIndex
    // - HybridVectorIndex
    // - DefaultVectorIndex
    //      - SparseVectorIndex
    //      - DenseVectorIndex
    private final VectorStore vectorStore;
    private final EmbeddingProvider embeddingModel;
    private final Retriever retriever;

    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel, Retriever retriever) {
        self.embeddingModel = embeddingModel;
        self.retriever = retriever;
        self.vectorStore = vectorStore;
    }

    public isolated function index(Document[] documents) returns Error? {
        VectorEntry[] entries = [];
        foreach var document in documents {
            float[]|SparseVector|Embedding embedding = check self.embeddingModel.embed(document.content);
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

public type EmbeddingProvider isolated object {
    public isolated function embed(string document) returns Vector|SparseVector|Embedding|Error;
};

public isolated class Retriever {
    private final VectorStore vectorStore;
    private final EmbeddingProvider embeddingModel;

    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel) {
        self.vectorStore = vectorStore;
        self.embeddingModel = embeddingModel;
    }

    public isolated function retrieve(string query) returns DocumentMatch[]|Error {
        Vector|SparseVector|Embedding embedding = check self.embeddingModel.embed(query);
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
