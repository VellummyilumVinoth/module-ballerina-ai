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

# Represents a dense vector with floating-point values.
public type Vector float[];

# Represents a sparse vector storing only non-zero values with their corresponding indices.
#
# + indices - Array of indices where non-zero values are located 
# + values - Array of non-zero floating-point values corresponding to the indices
public type SparseVector record {|
    int[] indices;
    Vector values;
|};

# Represents a hybrid embedding containing both dense and sparse vector representations..
#
# + dense - Dense vector representation of the embedding
# + sparse - Sparse vector representation of the embedding
public type HybridVector record {|
    Vector dense;
    SparseVector sparse;
|};

# Union type representing all possible embedding vector formats.
public type EmbeddingVector Vector|SparseVector|HybridVector;

# Enumeration of supported operators for Pinecone metadata filtering.
# These operators define how metadata values should be compared during vector searches.
public enum PineconeOperator {
    EQUALS,
    NOT_EQUALS,
    GREATER_THAN,
    LESS_THAN,
    GREATER_THAN_OR_EQUAL,
    LESS_THAN_OR_EQUAL,
    IN,
    NOT_IN
}

# Enumeration of logical conditions for combining multiple metadata filters.
# Defines how multiple filter conditions should be combined in vector searches.
public enum PineconeCondition {
    AND, OR
}

# Metadata filter for vector search operations.
# Defines conditions to filter vectors based on their associated metadata values.
#
# + key - The metadata field name to filter on
# + operator - The comparison operator to use (optional, defaults to EQUALS)
# + value - The value to compare against
public type MetadataFilter record {|
    string key;
    PineconeOperator operator?; // "==", "!=", ">", "<", ">=", "<=", "in", "nin"
    anydata value;
|};

# Container for multiple metadata filters with logical combination.
# Allows complex filtering by combining multiple conditions with AND/OR logic.
#
# + filter - Array of individual metadata filters to apply
# + condition - Logical operator to combine filters (optional, defaults to AND)
public type MetadataFilters record {|
    MetadataFilter[] filter?;
    PineconeCondition condition?; // "and", "or"
|};

# Represents a complete vector store query with embedding and optional filters.
# Combines the query vector with metadata filters for precise search operations.
#
# + embeddingVector - The vector to search for similar matches
# + filters - Optional metadata filters to narrow down search results
public type VectorStoreQuery record {|
    EmbeddingVector embeddingVector;
    MetadataFilters filters?;
|};

# Represents a document with content and optional metadata..
#
# + content - The textual content of the document 
# + metadata - Optional key-value pairs containing additional information about the document
public type Document record {|
    string content;
    map<anydata> metadata?;
|};

# Represents a vector entry combining an embedding with its source document.
#
# + embedding - The vector representation of the document content  
# + document - The original document associated with this embedding
public type VectorEntry record {|
    EmbeddingVector embedding;
    Document document;
|};

# Represents a vector match result with similarity score.
#
# + score - Similarity score indicating how closely the vector matches the query 
public type VectorMatch record {|
    *VectorEntry;
    float score;
|};

# Represents a document match result with similarity score.
#
# + document - The matched document  
# + score - Similarity score indicating document relevance to the query
public type DocumentMatch record {|
    Document document;
    float score;
|};

# Represents prompt templates for RAG (Retrieval-Augmented Generation) operations.
# Prompts structure how context documents and user queries are formatted and presented
# to language models for generating contextually relevant responses.
#
# + systemPrompt - System-level instructions that define the model's behavior, role, and response format
# + userPrompt - The user's question or query that needs to be answered using the provided context
public type Prompt record {|
    string systemPrompt?;
    string userPrompt?;
|};

# Enumeration of vector store query modes.
# Defines different search strategies for retrieving relevant documents
# based on the type of embeddings and search algorithms to be used.
public enum VectorStoreQueryMode {
    DENSE,
    SPARSE,
    HYBRID
};

# Interface for vector storage and retrieval operations.
# Vector stores provide persistence and search capabilities for embeddings.
public type VectorStore isolated object {
    # Adds vector entries to the store.
    #
    # + entries - Array of vector entries to be stored
    # + return - An error if the operation fails, otherwise nil
    public isolated function add(VectorEntry[] entries) returns Error?;

    # Searches for similar vectors in the store.
    #
    # + queryEmbedding - The query embedding to search for
    # + return - Array of matching vectors with similarity scores or an error if search fails
    public isolated function query(VectorStoreQuery queryEmbedding) returns VectorMatch[]|Error;
};

# Interface for embedding providers.
# Embedding providers convert text into vector representations for similarity search.
public type EmbeddingProvider isolated client object {
    # Converts text into an embedding vector.
    #
    # + document - The input text to be embedded
    # + return - The embedding vector representation or an error if embedding fails
    isolated remote function embed(string document) returns EmbeddingVector|Error;
};

# Document retriever for finding relevant documents based on query similarity.
# Retriever combines embedding generation and vector search to return
public isolated class Retriever {
    private final VectorStore vectorStore;
    private final EmbeddingProvider embeddingModel;

    # Initializes a new retriever instance.
    # Sets up the retriever with the necessary components for
    # query embedding and vector search operations.
    #
    # + vectorStore - The vector store to search in
    # + embeddingProvider - The embedding provider to use for query embedding
    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel) {
        self.vectorStore = vectorStore;
        self.embeddingModel = embeddingModel;
    }

    # Retrieves relevant documents for a given query.
    # Embeds the query text and searches for similar vectors,
    # returning matching documents with similarity scores.
    #
    # + query - The text query to search for
    # + return - Array of matching documents with scores or an error if retrieval fails
    public isolated function retrieve(string query) returns DocumentMatch[]|Error {
        EmbeddingVector queryEmbedding = check self.embeddingModel->embed(query);
        VectorStoreQuery vectorStoreQuery = {
            embeddingVector: queryEmbedding
        };
        VectorMatch[] matches = check self.vectorStore.query(vectorStoreQuery);
        return from VectorMatch 'match in matches
            select {document: 'match.document, score: 'match.score};
    }
}

# Vector knowledge base for managing document indexing and retrieval operations.
# The vector knowledge base handles the process of converting documents to embeddings 
# and storing them for retrieval.
public isolated class VectorKnowledgeBase {
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

    # Initializes a new vector index.
    # Creates a vector index with the specified storage and embedding capabilities.
    # The index manages the entire lifecycle from document ingestion to retrieval.
    #
    # + vectorStore - The vector store for persistence
    # + embeddingProvider - The embedding provider for vectorization
    public isolated function init(VectorStore vectorStore, EmbeddingProvider embeddingModel) {
        self.vectorStore = vectorStore;
        self.embeddingModel = embeddingModel;
        self.retriever = new Retriever(vectorStore, embeddingModel);
    }

    # Indexes a collection of documents.
    # Converts documents to embeddings and stores them in the vector store.
    # This operation makes the documents searchable through the retriever.
    #
    # + documents - Array of documents to be indexed
    # + return - An error if indexing fails, otherwise nil
    public isolated function index(Document[] documents) returns Error? {
        VectorEntry[] entries = [];
        foreach Document document in documents {
            EmbeddingVector embedding = check self.embeddingModel->embed(document.content);
            VectorEntry entry = {embedding, document};
            // generate sparse vectors
            entries.push(entry);
        }
        check self.vectorStore.add(entries);
    }

    # Returns the retriever instance for this index.
    # Provides access to the retriever for performing document searches against the indexed document collection.
    #
    # + return - The retriever instance
    public isolated function getRetriever() returns Retriever {
        return self.retriever;
    }
}

# Interface for building prompts from context documents and queries.
# Prompt builders structure how retrieved documents and user queries 
# are formatted for presentation to language models in RAG systems.
public type RagPromptBuilder isolated object {
    # Builds a prompt from context documents and a query.
    # Combines retrieved documents with the user query to create
    # structured prompts for language model processing.
    #
    # + contextDocuments - Array of relevant documents retrieved for the query
    # + query - The user's original query or question
    # + return - A structured prompt ready for language model consumption
    public isolated function build(Document[] contextDocuments, string query) returns Prompt;
};

# # Default implementation of prompt builder.
# Provides a standard template for combining context documents with user queries.
# Creates system prompts that instruct the model to answer based on provided context.
public isolated class RagDefaultPromptBuilder {
    *RagPromptBuilder;

    # Builds a default prompt template.
    # Creates a system prompt with context documents and a user prompt with the query.
    # The format follows common RAG patterns for context-aware question answering.
    #
    # + contextDocuments - Array of relevant documents to include as context
    # + query - The user's question to be answered
    # + return - A prompt with system instructions and user query
    public isolated function build(Document[] contextDocuments, string query) returns Prompt {
        // following is a sample implementation
        string systemPrompt = string `Answer the question based on the following provided context: `
            + string `<CONTEXT>${string:'join("\n", ...contextDocuments.'map(doc => doc.content))}</CONTEXT>"""`;
        string userPrompt = "Question:\n" + query;
        return {systemPrompt, userPrompt};
    }
}

# WSO2 model provider implementation.
# Provides chat completion capabilities using WSO2's language model services.
# This is a concrete implementation of the ModelProvider interface.
public isolated client class Wso2ModelProvider {
    *ModelProvider;

    # Processes chat messages and returns assistant response.
    # Handles conversation context and optional tool integration for enhanced responses.
    #
    # + messages - Array of chat messages for conversation context
    # + tools - Array of available functions/tools for the model
    # + stop - Optional stop sequence for response generation
    # + return - Assistant message response or LLM error
    isolated remote function chat(ChatMessage[] messages, ChatCompletionFunctions[] tools, string? stop)
    returns ChatAssistantMessage|LlmError {
        return {role: ASSISTANT, content: "Dummy response"};
    }
}

# In-memory vector store implementation.
# Provides a simple in-memory storage solution for vector entries.
# Suitable for development, testing, or small-scale applications.
public isolated class InMemoryVectorStore {
    *VectorStore;
    private final VectorEntry[] entries = [];

    # Adds vector entries to the in-memory store.
    # Only supports dense vectors in this implementation.
    #
    # + entries - Array of vector entries to store
    # + return - Error if non-dense vectors are provided, otherwise nil
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

    # Queries the vector store for similar vectors.
    # Uses cosine similarity for dense vector comparison.
    #
    # + query - The query embedding vector
    # + return - Array of vector matches sorted by similarity score or error
    public isolated function query(VectorStoreQuery query) returns VectorMatch[]|Error {
        if query.embeddingVector !is Vector {
            return error Error("InMemoryVectorStore implementation only supports dense vectors");
        }

        lock {
            VectorMatch[] results = [];
            foreach var entry in self.entries {
                float similarity = self.cosineSimilarity(<Vector>query.embeddingVector.clone(), <Vector>entry.embedding);
                results.push({document: entry.document, embedding: entry.embedding, score: similarity});
            }
            var sorted = from var entry in results
                order by entry.score descending
                select entry;
            return sorted.clone();
        }
    }

    # Calculates cosine similarity between two dense vectors.
    # Cosine similarity measures the cosine of the angle between two vectors.
    #
    # + a - First vector for comparison
    # + b - Second vector for comparison  
    # + return - Cosine similarity score between 0 and 1
    isolated function cosineSimilarity(Vector a, Vector b) returns float {
        if a.length() != b.length() {
            return 0.0;
        }

        float dot = 0.0; // Dot product
        float normA = 0.0; // Norm of vector A
        float normB = 0.0; // Norm of vector B

        foreach int i in 0 ..< a.length() {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        float denom = normA.sqrt() * normB.sqrt();
        return denom == 0.0 ? 0.0 : dot / denom;
    }
}

# WSO2 embedding provider implementation.
# Provides text embedding capabilities using WSO2's embedding model services.
# This is a concrete implementation of the EmbeddingProvider interface.
public isolated client class Wso2EmbeddingProvider {
    *EmbeddingProvider;

    # Converts document text to embedding vector.
    # Transforms textual content into numerical vector representation for similarity search.
    #
    # + document - The text document to embed
    # + return - Empty embedding vector or error
    isolated remote function embed(string document) returns EmbeddingVector|Error {
        return [];
    }
}

# RAG (Retrieval-Augmented Generation) query engine.
# The RAG class orchestrates the entire RAG pipeline: document retrieval,
# prompt construction, and language model generation to answer user queries.
public isolated class Rag {
    private final ModelProvider model;
    private final VectorKnowledgeBase knowledgeBase;
    private final RagPromptBuilder ragPromptBuilder;

    # Initializes a new RAG query engine.
    # Sets up the complete RAG pipeline with model provider, knowledge base, and prompt builder.
    #
    # + modelProvider - The language model provider for response generation
    # + knowledgeBase - The vector knowledge base containing searchable documents
    # + ragPromptBuilder - Optional custom RAG prompt builder (defaults to DefaultRagPromptBuilder)
    public isolated function init(ModelProvider model = new Wso2ModelProvider(),
            VectorKnowledgeBase knowledgeBase = new VectorKnowledgeBase(new InMemoryVectorStore(),
                new Wso2EmbeddingProvider()
            ),
            RagPromptBuilder ragPromptBuilder = new DefaultRagPromptBuilder()
            ) {
        self.model = model;
        self.knowledgeBase = knowledgeBase;
        self.ragPromptBuilder = ragPromptBuilder;
    }

    # Processes a query through the complete RAG pipeline.
    # Retrieves relevant documents, builds context-aware prompts, and generates responses.
    #
    # + query - The user's question or query
    # + return - The generated response or an error if processing fails
    public isolated function query(string query) returns string|Error {
        DocumentMatch[] contextDocuments = check self.knowledgeBase.getRetriever().retrieve(query);
        Prompt prompt = self.ragPromptBuilder.build(contextDocuments.'map(ctx => ctx.document), query);

        ChatMessage[] messages = self.mapPromptToChatMessages(prompt);

        ChatAssistantMessage response = check self.model->chat(messages, []);
        return response.content ?: error Error("Unable to obtain valid response from model");
    }

    # Ingests documents into the knowledge base.
    # Processes and indexes documents to make them searchable for future queries.
    #
    # + documents - Array of documents to ingest
    # + return - Error if ingestion fails, otherwise nil
    public isolated function ingest(Document[] documents) returns Error? {
        return self.knowledgeBase.index(documents);
    }

    # Converts a prompt to chat message format.
    # Transforms structured prompts into the chat message format expected by language models.
    #
    # + prompt - The prompt to convert
    # + return - Array of chat messages ready for model consumption
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
}

# Default RAG prompt builder implementation.
# Provides a standard template for combining context documents with user queries.
# Creates structured prompts that guide language models to answer based on retrieved context.
public isolated class DefaultRagPromptBuilder {
    *RagPromptBuilder;

    # Builds a default prompt template for RAG operations.
    # Creates a system prompt with context documents and a user prompt with the query.
    # Uses a standard format that instructs the model to answer based on provided context.
    #
    # + context - Array of relevant documents to include as context
    # + query - The user's question to be answered
    # + return - A prompt with system instructions and user query
    public isolated function build(Document[] context, string query) returns Prompt {
        string systemPrompt = string `Answer the question based on the following provided context: `
            + string `<CONTEXT>${string:'join("\n", ...context.'map(doc => doc.content))}</CONTEXT>"""`;

        string userPrompt = "Question:\n" + query;

        return {systemPrompt, userPrompt};
    }
}
