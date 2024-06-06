"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generate = void 0;
const hnswlib_1 = require("@langchain/community/vectorstores/hnswlib");
const openai_1 = require("@langchain/openai");
const watsonx_ai_1 = require("@langchain/community/llms/watsonx_ai");
const document_1 = require("langchain/util/document");
const runnables_1 = require("@langchain/core/runnables");
const prompts_1 = require("@langchain/core/prompts");
const generate = (question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Generating");
        console.log("Question", question);
        const vectorStoreRetriever = yield createVectorStoreRetriever();
        const model = new watsonx_ai_1.WatsonxAI({
            modelId: "meta-llama/llama-2-13b-chat",
            modelParameters: {
                max_new_tokens: 250,
                temperature: 0.5,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });
        console.log("Q: ", question);
        const response = yield finalChain(model, vectorStoreRetriever, question);
        console.log("Response", response);
        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);
        return cleanedResponse;
    }
    catch (error) {
        console.log(error);
    }
});
exports.generate = generate;
const createVectorStoreRetriever = () => __awaiter(void 0, void 0, void 0, function* () {
    console.log("Creating Vector Store Retriever");
    // Load the vector store
    const vectorStore = yield hnswlib_1.HNSWLib.load(__dirname + "/embeddings", new openai_1.OpenAIEmbeddings());
    return vectorStore.asRetriever();
});
const finalChain = (model, retriever, question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Final Chain");
        const qaSystemPrompt = `
        If the user interaction is a question and directly pertains to the context of the uploaded document, provide an answer to the question.
        If the user interaction is a question but is not relevant to the content of the document, respond with "The question is out of context."
        If the user interaction is not a question (e.g., a statement or command), reply with "I can answer questions related to the document."
        Please ensure that the responses are accurate and contextually relevant to the document content.

        Context delimited by triple double quotes
        """{context}"""

        `;
        const qaPrompt = prompts_1.ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            ["human", "{question}"],
        ]);
        const ragChain = runnables_1.RunnableSequence.from([
            {
                context: (input) => __awaiter(void 0, void 0, void 0, function* () {
                    const relevantDocs = yield retriever.getRelevantDocuments(input.question);
                    console.log("Relevant Docs", relevantDocs);
                    return (0, document_1.formatDocumentsAsString)(relevantDocs);
                }),
                question: (input) => input.question,
            },
            qaPrompt,
            model,
        ]);
        return yield ragChain.invoke({
            question,
        });
    }
    catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
});
const cleanResponse = (response) => {
    console.log("Cleaning Response");
    let res = response.trim();
    res = response.replace(/\s*AI:\s*/gm, "");
    res = res.replace(/(\r\n|\n|\r)/gm, "");
    return res;
};
//# sourceMappingURL=generate.js.map