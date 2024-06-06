import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { WatsonxAI } from "@langchain/community/llms/watsonx_ai";
import { formatDocumentsAsString } from "langchain/util/document";

import { RunnableSequence } from "@langchain/core/runnables";

import { PromptTemplate } from "@langchain/core/prompts";

export const generate = async (question: string) => {
    try {
        console.log("Generating");
        console.log("Question", question);
        const vectorStoreRetriever = await createVectorStoreRetriever();

        const model = new WatsonxAI({
            modelId: "meta-llama/llama-2-13b-chat",
            modelParameters: {
                max_new_tokens: 250,
                temperature: 0.3,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });

        console.log("Q: ", question);

        const response = await finalChain(
            model,
            vectorStoreRetriever,
            question
        );
        console.log("Response", response);

        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);

        return cleanedResponse;
    } catch (error) {
        console.log(error);
    }
};

const createVectorStoreRetriever = async () => {
    console.log("Creating Vector Store Retriever");
    // Load the vector store
    const vectorStore = await HNSWLib.load(
        __dirname + "/embeddings",
        new OpenAIEmbeddings()
    );
    return vectorStore.asRetriever();
};

const finalChain = async (
    model: WatsonxAI,
    retriever: any,
    question: string
) => {
    try {
        console.log("Final Chain");

        const qaSystemPrompt = PromptTemplate.fromTemplate(`
        If the user interaction is a question and directly pertains to the context of the uploaded document, provide an answer to the question.
        If the user interaction is a question but is not relevant to the content of the document, respond with "The question is out of context."
        If the user interaction is not a question (e.g., a statement or command), reply with "I can answer questions related to the document."
        Please ensure that the responses are accurate and contextually relevant to the document content.

        Context: {context}

        Question: {question}
        `);

        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const relevantDocs = await retriever.getRelevantDocuments(
                        input.question
                    );
                    console.log("Relevant Docs", relevantDocs);
                    return formatDocumentsAsString(relevantDocs);
                },
                question: (input) => input.question,
            },
            qaSystemPrompt,
            model,
        ]);

        return await ragChain.invoke({
            question,
        });
    } catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
};

const cleanResponse = (response: string) => {
    console.log("Cleaning Response");
    let res = response.replace(/\s\s+/g, " ");
    res = res.replace(/\s*AI:\s*/gm, "");
    res = res.replace(/(\r\n|\n|\r)/gm, "");
    res = res.trim();
    return res;
};
