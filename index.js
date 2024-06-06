// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { OpenAIEmbeddings } from "@langchain/openai";
import dotenv from "dotenv";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import pdfUtil from "pdf-to-text";

dotenv.config();

const generateEmbeddings = async () => {
    try {
        const embeddingsModel = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY,
            maxConcurrency: 5,
        });
        console.log("Generating embeddings.");
        // const embeddingsModel = new HuggingFaceTransformersEmbeddings({
        //     model: "Xenova/all-MiniLM-L6-v2",
        // });

        // Text data path
        const DATA_FILE_PATH = "./data/data.pdf";

        let data = "";

        await new Promise((resolve, reject) => {
            pdfUtil.pdfToText(DATA_FILE_PATH, async function (err, pdfData) {
                if (err) throw err;
                data = pdfData;

                resolve();
            });
        });

        const textSplitterChat = new RecursiveCharacterTextSplitter({
            chunkSize: 2000,
            chunkOverlap: 500,
        });

        const docs = await textSplitterChat.createDocuments([data]);

        const vectorStore = await HNSWLib.fromDocuments(docs, embeddingsModel);
        await vectorStore.save("./rag/src/embeddings");
        await vectorStore.save("./rag/bin/embeddings");
    } catch (err) {
        console.log(err);
    }
};

await generateEmbeddings();
