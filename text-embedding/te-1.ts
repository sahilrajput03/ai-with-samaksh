// Text Embedding
import openai from "./openai";

// Check this ChatGPT link for below q/a's - https://chatgpt.com/c/694191f9-6cb4-8324-8c2a-7822147e4f9c
//      0. What is rag in ai. be concise.
//      1. What is vectorized data?, what is embeddings in openai?
//      2. Can i get embeddings directly without calling openai's API and would be too much compute or is it that openai has secret sauce which I can't do locally?

async function semanticSearch(query: string, documents: string[]) {
    // Get embedding for the query
    const queryEmbedding = await getEmbedding(query);

    // Get embeddings for the documents
    const docEmbeddings = await Promise.all(documents.map((doc) => getEmbedding(doc)));

    // Compute cosine similarity between query and each document
    const similarities = docEmbeddings.map((embedding, index) => {
        return { doc: documents[index], similarity: cosineSimilarity(queryEmbedding, embedding) };
    });

    // Sort by highest similarity
    similarities.sort((a, b) => b.similarity - a.similarity);

    return similarities;
}

async function getEmbedding(text: string) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
        encoding_format: 'float' // "base64", "float" (default)
    });
    return response.data[0].embedding
}

function cosineSimilarity(vecA: number[], vecB: number[]) {
    const dotProduct = vecA.reduce((acc, val, idx) => acc + val * vecB[idx], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val ** 2, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val ** 2, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

// Example usage
const documents = ["Apple is a fruit.", "Google is a tech company.", "I like pizza."];
semanticSearch("Which company makes phones?", documents).then(console.log)
/**
 * Sample Output:
[
  { doc: 'Google is a tech company.', similarity: 0.8221700367394797 },
  { doc: 'Apple is a fruit.', similarity: 0.7888194890884669 },
  { doc: 'I like pizza.', similarity: 0.7353277402970958 }
]
 */