import openai from "./openai";
const tf = require('@tensorflow/tfjs');

// - // - // - // - // - // - // - // - 
// NOTE to Sahil: Clustering isn't much beneficial because of my lack of knowledge in machine learning and classififcaitons.
// - // - // - // - // - // - // - // - 

// ? LEARN: Using clustering algorithms like k-means, you can organize texts into groups.
// ===================================================================================
const { kmeans } = require('ml-kmeans'); // Install with: npm install ml-kmeans

async function textClustering(documents: string[], k = 2) {
    const embeddings = await Promise.all(documents.map(doc => getEmbedding(doc)));

    // Perform K-Means clustering
    const KMeanResult = kmeans(embeddings, k);

    // -----
    // Example usage:
    // embeddings would be the embeddings of your text data
    // centroids are from your K-means clustering result
    const nearest = findNearestSamples(embeddings, KMeanResult.centroids, documents, k);
    nearest.forEach(item => {
        console.log(item)
    })

    console.log('number of centroids?', KMeanResult.centroids.length)
    console.log('clusters?', KMeanResult.clusters);
}

async function getEmbedding(text: string) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
        encoding_format: 'float' // "base64", "float" (default)
    });
    return response.data[0].embedding
}


// Function to calculate Euclidean distance between two vectors
function euclideanDistance(vec1: number[], vec2: number[]) {
    return Math.sqrt(tf.sub(vec1, vec2).square().sum().arraySync());
}

// Find nearest text samples to each centroid
function findNearestSamples(embeddings: number[][], centroids: number[][], documents: string[], numSamples = 3) {
    let nearestSamples: { centroid: number; nearestTexts: string[]; }[] = [];

    centroids.forEach((centroid, cIndex) => {
        let distances = embeddings.map((embedding, eIndex) => {
            return {
                index: eIndex,
                distance: euclideanDistance(embedding, centroid)
            };
        });

        // Sort by distance and pick top numSamples
        distances.sort((a, b) => a.distance - b.distance);
        // nearestSamples.push({
        //     centroid: cIndex,
        //     nearestTexts: distances.slice(0, numSamples).map(d => d.index)
        // });

        nearestSamples.push({
            centroid: cIndex,
            nearestTexts: distances.slice(0, numSamples).map(d => documents[d.index]) // Get actual text from index
        });
    });

    return nearestSamples;
}

// Example usage
const docs = [
    "I love playing football.",
    "Soccer is my favorite sport.",
    "Python is great for programming.",
    "I enjoy coding in JavaScript.",
    "People are good at learning.",
    "Talking to individuals help develop social skills."
];
// From sahil: If I use k = 2 below then above docs are distinguished into numbers like 0 and 1.
//             If I use k = 3 below then above docs are distinguished into numbers like 0, 1 and 2.
textClustering(docs, 3)
// From sahil: k-means is useful when we want to divide a set of documents into n parts dependening on their relatedness.
