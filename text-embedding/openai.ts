import OpenAI from "openai";
import 'dotenv/config'

export const { OPEN_AI_API_KEY, GROQ_API_KEY } = process.env
// console.log('OPEN_AI_API_KEY?', OPEN_AI_API_KEY)

const openai = new OpenAI({
    apiKey: OPEN_AI_API_KEY
});

export default openai