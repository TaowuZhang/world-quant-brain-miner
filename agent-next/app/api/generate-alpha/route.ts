import { NextResponse } from 'next/server';
import { createChatCompletion } from '../../../lib/deepseek';
import pdfParse from 'pdf-parse';
import { Pinecone } from '@pinecone-database/pinecone';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get the index
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const pdfFile = formData.get('pdf') as File;
    const fields = formData.get('fields') as string;
    const operators = formData.get('operators') as string;

    if (!pdfFile || !fields || !operators) {
      return NextResponse.json({ error: 'PDF file, fields, and operators are required' }, { status: 400 });
    }

    // Read the PDF file and extract text using pdf-parse
    const pdfBuffer = await pdfFile.arrayBuffer();
    const pdfData = await pdfParse(Buffer.from(pdfBuffer));
    const pdfText = pdfData.text;

    // Generate embeddings for the PDF text and search for relevant data fields
    let relevantDataFields: Array<{
      id: string;
      score: number;
      name: string;
      category: string;
      description: string;
    }> = [];
    try {
      const embeddings = await pc.inference.embed(
        "multilingual-e5-large",
        [pdfText],
        {
          input_type: "passage",
          truncate: "END"
        }
      );

      const vector = (embeddings.data[0] as any).values;

      // Search for similar data fields in Pinecone, excluding news-related fields
      const searchResults = await index.namespace('data-fields').query({
        topK: 30, // Get more results to account for filtering
        vector,
        includeValues: false,
        includeMetadata: true,
        filter: {
          category: {
            $nin: ["News", "news", "NEWS", "News12", "news12", "NEWS12"]
          }
        }
      });

      relevantDataFields = searchResults.matches?.map(match => ({
        id: match.id,
        score: match.score || 0,
        name: String(match.metadata?.name || 'Unknown'),
        category: String(match.metadata?.category || 'Unknown'),
        description: String(match.metadata?.description || ''),
      })) || [];
    } catch (error) {
      console.error('Error searching for relevant data fields:', error);
      // Continue without relevant data fields if search fails
    }

    // Prepare the prompt for DeepSeek
    const relevantFieldsSection = relevantDataFields.length > 0 
      ? `\nRelevant Data Fields Found in Database (based on PDF content similarity):
${relevantDataFields.map(field => 
  `- ${field.name} (${field.category}): ${field.description} (Relevance Score: ${field.score?.toFixed(3)})`
).join('\n')}`
      : '';

    const prompt = `You are a financial research analyst specializing in alpha generation. Analyze the following research paper and generate alpha ideas based on the selected data fields and operators.

Research Paper Content:
${pdfText}

Selected Data Fields:
${fields}

Selected Operators:
${operators}${relevantFieldsSection}

Please generate 5 alpha ideas that combine insights from the research paper with the selected data fields and operators. Each idea should:
1. Be specific and actionable
2. Explain the rationale behind the idea
3. Suggest how to implement it using the selected fields and operators
4. Include potential risks or limitations
5. Consider incorporating relevant data fields from the database search results when appropriate

For each alpha idea, provide a strict alpha expression that implements the idea. The expression should:
1. Define all intermediate variables
2. Use proper alpha expression syntax (e.g., ts_mean, ts_std_dev, zscore, etc.)
3. Return a single value (if expression ends without semicolon)
4. Follow this format:
   - First, explain the idea and rationale
   - Then provide the complete alpha expression implementation
   - Finally, explain how the expression works

Example format:
{
  "title": "Idea title",
  "description": "Detailed description of the idea",
  "implementation": "How to implement using fields and operators",
  "rationale": "Why this idea might work",
  "risks": "Potential risks or limitations",
  "alpha_expression": "m_ret = group_mean(returns, rank(ts_mean(cap, 20)), market);\nhorro = abs(returns - m_ret) / (abs(returns)+abs(m_ret)+0.1);\nhorro_day = ts_mean(horro, 22);\nret_std = ts_std_dev(returns, 22);\nadj_ret = horro_day * ret_std * returns;\nadj_ret_mean = ts_mean(adj_ret, 22);\nadj_ret_std = ts_std_dev(adj_ret, 22);\nhorro_std_bonus = zscore(adj_ret_mean) + zscore(adj_ret_std);\n-horro_std_bonus"
}

IMPORTANT: Return ONLY a valid JSON array of objects with the following structure, with no additional text or markdown formatting:
[
  {
    "title": "Idea title",
    "description": "Detailed description of the idea",
    "implementation": "How to implement using fields and operators",
    "rationale": "Why this idea might work",
    "risks": "Potential risks or limitations",
    "alpha_expression": "Complete alpha expression implementation"
  }
]`;

    // Call DeepSeek API
    const response = await createChatCompletion([
      {
        role: 'system',
        content: 'You are a financial research analyst specializing in alpha generation. Always respond with valid JSON only.'
      },
      {
        role: 'user',
        content: prompt
      }
    ]);

    // Clean the response to ensure it's valid JSON
    const cleanedResponse = response
      .trim()
      .replace(/^```json\n?/, '')  // Remove opening ```json
      .replace(/\n```$/, '')       // Remove closing ```
      .trim();

    let alphaIdeas;
    
    try {
      alphaIdeas = JSON.parse(cleanedResponse);
    } catch (parseError: unknown) {
      console.error('Failed to parse response:', cleanedResponse);
      return NextResponse.json({ 
        error: 'Failed to parse alpha ideas from the response',
        details: parseError instanceof Error ? parseError.message : 'Unknown parsing error'
      }, { status: 500 });
    }

    return NextResponse.json(alphaIdeas);
  } catch (error) {
    console.error('Error in generate-alpha API route:', error);
    return NextResponse.json({ 
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS() {
  return NextResponse.json({}, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
} 