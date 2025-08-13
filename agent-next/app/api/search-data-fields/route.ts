import { NextRequest, NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import pdfParse from 'pdf-parse';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get the index
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const pdfFile = formData.get('pdf') as File;
    const topK = parseInt(formData.get('topK') as string) || 10;

    if (!pdfFile) {
      return NextResponse.json({ error: 'PDF file is required' }, { status: 400 });
    }

    // Read the PDF file and extract text using pdf-parse
    const pdfBuffer = await pdfFile.arrayBuffer();
    const pdfData = await pdfParse(Buffer.from(pdfBuffer));
    const pdfText = pdfData.text;

    // Generate embeddings for the PDF text using Pinecone's inference API
    const embeddings = await pc.inference.embed(
      "multilingual-e5-large",
      [pdfText],
      {
        input_type: "passage",
        truncate: "END"
      }
    );

    // Extract the vector from the embeddings response
    const vector = (embeddings.data[0] as any).values;

    // Search for similar data fields in Pinecone, excluding news-related fields
    const searchResults = await index.namespace('data-fields').query({
      topK: topK * 2, // Get more results to account for filtering
      vector,
      includeValues: false,
      includeMetadata: true,
      filter: {
        category: {
          $nin: ["News", "news", "NEWS", "News12", "news12", "NEWS12"]
        }
      }
    });

    // Format the results
    const formattedResults = searchResults.matches?.map(match => ({
      id: match.id,
      score: match.score || 0,
      name: String(match.metadata?.name || 'Unknown'),
      category: String(match.metadata?.category || 'Unknown'),
      description: String(match.metadata?.description || ''),
      timestamp: String(match.metadata?.timestamp || '')
    })) || [];

    return NextResponse.json({
      success: true,
      results: formattedResults,
      totalFound: formattedResults.length,
      pdfTextLength: pdfText.length
    });

  } catch (error) {
    console.error('Error in search-data-fields API route:', error);
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
