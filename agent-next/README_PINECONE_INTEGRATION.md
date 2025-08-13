# Pinecone Integration for Alpha Generator

This document explains the new Pinecone vector database integration that enhances the alpha generation process by finding relevant data fields based on uploaded PDF research papers.

## Overview

The alpha generator now includes a feature that searches the Pinecone vector database for relevant data fields based on the content of uploaded PDF research papers. This provides more context and relevant data fields to improve the quality of generated alpha ideas.

## Features

### 1. PDF Content Analysis
- Extracts text content from uploaded PDF research papers
- Generates embeddings using Pinecone's multilingual-e5-large model
- Searches the `data-fields` namespace in the Pinecone database

### 2. Relevant Data Fields Discovery
- Finds up to 15 most relevant data fields based on PDF content similarity
- Displays relevance scores for each field
- Shows field categories and descriptions
- Integrates discovered fields into the alpha generation prompt

### 3. Enhanced Alpha Generation
- Incorporates relevant data fields from the database search
- Provides more context to the AI model for better alpha ideas
- Maintains the original selected fields and operators

## API Endpoints

### `/api/search-data-fields`
**Method:** POST  
**Purpose:** Search for relevant data fields based on PDF content

**Request:**
- `pdf`: PDF file (FormData)
- `topK`: Number of results to return (optional, default: 10)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "field_id",
      "score": 0.85,
      "name": "Field Name",
      "category": "Fundamental",
      "description": "Field description",
      "timestamp": "2025-01-01T00:00:00.000Z"
    }
  ],
  "totalFound": 15,
  "pdfTextLength": 5000
}
```

### Updated `/api/generate-alpha`
**Method:** POST  
**Purpose:** Generate alpha ideas with enhanced context from Pinecone search

**Enhancements:**
- Automatically searches for relevant data fields
- Includes discovered fields in the generation prompt
- Provides more comprehensive context for better alpha ideas

## Usage

### 1. Upload PDF Research Paper
- Upload a PDF file containing financial research content
- The system will extract text content for analysis

### 2. Search for Relevant Data Fields
- Click the "Search Database" button to find relevant data fields
- Review the discovered fields with their relevance scores
- The system will show up to 15 most relevant fields

### 3. Generate Alpha Ideas
- Select your desired fields and operators as before
- Click "Generate Alpha Ideas"
- The system will now include relevant data fields from the database search in the generation process

## Environment Variables

Make sure to set the following environment variables:

```bash
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=worldquant-miner
```

## Database Structure

The integration expects the following structure in the Pinecone database:

**Index:** `worldquant-miner`  
**Namespace:** `data-fields`

**Vector Metadata:**
- `name`: Field name
- `category`: Field category (e.g., "Fundamental", "Market", "Analyst")
- `description`: Field description
- `timestamp`: Creation timestamp

## Technical Details

### Embedding Model
- Uses Pinecone's `multilingual-e5-large` model
- Input type: "passage"
- Truncation: "END"

### Search Configuration
- Top-K: 30 results (configurable, increased to account for filtering)
- Include metadata: true
- Include values: false
- Namespace: "data-fields"
- Filter: Excludes news-related data fields (News, news, NEWS, News12, news12, NEWS12)

### Error Handling
- Graceful fallback if Pinecone search fails
- Continues alpha generation without relevant fields if search fails
- Comprehensive error logging

## Benefits

1. **Better Context:** Provides relevant data fields based on research content
2. **Improved Quality:** More informed alpha generation with database context
3. **Discovery:** Helps users discover relevant fields they might not have considered
4. **Efficiency:** Automates the process of finding relevant data fields

## Troubleshooting

### Common Issues

1. **No relevant fields found:**
   - Check if the PDF contains financial research content
   - Verify the Pinecone database has data in the `data-fields` namespace
   - Ensure the API key has proper permissions

2. **Search fails:**
   - Verify `PINECONE_API_KEY` is set correctly
   - Check network connectivity to Pinecone
   - Review server logs for detailed error messages

3. **Poor relevance scores:**
   - The PDF content might not be financial research
   - Consider uploading a different research paper
   - Check if the database contains relevant financial data fields

### Debug Information

The system provides debug information including:
- PDF text length
- Number of fields found
- Relevance scores for each field
- Error details if search fails
