import { NextResponse } from 'next/server';
import { getStoredJWT } from '@/lib/auth';

export async function POST(request: Request) {
  try {
    const { alpha_expression, jwtToken } = await request.json();

    const simulationData = {
      type: 'REGULAR',
      settings: {
        instrumentType: 'EQUITY',
        region: 'USA',
        universe: 'TOP3000',
        delay: 1,
        decay: 0,
        neutralization: 'INDUSTRY',
        truncation: 0.08,
        pasteurization: 'ON',
        unitHandling: 'VERIFY',
        nanHandling: 'OFF',
        language: 'FASTEXPR',
        visualization: false,
      },
      regular: alpha_expression
    };

    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500); // 1.5 second timeout

    const response = await fetch('https://api.worldquantbrain.com/simulations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${jwtToken}`,
        'Cookie': `t=${jwtToken}`
      },
      body: JSON.stringify(simulationData),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (response.status === 401) {
      return NextResponse.json({ error: 'Authentication expired' }, { status: 401 });
    }

    if (response.status !== 201) {
      return NextResponse.json({ error: await response.text() }, { status: response.status });
    }

    const progressUrl = response.headers.get('location');
    if (!progressUrl) {
      return NextResponse.json({ error: 'No progress URL received' }, { status: 500 });
    }

    return NextResponse.json({ progress_url: progressUrl });
  } catch (error) {
    console.error('Error submitting simulation:', error);
    
    // Handle timeout errors specifically
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { error: 'Request timed out after 1.5 seconds' },
        { status: 408 }
      );
    }
    
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
} 