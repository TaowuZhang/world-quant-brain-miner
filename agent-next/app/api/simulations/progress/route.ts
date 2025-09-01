import { NextResponse } from 'next/server';
import { getStoredJWT } from '@/lib/auth';

export async function POST(request: Request) {
  const { searchParams } = new URL(request.url);
  const progressUrl = searchParams.get('url');
  const jwtToken = (await request.json()).jwtToken;
  if (!progressUrl) {
    return NextResponse.json({ error: 'Progress URL required' }, { status: 400 });
  }

  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  const checkProgress = async () => {
    try {
      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1500); // 1.5 second timeout

      const response = await fetch(progressUrl, {
        headers: {
          'Authorization': `Bearer ${jwtToken}`,
          'Cookie': `t=${jwtToken}`
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      console.log('response', response);

      if (response.status === 429) {
        await writer.write(encoder.encode('data: ' + JSON.stringify({ status: 'rate_limit' }) + '\n\n'));
        return;
      }

      const result = await response.json();
      const status = result.status;

      if (status === 'COMPLETE') {
        const alphaId = result.alpha;
        if (alphaId) {
          // Create another AbortController for alpha fetch with timeout
          const alphaController = new AbortController();
          const alphaTimeoutId = setTimeout(() => alphaController.abort(), 1500);

          const alphaResponse = await fetch(`https://api.worldquantbrain.com/alphas/${alphaId}`, {
            headers: {
              'Authorization': `Bearer ${jwtToken}`,
              'Cookie': `t=${jwtToken}`
            },
            signal: alphaController.signal
          });

          clearTimeout(alphaTimeoutId);

          if (alphaResponse.status === 200) {
            const alphaData = await alphaResponse.json();
            await writer.write(encoder.encode('data: ' + JSON.stringify({
              status: 'complete',
              result: {
                fitness: alphaData.is?.fitness || 0,
                sharpe: alphaData.is?.sharpe || 0,
                turnover: alphaData.is?.turnover || 0
              }
            }) + '\n\n'));
            await writer.close();
            return;
          }
        }
      } else if (status === 'ERROR') {
        await writer.write(encoder.encode('data: ' + JSON.stringify({
          status: 'error',
          error: result.message || 'Simulation failed'
        }) + '\n\n'));
        await writer.close();
        return;
      }

      // If not complete or error, continue checking
      await writer.write(encoder.encode('data: ' + JSON.stringify({
        status: 'in_progress',
        progress: result.progress || 0
      }) + '\n\n'));
    } catch (error) {
      // Handle timeout errors specifically
      if (error instanceof Error && error.name === 'AbortError') {
        await writer.write(encoder.encode('data: ' + JSON.stringify({
          status: 'timeout',
          error: 'Request timed out after 1.5 seconds'
        }) + '\n\n'));
      } else {
        await writer.write(encoder.encode('data: ' + JSON.stringify({
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        }) + '\n\n'));
      }
      await writer.close();
    }
  };

  // Start checking progress
  checkProgress();

  return new NextResponse(stream.readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
} 