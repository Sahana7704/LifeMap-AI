import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 60; // Max allowed on Vercel Hobby plan
export const dynamic = 'force-dynamic';

async function handleProxy(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  const path = (resolvedParams.path || []).join('/');
  const isProduction = process.env.NODE_ENV === 'production';
  const configuredBackend = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
  if (!configuredBackend && isProduction) {
    return NextResponse.json(
      { error: 'NEXT_PUBLIC_API_URL or BACKEND_URL environment variable is required in production with no localhost fallback.' },
      { status: 500 }
    );
  }

  const backendBase = (
    configuredBackend ||
    'http://127.0.0.1:5000/api'
  ).replace(/\/api\/?$/, '');

  const search = req.nextUrl.search || '';
  const targetUrl = `${backendBase}/api/${path}${search}`;

  const headers = new Headers();
  req.headers.forEach((val, key) => {
    const k = key.toLowerCase();
    if (k !== 'host' && k !== 'connection' && k !== 'content-length') {
      headers.set(key, val);
    }
  });

  const abortController = new AbortController();
  const timeoutId = setTimeout(() => abortController.abort(), 90000); // 90s timeout

  try {
    const fetchOptions: RequestInit = {
      method: req.method,
      headers,
      signal: abortController.signal,
      redirect: 'manual',
    };

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      // Forward body as ArrayBuffer for multipart uploads and JSON
      const bodyBuffer = await req.arrayBuffer();
      if (bodyBuffer.byteLength > 0) {
        fetchOptions.body = bodyBuffer;
      }
    }

    const upstreamRes = await fetch(targetUrl, fetchOptions);
    clearTimeout(timeoutId);

    const resHeaders = new Headers();
    upstreamRes.headers.forEach((val, key) => {
      const k = key.toLowerCase();
      if (k !== 'transfer-encoding' && k !== 'content-encoding') {
        resHeaders.set(key, val);
      }
    });

    const resBody = await upstreamRes.arrayBuffer();
    return new NextResponse(resBody, {
      status: upstreamRes.status,
      statusText: upstreamRes.statusText,
      headers: resHeaders,
    });
  } catch (err: any) {
    clearTimeout(timeoutId);
    const isTimeout = err.name === 'AbortError' || err.message?.toLowerCase().includes('abort');
    console.error(`[Vercel Proxy Error] ${req.method} ${targetUrl}:`, err.message);

    if (isTimeout) {
      return NextResponse.json(
        {
          error: 'The AI backend service timed out while waking up from sleep (Render free tier).',
          details: 'Upstream gateway timeout after 90s.',
          retryable: true,
          code: 'COLD_START',
        },
        { status: 504 }
      );
    }

    return NextResponse.json(
      {
        error: 'Unable to reach backend AI service.',
        details: err.message,
        retryable: true,
        code: 'NETWORK_ERROR',
      },
      { status: 502 }
    );
  }
}

export const GET = handleProxy;
export const POST = handleProxy;
export const PUT = handleProxy;
export const DELETE = handleProxy;
export const OPTIONS = handleProxy;
