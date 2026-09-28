import { createSseParser } from './sseParser';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

/**
 * POST a question and invoke `onEvent({ type, data })` for each streamed event.
 * Resolves when the stream ends; rejects on HTTP or network errors.
 */
export async function streamChat(question, { onEvent, signal } = {}) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await describeError(response));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const push = createSseParser(onEvent);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    push(value);
  }
}

async function describeError(response) {
  try {
    const body = await response.json();
    const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
    return detail || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}
