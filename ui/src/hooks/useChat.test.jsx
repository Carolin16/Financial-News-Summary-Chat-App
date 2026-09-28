import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useChat } from './useChat';

const citation = { number: 1, title: 'Intel jumps', link: 'https://x/1', is_partial: false };

function fakeSend(events, error) {
  return async (question, { onEvent }) => {
    events.forEach((event) => onEvent(event));
    if (error) throw error;
  };
}

describe('useChat', () => {
  it('shows the draft, then replaces it with the verified final answer', async () => {
    const send = fakeSend([
      { type: 'sources', data: { sources: [citation] } },
      { type: 'delta', data: { text: 'Draft with $999 ' } },
      { type: 'final', data: { answer: 'Intel rose [1].', citations: [citation] } },
    ]);
    const { result } = renderHook(() => useChat(send));

    await act(() => result.current.ask('Intel?'));

    const [user, assistant] = result.current.messages;
    expect(user).toMatchObject({ role: 'user', text: 'Intel?' });
    expect(assistant).toMatchObject({ text: 'Intel rose [1].', status: 'done' });
    expect(result.current.isStreaming).toBe(false);
  });

  it('surfaces request failures on the assistant message', async () => {
    const { result } = renderHook(() => useChat(fakeSend([], new Error('Request failed (422)'))));

    await act(() => result.current.ask('x'));

    expect(result.current.messages[1]).toMatchObject({
      error: 'Request failed (422)',
      status: 'failed',
    });
  });
});
