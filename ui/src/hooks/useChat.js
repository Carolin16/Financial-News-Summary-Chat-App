import { useCallback, useRef, useState } from 'react';
import { streamChat } from '../services/chatApi';

let nextId = 0;
const newId = () => `m${(nextId += 1)}`;

/**
 * Chat state: a list of user/assistant messages. The assistant message shows the streamed
 * draft until the server's verified `final` event replaces it.
 */
export function useChat(send = streamChat) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const controller = useRef(null);

  const update = useCallback((id, patch) => {
    setMessages((current) => current.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));
  }, []);

  const ask = useCallback(
    async (question) => {
      const assistantId = newId();
      setMessages((current) => [
        ...current,
        { id: newId(), role: 'user', text: question },
        { id: assistantId, role: 'assistant', text: '', citations: [], status: 'streaming' },
      ]);
      setIsStreaming(true);
      controller.current = new AbortController();

      const handlers = {
        sources: ({ sources }) => update(assistantId, () => ({ citations: sources })),
        delta: ({ text }) => update(assistantId, (m) => ({ text: m.text + text })),
        final: ({ answer, citations }) =>
          update(assistantId, () => ({ text: answer, citations, status: 'done' })),
        error: ({ message }) => update(assistantId, () => ({ error: message })),
      };

      try {
        await send(question, {
          signal: controller.current.signal,
          onEvent: ({ type, data }) => handlers[type]?.(data),
        });
      } catch (error) {
        if (error.name !== 'AbortError') {
          update(assistantId, () => ({ error: error.message, status: 'failed' }));
        }
      } finally {
        update(assistantId, (m) => ({ status: m.status === 'streaming' ? 'done' : m.status }));
        setIsStreaming(false);
      }
    },
    [send, update],
  );

  const stop = useCallback(() => controller.current?.abort(), []);

  return { messages, isStreaming, ask, stop };
}
