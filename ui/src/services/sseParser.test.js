import { describe, expect, it } from 'vitest';
import { createSseParser } from './sseParser';

function collect(chunks) {
  const events = [];
  const push = createSseParser((event) => events.push(event));
  chunks.forEach(push);
  return events;
}

describe('createSseParser', () => {
  it('parses named events with JSON data', () => {
    const events = collect(['event: delta\ndata: {"text":"Hi"}\n\n']);
    expect(events).toEqual([{ type: 'delta', data: { text: 'Hi' } }]);
  });

  it('handles events split across chunks and CRLF line endings', () => {
    const events = collect([
      'event: fin',
      'al\r\ndata: {"answer":',
      '"ok"}\r\n\r\nevent: delta\r\n',
    ]);
    expect(events).toEqual([{ type: 'final', data: { answer: 'ok' } }]);
  });

  it('ignores comments and keep-alive pings', () => {
    const events = collect([': ping\n\n', 'event: meta\ndata: {"intent":"news"}\n\n']);
    expect(events).toEqual([{ type: 'meta', data: { intent: 'news' } }]);
  });

  it('parses several events from one chunk', () => {
    const events = collect([
      'event: delta\ndata: {"text":"a"}\n\nevent: delta\ndata: {"text":"b"}\n\n',
    ]);
    expect(events.map((e) => e.data.text)).toEqual(['a', 'b']);
  });
});
