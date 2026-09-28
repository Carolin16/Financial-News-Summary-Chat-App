/**
 * Incremental parser for a text/event-stream body.
 *
 * EventSource only supports GET, and the chat endpoint is a POST, so the stream is read
 * with fetch and parsed here. Chunks may split events anywhere; incomplete data is kept
 * until the rest arrives.
 */
export function createSseParser(onEvent) {
  let buffer = '';

  return function push(chunk) {
    buffer += chunk.replace(/\r\n/g, '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseBlock(block);
      if (event) onEvent(event);
      boundary = buffer.indexOf('\n\n');
    }
  };
}

function parseBlock(block) {
  let name = 'message';
  const dataLines = [];
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue; // comment / keep-alive ping
    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? '' : line.slice(separator + 1).replace(/^ /, '');
    if (field === 'event') name = value;
    if (field === 'data') dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { type: name, data: JSON.parse(dataLines.join('\n')) };
}
