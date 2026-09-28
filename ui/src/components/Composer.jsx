import { useState } from 'react';

export function Composer({ onSend, onStop, isStreaming }) {
  const [text, setText] = useState('');

  const submit = (event) => {
    event.preventDefault();
    const question = text.trim();
    if (!question || isStreaming) return;
    onSend(question);
    setText('');
  };

  return (
    <form className="composer" onSubmit={submit}>
      <input
        aria-label="Question"
        placeholder="Ask about recent financial news…"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      {isStreaming ? (
        <button type="button" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" disabled={!text.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
