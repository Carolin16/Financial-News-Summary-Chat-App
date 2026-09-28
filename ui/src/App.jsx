import { useEffect, useRef } from 'react';
import { Composer } from './components/Composer';
import { Message } from './components/Message';
import { useChat } from './hooks/useChat';

const EXAMPLES = [
  "What's the latest news on Intel?",
  "What's Nvidia's price target?",
  'Why did Intel stock jump?',
  "What's the news on Tesla?",
];

export default function App() {
  const { messages, isStreaming, ask, stop } = useChat();
  const bottom = useRef(null);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth' });
  }, [messages]);

  return (
    <main className="app">
      <header>
        <h1>Financial News Chat</h1>
        <p className="muted">
          Answers are summaries of the news dataset, with sources. Not investment advice.
        </p>
      </header>

      <section className="messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button key={example} type="button" onClick={() => ask(example)}>
                {example}
              </button>
            ))}
          </div>
        )}
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        <div ref={bottom} />
      </section>

      <Composer onSend={ask} onStop={stop} isStreaming={isStreaming} />
    </main>
  );
}
