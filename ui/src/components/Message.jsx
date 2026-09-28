import { AnswerText } from './AnswerText';

export function Message({ message }) {
  if (message.role === 'user') {
    return <div className="message user">{message.text}</div>;
  }

  const isDraft = message.status === 'streaming';
  return (
    <div className="message assistant" aria-busy={isDraft}>
      {message.text ? (
        <AnswerText text={message.text} citations={message.citations} />
      ) : (
        isDraft && <p className="muted">Searching the news…</p>
      )}
      {message.error && <p className="error">⚠ {message.error}</p>}
      {!isDraft && message.citations.length > 0 && (
        <details className="sources">
          <summary>Sources ({message.citations.length})</summary>
          <ol>
            {message.citations.map((c) => (
              <li key={c.number} value={c.number}>
                <a href={c.link} target="_blank" rel="noreferrer">
                  {c.title}
                </a>
                {c.is_partial && <span className="muted"> (partial article)</span>}
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
