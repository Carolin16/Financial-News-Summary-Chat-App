import { Fragment } from 'react';

const CITATION = /\[(\d+)\]/g;
const BULLET = /^\s*[-*]\s+/;

/** Renders the answer's light Markdown (paragraphs, bullets) with [n] as source links. */
export function AnswerText({ text, citations }) {
  const links = new Map(citations.map((c) => [c.number, c]));
  const blocks = groupLines(text.split('\n'));
  return blocks.map((block, index) =>
    block.type === 'list' ? (
      <ul key={index}>
        {block.lines.map((line, i) => (
          <li key={i}>{withCitations(line.replace(BULLET, ''), links)}</li>
        ))}
      </ul>
    ) : (
      <p key={index}>{withCitations(block.lines.join(' '), links)}</p>
    ),
  );
}

function groupLines(lines) {
  const blocks = [];
  for (const line of lines) {
    if (!line.trim()) {
      blocks.push(null);
      continue;
    }
    const type = BULLET.test(line) ? 'list' : 'paragraph';
    const last = blocks.at(-1);
    if (last && last.type === type) last.lines.push(line);
    else blocks.push({ type, lines: [line] });
  }
  return blocks.filter(Boolean);
}

function withCitations(line, links) {
  return line.split(CITATION).map((part, i) => {
    if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
    const citation = links.get(Number(part));
    if (!citation) return null;
    return (
      <a
        key={i}
        className="cite"
        href={citation.link}
        target="_blank"
        rel="noreferrer"
        title={citation.title}
      >
        [{part}]
      </a>
    );
  });
}
