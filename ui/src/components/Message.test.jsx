import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Message } from './Message';

const citations = [
  { number: 1, title: 'DBS cuts Nvidia target', link: 'https://x/dbs', is_partial: true },
  { number: 2, title: 'Nvidia earnings preview', link: 'https://x/earn', is_partial: false },
];

describe('Message', () => {
  it('renders bullets with citation markers linked to their sources', () => {
    const text =
      'Two views on Nvidia [2]:\n- DBS cut its target to $160 [1].\n- Earnings loom [2][9].';
    render(<Message message={{ role: 'assistant', text, citations, status: 'done' }} />);

    expect(screen.getAllByRole('listitem').length).toBeGreaterThanOrEqual(2);
    const links = screen.getAllByRole('link', { name: '[1]' });
    expect(links[0]).toHaveAttribute('href', 'https://x/dbs');
    // A marker without a matching source is not rendered as a link.
    expect(screen.queryByRole('link', { name: '[9]' })).toBeNull();
  });

  it('labels partial articles in the source list', () => {
    render(<Message message={{ role: 'assistant', text: 'x [1]', citations, status: 'done' }} />);
    expect(screen.getByText('(partial article)', { exact: false })).toBeInTheDocument();
  });

  it('shows a searching placeholder while streaming with no text yet', () => {
    render(
      <Message message={{ role: 'assistant', text: '', citations: [], status: 'streaming' }} />,
    );
    expect(screen.getByText('Searching the news…')).toBeInTheDocument();
  });
});
