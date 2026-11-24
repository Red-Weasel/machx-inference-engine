export function parseCSV(text, { delimiter = ',' } = {}) {
  if (typeof text !== 'string') throw new TypeError('text must be a string');
  if (typeof delimiter !== 'string' || delimiter.length !== 1 || '"\r\n'.includes(delimiter)) throw new TypeError('bad delimiter');
  const rows = [];
  const n = text.length;
  if (n === 0) return rows;
  let row = [], field = '', atStart = true, i = 0;
  const endField = () => { row.push(field); field = ''; atStart = true; };
  const endRow = () => { endField(); rows.push(row); row = []; };
  while (i < n) {
    const c = text[i];
    if (atStart && c === '"') {
      i++;
      for (;;) {
        if (i >= n) throw new SyntaxError('unclosed quote');
        const ch = text[i];
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
          i++; break;
        }
        field += ch; i++;
      }
      atStart = false;
      if (i < n && text[i] !== delimiter && text[i] !== '\n' && text[i] !== '\r')
        throw new SyntaxError(`unexpected character after a closing quote at ${i}`);
      continue;
    }
    if (c === delimiter) { endField(); i++; continue; }
    if (c === '\r' && text[i + 1] === '\n') { endRow(); i += 2; continue; }
    if (c === '\n' || c === '\r') { endRow(); i++; continue; }
    field += c; atStart = false; i++;
  }
  const last = text[n - 1];
  if (!(last === '\n' || last === '\r') || !atStart || row.length > 0) endRow();
  return rows;
}
