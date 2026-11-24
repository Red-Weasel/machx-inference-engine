import assert from 'node:assert/strict';
import { compareSemver as c } from './solution.mjs';
const lt = (a, b) => { assert.equal(c(a, b), -1, `${a} < ${b}`); assert.equal(c(b, a), 1, `${b} > ${a}`); };
lt('1.9.0', '1.10.0'); lt('1.0.0', '2.0.0'); lt('1.2.3', '1.2.4'); lt('0.9.9', '1.0.0');
lt('1.0.0-alpha', '1.0.0'); lt('1.0.0-alpha', '1.0.0-alpha.1'); lt('1.0.0-alpha.2', '1.0.0-alpha.10');
lt('1.0.0-alpha.1', '1.0.0-alpha.beta'); lt('1.0.0-beta', '1.0.0-beta.2'); lt('1.0.0-beta.11', '1.0.0-rc.1');
lt('1.0.0-1', '1.0.0-alpha'); lt('1.0.0-rc.1', '1.0.0');
assert.equal(c('1.0.0+build.5', '1.0.0+build.9'), 0); assert.equal(c('1.2.3', '1.2.3'), 0);
assert.equal(c('1.0.0-rc.1+x', '1.0.0-rc.1+y'), 0);
assert.equal(c('1.0.0-0a', '1.0.0-0b'), -1);
for (const bad of ['1.2', '01.2.3', 'v1.2.3', '1.2.3-', '1.2.3.4', '', 'a.b.c', '1.0.0-01', '1.0.0-alpha.00'])
  assert.throws(() => c(bad, '1.0.0'), Error, bad);
console.log('OK');
