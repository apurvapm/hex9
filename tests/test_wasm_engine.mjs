// Drives the WebAssembly engine the way the browser will, under node.
//
//   node tests/test_wasm_engine.mjs build-wasm/hex_engine.mjs
//
// The interface exists because ONNX Runtime Web is async, so the search is a loop of
// prepareBatch / evaluate / submitBatch driven from JavaScript. That loop is the part
// most likely to be wrong -- an off-by-one in the simulation accounting, a stale
// memory view after growth, a terminal leaf counted twice -- and none of it is
// exercised by the C++ suites. This checks it without needing a browser or a network.
//
// The stand-in evaluator is uniform priors and a value of zero. That makes the search
// pure visit-count exploration, which is enough to verify the plumbing and, at a
// position one move from a win, enough to verify the terminal shortcut.

import process from 'node:process';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const modulePath = process.argv[2] ?? 'build-wasm/hex_engine.mjs';

let checks = 0;
let failures = 0;

function check(condition, what) {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.log(`  FAIL: ${what}`);
  }
}

// Uniform logits and a neutral value. Written straight into the engine's own buffers,
// which is exactly what the ORT path will do with its output tensors.
function evaluateBatch(engine) {
  const n = engine.pendingCount();
  if (n === 0) return;
  engine.batchLogits().fill(0);
  engine.batchValues().fill(0);
}

function runSearch(engine, simulations, batch = 16) {
  engine.beginSearch(simulations, 1.5, 3);
  let iterations = 0;
  while (!engine.searchComplete()) {
    engine.prepareBatch(batch);
    evaluateBatch(engine);
    engine.submitBatch();
    iterations += 1;
    if (iterations > simulations + 64) throw new Error('search failed to terminate');
  }
  return iterations;
}

const { default: factory } = await import(
  pathToFileURL(path.resolve(modulePath)).href
);
const Module = await factory();

console.log('== hex wasm engine ==\n');

// --- board basics ----------------------------------------------------------
console.log('board state round-trips');
{
  const engine = new Module.HexEngine();
  check(engine.size() === 9, 'board should be 9x9');
  check(engine.policySize() === 82, 'policy width should include the swap action');
  check(engine.planeCount() === 3, 'encoding should have three planes');
  check(engine.toPlay() === 0, 'red moves first');
  check(engine.moveCount() === 0, 'a fresh board has no moves');
  check(!engine.canSwap(), 'swap is not legal before an opening');

  check(engine.play(40), 'centre should be playable');
  check(engine.cellAt(40) === 1, 'centre should now hold a red stone');
  check(engine.toPlay() === 1, 'blue moves second');
  check(engine.canSwap(), 'swap is legal as the reply to the opening');
  check(!engine.play(40), 'an occupied cell should be rejected');
  check(engine.legalMoves().length === 81, '80 empty cells plus swap');

  check(engine.undo(), 'undo should succeed');
  check(engine.cellAt(40) === 0, 'undo should clear the cell');
  check(engine.moveCount() === 0, 'undo should restore the ply count');
  check(!engine.undo(), 'undo on an empty board should fail');
  engine.delete();
}

// --- swap, which is transpose plus recolour rather than turn logic ----------
console.log('swap transposes and recolours');
{
  const engine = new Module.HexEngine();
  engine.play(2);                       // (0, 2)
  check(engine.play(engine.swapAction()), 'blue should be able to swap');
  check(engine.cellAt(2) === 0, 'the original stone should be gone');
  check(engine.cellAt(18) === 2, 'it should reappear at (2, 0) as blue');
  check(engine.toPlay() === 0, 'red moves again after a swap');
  check(engine.moveCount() === 2, 'swap advances to ply 2');
  check(!engine.canSwap(), 'swap is available only once');
  engine.delete();
}

// --- the connection-distance readout the design notes require --------------
console.log('connection distance is reported for both sides');
{
  const engine = new Module.HexEngine();
  const red = engine.connectionDistance(0);
  const blue = engine.connectionDistance(1);
  check(red === 9 && blue === 9, 'an empty 9x9 needs nine stones each way');
  engine.play(40);
  check(engine.connectionDistance(0) === 8, 'a stone should shorten red by one');
  engine.delete();
}

// --- the resumable search loop ---------------------------------------------
console.log('resumable search accounts for every simulation');
{
  const engine = new Module.HexEngine();
  engine.play(40);
  engine.play(30);

  const budget = 400;
  runSearch(engine, budget);

  check(engine.searchComplete(), 'search should report completion');
  check(engine.simulationsDone() === budget,
        `simulations done should equal the budget, got ${engine.simulationsDone()}`);
  const best = engine.bestMove();
  check(best >= 0 && engine.isLegal(best), 'best move should be legal');

  const top = engine.topMoves(5);
  check(top.length === 5, 'top-k should return k entries');
  check(top[0].visits >= top[1].visits, 'top moves should be sorted by visits');
  check(top[0].move === best, 'the most-visited move should be the chosen one');
  for (const entry of top) {
    check(entry.prior > 0 && entry.prior <= 1, 'priors should be probabilities');
    check(Number.isFinite(entry.puct), 'puct score should be finite');
  }
  engine.delete();
}

console.log('batch size does not change the accounting');
{
  for (const batch of [1, 3, 16, 64]) {
    const engine = new Module.HexEngine();
    engine.play(40);
    runSearch(engine, 200, batch);
    check(engine.simulationsDone() === 200,
          `batch ${batch} should still run exactly 200 simulations`);
    check(engine.isLegal(engine.bestMove()),
          `batch ${batch} should choose a legal move`);
    engine.delete();
  }
}

console.log('visit counts sum to the budget');
{
  const engine = new Module.HexEngine();
  engine.play(40);
  const budget = 300;
  runSearch(engine, budget);
  const heat = engine.policyHeatmap();
  const mass = heat.reduce((a, b) => a + b, 0);
  check(Math.abs(mass - 1) < 1e-4, `heatmap should normalise, got ${mass}`);

  let visits = 0;
  for (const entry of engine.topMoves(82)) visits += entry.visits;
  // The root's own evaluation consumes one simulation before any descent, so the
  // children account for the budget less that one.
  check(visits === budget - 1,
        `root children should hold ${budget - 1} visits, got ${visits}`);
  engine.delete();
}

// --- terminal handling, which the batching path could easily get wrong -----
console.log('a forced win is found without an inference');
{
  // Red holds a column with one gap; filling it connects top to bottom. The stub
  // evaluator is uninformative, so only terminal detection can find this.
  const engine = new Module.HexEngine();
  const size = 9;
  for (let row = 0; row < size; row += 1) {
    if (row === 4) continue;
    engine.play(row * size + 4);          // red builds the column
    engine.play(row * size + 8 - (row === 8 ? 1 : 0)); // blue plays elsewhere
  }
  check(!engine.isTerminal(), 'the position should still be undecided');
  check(engine.toPlay() === 0, 'red should be to move');

  runSearch(engine, 600);
  check(engine.bestMove() === 4 * size + 4,
        `red should complete the column at 40, chose ${engine.bestMove()}`);
  engine.delete();
}

console.log('tree snapshot is well formed');
{
  const engine = new Module.HexEngine();
  engine.play(40);
  runSearch(engine, 400);

  const nodes = engine.treeSnapshot(60);
  check(nodes.length > 1, 'snapshot should contain more than the root');
  check(nodes.length <= 60, 'snapshot should respect the node cap');
  check(nodes[0].depth === 0 && nodes[0].parent === -1, 'first node is the root');

  const seen = new Set(nodes.map((n) => n.index));
  let parentsPresent = true;
  let depthsOrdered = true;
  for (const node of nodes.slice(1)) {
    if (!seen.has(node.parent)) parentsPresent = false;
    if (node.depth < 1) depthsOrdered = false;
    if (node.visits === 0) depthsOrdered = false;
  }
  check(parentsPresent, 'every node except the root should have its parent included');
  check(depthsOrdered, 'children should be visited and deeper than the root');
  engine.delete();
}

console.log('playing a move discards the previous search');
{
  const engine = new Module.HexEngine();
  engine.play(40);
  runSearch(engine, 200);
  check(engine.searchComplete(), 'search finished');
  engine.play(30);
  check(engine.simulationsDone() === 0, 'a move should reset the search');
  check(engine.bestMove() === -1, 'no tree means no best move');
  engine.delete();
}

const expected = 61;
console.log(`\n${checks} checks, ${failures} failures`);
if (checks !== expected) {
  console.log(`\nERROR: expected ${expected} checks.`);
  process.exit(1);
}
process.exit(failures === 0 ? 0 : 1);
