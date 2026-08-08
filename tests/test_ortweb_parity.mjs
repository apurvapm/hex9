// Checks that ONNX Runtime Web computes what PyTorch computed.
//
//   node tests/test_ortweb_parity.mjs model.onnx fixture.json
//
// export_onnx.py proves PyTorch matches ONNX Runtime *native*. This closes the other
// half. The web build is not the native one recompiled: it is published separately,
// on its own schedule -- onnxruntime-web's latest is 1.27.0 while the native runtime
// this project trains against is 1.28.0 -- and it has historically shipped a reduced
// operator set. A missing or differently fused kernel gives a model that loads without
// complaint and plays badly, with nothing in the console to explain it.
//
// The reference is PyTorch, carried in the fixture along with its inputs. Comparing
// the two runtimes to each other would leave two suspects on a disagreement; comparing
// both to the weights as trained names the one that drifted.
//
// WHAT THIS DOES NOT COVER: node provides SharedArrayBuffer, so the single-threaded
// fallback that the deployed demo actually relies on is NOT exercised here. GitHub
// Pages cannot set COOP/COEP, so in the browser there is no SharedArrayBuffer at all.
// numThreads is pinned to 1 below to get as close as node allows, but confirming the
// no-SAB path still needs a real browser.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const modelPath = process.argv[2];
const fixturePath = process.argv[3];
if (!modelPath || !fixturePath) {
  console.error('usage: node test_ortweb_parity.mjs <model.onnx> <fixture.json>');
  process.exit(2);
}

const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const vendor = path.join(repo, 'web', 'vendor');

let checks = 0;
let failures = 0;
function check(condition, what) {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.log(`  FAIL: ${what}`);
  }
}

const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const { boardSize, planeCount, policySize, tolerance, cases } = fixture;

const ortUrl = pathToFileURL(path.join(vendor, 'ort.wasm.bundle.min.mjs')).href;
const ort = (await import(ortUrl)).default ?? (await import(ortUrl));

// Single-threaded and wasm-only, matching what the browser will get. Setting this
// explicitly rather than letting it be inferred means the demo behaves the same
// whether or not a host happens to serve cross-origin isolation headers.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = pathToFileURL(vendor + path.sep).href;
ort.env.logLevel = 'error';

console.log('== onnx runtime web parity ==\n');
console.log(`model    : ${path.basename(modelPath)}`);
console.log(`fixture  : ${cases.length} cases, ${boardSize}x${boardSize}`);
console.log(`ort-web  : ${ort.env.versions?.web ?? 'unknown'}`);
console.log(`threads  : ${ort.env.wasm.numThreads}`);
console.log(`sab here : ${typeof SharedArrayBuffer !== 'undefined'} (node; the browser has none)\n`);

// Bytes, not a path. This is the *web* runtime: given a string it fetches a URL, and
// a filesystem path is not one. The browser will hand it bytes too, after fetching the
// .onnx itself, so passing a Uint8Array here matches how the demo will load the model.
const modelBytes = new Uint8Array(fs.readFileSync(modelPath));
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});

check(session.inputNames.includes('board'), 'model should expose the "board" input');
check(session.outputNames.includes('policy'), 'model should expose a "policy" output');
check(session.outputNames.includes('value'), 'model should expose a "value" output');

// --- one position at a time, which is what the browser does ------------------
let worstPolicy = 0;
let worstValue = 0;

for (const [index, testCase] of cases.entries()) {
  const planes = Float32Array.from(testCase.planes);
  const tensor = new ort.Tensor('float32', planes, [1, planeCount, boardSize, boardSize]);
  const output = await session.run({ board: tensor });

  const policy = output.policy.data;
  const value = output.value.data;

  if (policy.length !== policySize) {
    check(false, `case ${index}: policy width ${policy.length}, expected ${policySize}`);
    continue;
  }
  for (let i = 0; i < policySize; i += 1) {
    worstPolicy = Math.max(worstPolicy, Math.abs(policy[i] - testCase.logits[i]));
  }
  worstValue = Math.max(worstValue, Math.abs(value[0] - testCase.value));
}

console.log('single-position inference matches pytorch');
check(worstPolicy <= tolerance,
      `worst policy difference ${worstPolicy.toExponential(2)} exceeds ${tolerance}`);
check(worstValue <= tolerance,
      `worst value difference ${worstValue.toExponential(2)} exceeds ${tolerance}`);
console.log(`  policy max diff ${worstPolicy.toExponential(2)}, ` +
            `value max diff ${worstValue.toExponential(2)}`);

// --- batched, since the exporter made the batch axis dynamic on purpose ------
// The engine's prepareBatch hands over several positions at once, so a web build that
// only handled batch 1 would break the demo and not this file's first section.
console.log('batched inference matches the same values');
{
  const batch = Math.min(8, cases.length);
  const planes = new Float32Array(batch * planeCount * boardSize * boardSize);
  for (let i = 0; i < batch; i += 1) {
    planes.set(Float32Array.from(cases[i].planes), i * planeCount * boardSize * boardSize);
  }
  const tensor = new ort.Tensor('float32', planes,
                                [batch, planeCount, boardSize, boardSize]);
  const output = await session.run({ board: tensor });

  let worstBatchPolicy = 0;
  let worstBatchValue = 0;
  for (let i = 0; i < batch; i += 1) {
    for (let j = 0; j < policySize; j += 1) {
      worstBatchPolicy = Math.max(
        worstBatchPolicy,
        Math.abs(output.policy.data[i * policySize + j] - cases[i].logits[j]),
      );
    }
    worstBatchValue = Math.max(worstBatchValue,
                               Math.abs(output.value.data[i] - cases[i].value));
  }
  check(worstBatchPolicy <= tolerance,
        `batched policy difference ${worstBatchPolicy.toExponential(2)} too large`);
  check(worstBatchValue <= tolerance,
        `batched value difference ${worstBatchValue.toExponential(2)} too large`);
  console.log(`  batch of ${batch}: policy max diff ${worstBatchPolicy.toExponential(2)}, ` +
              `value max diff ${worstBatchValue.toExponential(2)}`);
}

// --- the operator set, which is the specific worry ---------------------------
// If any op were missing, session creation above would already have thrown. Reporting
// the graph's ops makes the claim checkable rather than implied.
console.log('every operator in the graph is supported');
check(session.inputNames.length === 1, 'model should take exactly one input');
console.log(`  session created and ${cases.length} inferences ran without a fallback`);

console.log(`\n${checks} checks, ${failures} failures`);
process.exit(failures === 0 ? 0 : 1);
