// neural_network.js
// Minimal NN from scratch (no deps). Biases, Glorot init, activations, MSE.
// Stable: batch-averaged grads + L2 clipping + DESCENT update.

class Utils {
  // ---- RNG ----
  random(rows, cols, lo = -1, hi = 1) {
    const arr = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      for (let j = 0; j < cols; j++) row[j] = Math.random() * (hi - lo) + lo;
      arr[i] = row;
    }
    return arr;
  }
  glorot(rows, cols) {
    const limit = Math.sqrt(6 / (rows + cols));
    return this.random(rows, cols, -limit, limit);
  }

  // ---- activations ----
  actForward(matrix, name) {
    switch ((name || "sigmoid").toLowerCase()) {
      case "sigmoid":
      case "sig":
        return matrix.map((r) =>
          r.map((v) => {
            if (v >= 0) {
              const e = Math.exp(-v);
              return 1 / (1 + e);
            }
            const e = Math.exp(v);
            return e / (1 + e);
          })
        );
      case "tanh":
        return matrix.map((r) => r.map((v) => Math.tanh(v)));
      case "identity":
      case "lin":
        return matrix.map((r) => r.map((v) => v));
      default:
        throw new Error(`Unknown activation: ${name}`);
    }
  }
  actDerivativeFromActivation(activation, name) {
    switch ((name || "sigmoid").toLowerCase()) {
      case "sigmoid":
      case "sig":
        return activation.map((r) => r.map((s) => s * (1 - s)));
      case "tanh":
        return activation.map((r) => r.map((s) => 1 - s * s));
      case "identity":
      case "lin":
        return activation.map((r) => r.map(() => 1));
      default:
        throw new Error(`Unknown activation: ${name}`);
    }
  }

  // ---- array ops ----
  transpose(a) {
    const m = a.length,
      n = a[0].length;
    const out = new Array(n);
    for (let j = 0; j < n; j++) {
      const row = new Array(m);
      for (let i = 0; i < m; i++) row[i] = a[i][j];
      out[j] = row;
    }
    return out;
  }
  multiply(a, b) {
    const m = a.length,
      n = a[0].length,
      n2 = b.length,
      p = b[0].length;
    if (n !== n2)
      throw new Error(`multiply shape mismatch: ${m}x${n} @ ${n2}x${p}`);
    const out = Array.from({ length: m }, () => Array(p).fill(0));
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < n; k++) {
        const aik = a[i][k];
        for (let j = 0; j < p; j++) out[i][j] += aik * b[k][j];
      }
    }
    return out;
  }
  add(a, b) {
    const m = a.length,
      n = a[0].length;
    if (b.length !== m || b[0].length !== n) {
      throw new Error(
        `add shape mismatch: ${m}x${n} vs ${b.length}x${b[0].length}`
      );
    }
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[i][j] = a[i][j] + b[i][j];
    return out;
  }
  subtract(a, b) {
    const m = a.length,
      n = a[0].length;
    if (b.length !== m || b[0].length !== n) {
      throw new Error(
        `subtract shape mismatch: ${m}x${n} vs ${b.length}x${b[0].length}`
      );
    }
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[i][j] = a[i][j] - b[i][j];
    return out;
  }
  dotmultiply(a, b) {
    const m = a.length,
      n = a[0].length;
    if (b.length !== m || b[0].length !== n) {
      throw new Error(
        `dotmultiply shape mismatch: ${m}x${n} vs ${b.length}x${b[0].length}`
      );
    }
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[i][j] = a[i][j] * b[i][j];
    return out;
  }
  scale(a, scalar) {
    const m = a.length,
      n = a[0].length;
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[i][j] = a[i][j] * scalar;
    return out;
  }
  addRowwise(a, row) {
    const m = a.length,
      n = a[0].length;
    if (row.length !== 1 || row[0].length !== n) {
      throw new Error(
        `addRowwise shape mismatch: ${m}x${n} vs ${row.length}x${row[0].length}`
      );
    }
    return a.map((r) => r.map((v, j) => v + row[0][j]));
  }
  sumColumns(a) {
    const m = a.length,
      n = a[0].length;
    const row = new Array(n).fill(0);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) row[j] += a[i][j];
    return [row]; // 1 x n
  }
  abs(a) {
    return a.map((r) => r.map(Math.abs));
  }
  mean(a) {
    let s = 0,
      c = 0;
    for (const r of a)
      for (const v of r) {
        s += v;
        c++;
      }
    return s / c;
  }

  // ---- norms, clipping, finite checks ----
  l2norm(a) {
    let s = 0;
    for (const r of a) for (const v of r) s += v * v;
    return Math.sqrt(s);
  }
  clipByL2(a, maxNorm) {
    const n = this.l2norm(a);
    if (!isFinite(n) || n === 0) return a.map((r) => r.slice());
    if (n <= maxNorm) return a.map((r) => r.slice());
    const scale = maxNorm / n;
    return this.scale(a, scale);
  }
  isFiniteMatrix(a) {
    for (const r of a) for (const v of r) if (!Number.isFinite(v)) return false;
    return true;
  }
  assertFiniteMatrix(a, name) {
    if (!this.isFiniteMatrix(a))
      throw new Error(`Non-finite values detected in ${name}`);
  }

  // ---- data utils ----
  shuffleXY(X, Y) {
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [X[i], X[j]] = [X[j], X[i]];
      [Y[i], Y[j]] = [Y[j], Y[i]];
    }
  }
}

const utils = new Utils();

class Neural_Network {
  constructor(input_nodes, hidden_nodes, output_nodes, opts = {}) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.hidden_act = opts.hidden_act || "tanh";
    this.output_act = opts.output_act || "identity";
    this.epochs = opts.epochs || 60000;
    this.lr = opts.lr || 0.02; // averaged later by N
    this.shuffle = opts.shuffle ?? true;
    this.clip_norm = opts.clip_norm ?? 1.0;

    // Weights (Glorot) + zero biases
    this.W0 = utils.glorot(this.input_nodes, this.hidden_nodes);
    this.W1 = utils.glorot(this.hidden_nodes, this.output_nodes);
    this.b0 = utils.random(1, this.hidden_nodes, 0, 0);
    this.b1 = utils.random(1, this.output_nodes, 0, 0);

    this.output = null;
  }

  train(X, T) {
    if (X.length !== T.length)
      throw new Error(`X/T length mismatch: ${X.length} vs ${T.length}`);
    for (let i = 1; i < X.length; i++) {
      if (X[i].length !== X[0].length) throw new Error("ragged X rows");
      if (T[i].length !== T[0].length) throw new Error("ragged T rows");
    }

    const N = X.length;
    const lrScale = this.lr / N;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      if (this.shuffle) utils.shuffleXY(X, T);

      // ---- Forward
      const Z0 = utils.addRowwise(utils.multiply(X, this.W0), this.b0); // N x H
      const A0 = utils.actForward(Z0, this.hidden_act); // N x H
      const Z1 = utils.addRowwise(utils.multiply(A0, this.W1), this.b1); // N x O
      const A1 = utils.actForward(Z1, this.output_act); // N x O

      utils.assertFiniteMatrix(A0, "A0");
      utils.assertFiniteMatrix(A1, "A1");

      // ---- Loss (MSE)
      const error = utils.subtract(T, A1); // N x O
      const mse = utils.mean(error.map((r) => r.map((v) => v * v)));

      // ---- Backward
      // d(MSE)/dA1 = (A1 - T); we compute it as -error for clarity.
      const dA1 = utils.scale(error, -1);

      const dZ1 =
        this.output_act === "identity" || this.output_act === "lin"
          ? dA1
          : utils.dotmultiply(
              dA1,
              utils.actDerivativeFromActivation(A1, this.output_act)
            );

      // Unscaled grads
      let dW1 = utils.multiply(utils.transpose(A0), dZ1); // H x O
      let db1 = utils.sumColumns(dZ1); // 1 x O
      const dA0 = utils.multiply(dZ1, utils.transpose(this.W1)); // N x H
      const dZ0 = utils.dotmultiply(
        dA0,
        utils.actDerivativeFromActivation(A0, this.hidden_act)
      );
      let dW0 = utils.multiply(utils.transpose(X), dZ0); // I x H
      let db0 = utils.sumColumns(dZ0); // 1 x H

      // Clip grads by L2 norm (per-tensor)
      if (this.clip_norm && this.clip_norm > 0) {
        dW1 = utils.clipByL2(dW1, this.clip_norm);
        db1 = utils.clipByL2(db1, this.clip_norm);
        dW0 = utils.clipByL2(dW0, this.clip_norm);
        db0 = utils.clipByL2(db0, this.clip_norm);
      }

      // Scale by lr/N
      dW1 = utils.scale(dW1, lrScale);
      db1 = utils.scale(db1, lrScale);
      dW0 = utils.scale(dW0, lrScale);
      db0 = utils.scale(db0, lrScale);

      // ---- Gradient DESCENT (subtract)
      this.W1 = utils.subtract(this.W1, dW1);
      this.b1 = utils.subtract(this.b1, db1);
      this.W0 = utils.subtract(this.W0, dW0);
      this.b0 = utils.subtract(this.b0, db0);

      utils.assertFiniteMatrix(this.W0, "W0");
      utils.assertFiniteMatrix(this.W1, "W1");
      utils.assertFiniteMatrix(this.b0, "b0");
      utils.assertFiniteMatrix(this.b1, "b1");

      this.output = A1;

      if (epoch % 5000 === 0) {
        console.log(`Epoch ${epoch}: MSE=${mse.toFixed(6)}`);
      }
    }

    const finalErr = utils.subtract(T, this.output);
    const finalMSE = utils.mean(finalErr.map((r) => r.map((v) => v * v)));
    console.log(`Final MSE: ${finalMSE.toFixed(6)}`);
  }

  predict(X) {
    const Z0 = utils.addRowwise(utils.multiply(X, this.W0), this.b0);
    const A0 = utils.actForward(Z0, this.hidden_act);
    const Z1 = utils.addRowwise(utils.multiply(A0, this.W1), this.b1);
    const A1 = utils.actForward(Z1, this.output_act);
    return A1;
  }
}

/* ===========================
   DEMO: sine regression (default)
   =========================== */
(function runSineDemo() {
  const input = [];
  const target = [];
  const N = 201; // dense coverage of [0,1]
  for (let i = 0; i < N; i++) {
    const x = i / (N - 1);
    const y = Math.sin(2 * Math.PI * x);
    input.push([x]); // N x 1
    target.push([y]); // N x 1
  }

  const net = new Neural_Network(1, 16, 1, {
    hidden_act: "tanh",
    output_act: "identity",
    lr: 0.02, // averaged by N inside
    epochs: 120000,
    shuffle: true,
    clip_norm: 1.0,
  });

  console.log("\nTraining on sin(2πx)...");
  net.train(input, target);

  const testXs = [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1];
  console.log("\nSample predictions (x, true, pred):");
  for (const x of testXs) {
    const t = Math.sin(2 * Math.PI * x);
    const p = net.predict([[x]])[0][0];
    console.log(`${x.toFixed(3)}  true=${t.toFixed(4)}  pred=${p.toFixed(4)}`);
  }
})();
