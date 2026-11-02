// Two-view (epipolar) geometry of two pinhole cameras: pure maths, no DOM, so it runs under node too. A classic script
// like camera.js: in the browser it defines the global Epi, under node it is require('./epipolar.js').
//
// Each camera maps the world by X_c = R X_w + t (t = −R C) and pixels by x = K X_c / Z_c (camera.js's conventions).
//   relative pose        X_2 = R X_1 + t,   R = R_2 R_1ᵀ,   t = t_2 − R t_1
//   essential matrix     E = [t]× R           x̂_2ᵀ E x̂_1 = 0 for normalised coordinates x̂ = K⁻¹ x
//   fundamental matrix   F = K_2⁻ᵀ E K_1⁻¹    x_2ᵀ F x_1 = 0 for pixels (homogeneous)
//   epipolar lines       l_2 = F x_1 in image 2, l_1 = Fᵀ x_2 in image 1   (a u + b v + c = 0)
//   epipoles             e_1 = the image of C_2 in camera 1 (F e_1 = 0), e_2 = the image of C_1 in camera 2 (Fᵀ e_2 = 0)

(function (root) {
  'use strict';
  const mul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
  const mulv = (A, v) => A.map(row => row.reduce((s, a, k) => s + a * v[k], 0));
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  // [a]×: the matrix with [a]× b = a × b
  const skew = ([x, y, z]) => [[0, -z, y], [z, 0, -x], [-y, x, 0]];

  // the inverse of an intrinsic matrix [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
  function invK(K) {
    const [[fx, s, cx], [, fy, cy]] = K;
    return [[1 / fx, -s / (fx * fy), (s * cy - cx * fy) / (fx * fy)], [0, 1 / fy, -cy / fy], [0, 0, 1]];
  }

  // camera 2 relative to camera 1: X_2 = R X_1 + t
  function relativePose(cam1, cam2) {
    const R = mul(cam2.R, transpose(cam1.R));
    return { R, t: sub(cam2.t, mulv(R, cam1.t)) };
  }
  const essential = (R, t) => mul(skew(t), R);
  function fundamental(cam1, cam2) {
    const { R, t } = relativePose(cam1, cam2), E = essential(R, t);
    return { R, t, E, F: mul(mul(transpose(invK(cam2.K)), E), invK(cam1.K)) };
  }

  // the epipole of camera a: the image of the other camera's centre C_b, homogeneous [u, v, w] (w = 0: at infinity,
  // the direction (u, v) in the picture)
  const centreOf = cam => mulv(transpose(cam.R), cam.t.map(x => -x));
  const epipole = (camA, camB) => mulv(camA.K, mulv(camA.R, sub(centreOf(camB), centreOf(camA))));

  const lineInImage2 = (F, [u, v]) => mulv(F, [u, v, 1]);
  const lineInImage1 = (F, [u, v]) => mulv(transpose(F), [u, v, 1]);
  // distance in pixels of a point from a line a u + b v + c = 0
  const pointLineDistance = ([a, b, c], [u, v]) => (a * u + b * v + c) / Math.hypot(a, b);
  const residual = (F, x1, x2) => dot([x2[0], x2[1], 1], mulv(F, [x1[0], x1[1], 1]));

  // the part of the line a u + b v + c = 0 inside the rectangle [x0, x1] × [y0, y1], as two points, or null
  function clipLine([a, b, c], x0, y0, x1, y1) {
    const pts = [];
    if (Math.abs(b) > 1e-12) for (const u of [x0, x1]) { const v = -(a * u + c) / b; if (v >= y0 - 1e-9 && v <= y1 + 1e-9) pts.push([u, v]); }
    if (Math.abs(a) > 1e-12) for (const v of [y0, y1]) { const u = -(b * v + c) / a; if (u >= x0 - 1e-9 && u <= x1 + 1e-9) pts.push([u, v]); }
    let best = null, bd = -1;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (d > bd) { bd = d; best = [pts[i], pts[j]]; }
    }
    return bd > 1e-9 ? best : null;
  }

  // a pixel's ray: the camera centre and a unit direction in the world
  function pixelRay(cam, [u, v]) {
    const d = mulv(transpose(cam.R), mulv(invK(cam.K), [u, v, 1])), n = Math.hypot(...d);
    return { C: centreOf(cam), dir: d.map(x => x / n) };
  }

  // the nearest points of two rays C + d · dir (unit directions): d1 and d2 along them, the points P1 and P2 and the gap
  // between them (0 where the rays meet); null when they are parallel
  function closestPoints(ray1, ray2) {
    const w = sub(ray1.C, ray2.C), c = dot(ray1.dir, ray2.dir), den = 1 - c * c;
    if (den < 1e-12) return null;
    const d1 = (c * dot(ray2.dir, w) - dot(ray1.dir, w)) / den, d2 = dot(ray2.dir, w) + d1 * c;
    const P1 = ray1.C.map((x, i) => x + d1 * ray1.dir[i]), P2 = ray2.C.map((x, i) => x + d2 * ray2.dir[i]);
    return { d1, d2, P1, P2, gap: Math.hypot(...sub(P1, P2)) };
  }

  // the singular value decomposition of a 3 × 3 matrix, A = U diag(S) Vᵀ, S descending, U and V rotations (det +1; the
  // sign goes to S[2] when det A < 0). V from the eigenvectors of AᵀA (cyclic Jacobi), U from A V, its last column
  // completed by a cross product when A is singular.
  function svd3(A) {
    const M = mul(transpose(A), A).map(r => r.slice());
    let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let sweep = 0; sweep < 50; sweep++) {
      const off = M[0][1] ** 2 + M[0][2] ** 2 + M[1][2] ** 2;
      if (off < 1e-30 * (M[0][0] ** 2 + M[1][1] ** 2 + M[2][2] ** 2 + 1e-300)) break;
      for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
        if (Math.abs(M[p][q]) < 1e-300) continue;
        const th = (M[q][q] - M[p][p]) / (2 * M[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 3; k++) {   // M ← Jᵀ M J, V ← V J
          const a = M[k][p], b = M[k][q]; M[k][p] = c * a - s * b; M[k][q] = s * a + c * b;
        }
        for (let k = 0; k < 3; k++) {
          const a = M[p][k], b = M[q][k]; M[p][k] = c * a - s * b; M[q][k] = s * a + c * b;
        }
        for (let k = 0; k < 3; k++) {
          const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b;
        }
      }
    }
    const order = [0, 1, 2].sort((i, j) => M[j][j] - M[i][i]);
    let Vc = order.map(i => V.map(r => r[i]));   // columns of V, by descending eigenvalue
    if (dot(cross(Vc[0], Vc[1]), Vc[2]) < 0) Vc[2] = Vc[2].map(x => -x);
    const S = order.map(i => Math.sqrt(Math.max(0, M[i][i])));
    const Uc = Vc.map((v, i) => { const u = mulv(A, v), n = Math.hypot(...u); return n > 1e-12 * (S[0] || 1) ? u.map(x => x / n) : null; });
    Uc[0] = Uc[0] || [1, 0, 0];
    if (!Uc[1]) { const a = Math.abs(Uc[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], c = cross(Uc[0], a), n = Math.hypot(...c); Uc[1] = c.map(x => x / n); }
    const u3 = cross(Uc[0], Uc[1]);
    if (Uc[2] && dot(Uc[2], u3) < 0) S[2] = -S[2];   // det A < 0: the sign on the smallest singular value
    Uc[2] = u3;
    return { U: transpose(Uc), S, V: transpose(Vc) };
  }

  // the four (R, t) an essential matrix allows (t of unit length: the scale is lost): R = U W Vᵀ or U Wᵀ Vᵀ, t = ±u₃
  function decomposeEssential(E) {
    const { U, V } = svd3(E), W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]], Vt = transpose(V);
    const Ra = mul(mul(U, W), Vt), Rb = mul(mul(U, transpose(W)), Vt), u3 = U.map(r => r[2]);
    return [[Ra, u3], [Ra, u3.map(x => -x)], [Rb, u3], [Rb, u3.map(x => -x)]].map(([R, t]) => ({ R, t }));
  }

  // the depths (Z in each camera) of the point seen at the normalised coordinates x̂₁, x̂₂ ([x, y], z = 1), triangulated
  // with camera 2 at X₂ = R X₁ + t: both positive for the one (R, t) of the four that puts it in front of both cameras
  function depthsOf(R, t, xh1, xh2) {
    const C2 = mulv(transpose(R), t.map(x => -x)), d2 = mulv(transpose(R), [xh2[0], xh2[1], 1]);
    const cp = closestPoints({ C: [0, 0, 0], dir: [xh1[0], xh1[1], 1].map(x => x / Math.hypot(xh1[0], xh1[1], 1)) },
      { C: C2, dir: d2.map(x => x / Math.hypot(...d2)) });
    if (!cp) return null;
    const P = cp.P1;
    return { z1: P[2], z2: mulv(R, P)[2] + t[2] };
  }

  const Epi = { skew, invK, relativePose, essential, fundamental, centreOf, epipole, lineInImage1, lineInImage2,
    pointLineDistance, residual, clipLine, pixelRay, closestPoints, svd3, decomposeEssential, depthsOf, cross, dot };
  root.Epi = Epi;
  if (typeof module !== 'undefined' && module.exports) module.exports = Epi;
})(typeof globalThis !== 'undefined' ? globalThis : this);
