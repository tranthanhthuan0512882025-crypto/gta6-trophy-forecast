// GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// Method: a stale-aware earners-over-players forecast.
//
// What the program knows is only the case file: every read the house made of
// Exophase's GTA VI trophy page before the checkpoint, oldest first. Each
// read lists the players Exophase tracks and every trophy's earn rate. A
// rate is a fraction, earners over tracked players, and the forecast
// treats the two sides separately instead of extrapolating the bare rate:
//
//   - Earners: for each trophy, the count of tracked players who hold it
//     grows as players reach the content it needs, then bends toward how
//     many will ever hold it. The program fits a saturating curve,
//     E(u) = C + B * exp(-k * u), to the earner counts against hours since
//     release, by a grid over k with weighted least squares for C and B.
//   - Tracked players: the base grows as the launch's buyers arrive and
//     sync, then bends toward its own ceiling. The same saturating curve
//     is fit to the players-tracked series.
//   - The median rate at the target is the fitted earners over the fitted
//     base. A straight line in log-time, the house curve's shape, has no
//     ceiling and overshoots every trophy that has already bent; fitting
//     the two sides separately also keeps a late dilution -- the base
//     still growing while earners have flattened -- from looking like a
//     falling trophy, and vice versa. Both fits are recency-weighted
//     (12-hour half-life).
//   - Stale reads: Exophase refreshes on its own schedule, so a run of
//     identical reads means the page froze, not that nothing happened. When
//     the tail of the history has not changed for an hour or more, the
//     program forecasts from the last read that did change: earners are
//     carried forward on their own pre-freeze trend (flat when there is
//     none), while the tracked base is grown at a launch-night doubling
//     prior of 2 hours, so the median rate is diluted by the players who
//     arrived while the page was frozen. One exception: a trophy every
//     tracked player already holds (rate at or above 99.5%) behaves as
//     universal -- new arrivals earn it too, so its rate is not diluted.
//     The spread then also carries an extra term for how uncertain that
//     doubling is.
//   - Spread: the quantiles are the median's logit plus the normal
//     quantiles times a spread built from three measured pieces, never from
//     a fixed multiplier: the residual scatter of the fitted rate in logit
//     space (shrunk toward the pooled scatter across trophies), a
//     random-walk floor that grows with the square root of the horizon,
//     and the staleness term above. A forecast that is too narrow pays for
//     every trophy outside it, one too wide pays at every trophy, so the
//     spread is sized from the data, not picked.
//
// The program is deterministic: no clock reads, no randomness, no network.
// It reads every *.in file in the input directory in name order and writes
// one *.out file per case. Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk floor, logit units per square-root day of horizon. */
const WALK_SIGMA = 0.5;
/** Recency half-life of the fits, in hours. */
const HALF_LIFE_H = 12;
/** How far back the fits look, in hours. */
const WINDOW_H = 48;
/** A tail this long without any change counts as a frozen page, in hours. */
const STALE_AFTER_H = 1;
/** Launch-night prior for how fast the tracked base doubles, in hours. */
const PLAYER_DOUBLING_H = 2;
/** Extra spread for a frozen page, logit units at 4 stale hours. */
const STALE_SPREAD = 0.45;
/** A rate at or above this is universal: new arrivals hold it too. */
const UNIVERSAL_RATE = 99.5;
/** Prior weight (in reads) when shrinking a trophy's residual scatter. */
const SHRINK_WEIGHT = 6;
/** Decay rates tried for the saturating curves, per hour. */
const DECAYS = [
  0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2,
  0.3, 0.5, 0.8, 1.2,
];

interface Trophy {
  id: number;
}

interface Read {
  slot: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

interface Point {
  x: number;
  u: number;
  rate: number | null;
  tH: number;
  players: number;
  earners: number | null;
  slotTime: string;
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hoursBetween = (a: string, b: string): number =>
  (Date.parse(b) - Date.parse(a)) / 3_600_000;

/** Weighted least squares of y on x; returns [intercept, slope]. */
function weightedLine(
  xs: number[],
  ys: number[],
  ws: number[],
): [number, number] {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-9) return [my, 0];
  const slope = sxy / sxx;
  return [my - slope * mx, slope];
}

interface SaturatingFit {
  /** Fitted value at hours-since-release u. */
  predict: (u: number) => number;
}

/**
 * Fit y(u) = C + B * exp(-k * u) to weighted points (u, y), by a grid over
 * k with weighted least squares for C and B, keeping the k with the
 * smallest weighted squared error. With fewer than 3 points the curve is
 * the last value, flat. The value is clamped at zero: counts never go
 * negative.
 */
function fitSaturating(
  us: number[],
  ys: number[],
  ws: number[],
): SaturatingFit {
  const lastY = ys.length > 0 ? ys[ys.length - 1] : 0;
  if (us.length < 3) return { predict: () => Math.max(0, lastY) };
  let best: { c: number; b: number; k: number; sse: number } | null = null;
  for (const k of DECAYS) {
    const phi = us.map((u) => Math.exp(-k * u));
    let sw = 0;
    let sp = 0;
    let sr = 0;
    let spp = 0;
    let spr = 0;
    for (let i = 0; i < us.length; i++) {
      sw += ws[i];
      sp += ws[i] * phi[i];
      sr += ws[i] * ys[i];
      spp += ws[i] * phi[i] * phi[i];
      spr += ws[i] * phi[i] * ys[i];
    }
    const det = sw * spp - sp * sp;
    if (Math.abs(det) < 1e-12) continue;
    const c = (sr * spp - sp * spr) / det;
    const b = (sw * spr - sp * sr) / det;
    let sse = 0;
    for (let i = 0; i < us.length; i++) {
      const r = ys[i] - (c + b * phi[i]);
      sse += ws[i] * r * r;
    }
    if (best === null || sse < best.sse) best = { c, b, k, sse };
  }
  if (best === null) return { predict: () => Math.max(0, lastY) };
  const { c, b, k } = best;
  return { predict: (u: number) => Math.max(0, c + b * Math.exp(-k * u)) };
}

/**
 * Forecast one trophy's five quantiles. `series` is the trophy's reads in
 * order; `playersFit` is the fitted tracked-base curve; `scatter` is the
 * residual scatter already shrunk toward the pooled value; `stale`
 * describes the frozen tail (null when the reads are fresh).
 */
function forecastTrophy(
  series: Point[],
  playersFit: SaturatingFit,
  xTargetHours: number,
  horizonDays: number,
  stale: { hours: number; changeIndex: number } | null,
  scatter: number,
): number[] {
  const usable = series.filter((s) => s.rate !== null);
  if (usable.length === 0) {
    // No read ever listed this trophy: a flat 0.5% with a wide spread.
    const spread = 1.5;
    return Z.map((z) =>
      round4(Math.min(100, Math.max(0, invLogitPct(logit(0.5) + z * spread)))),
    );
  }

  const tLast = series[series.length - 1].tH;
  const inWindow = usable.filter((s) => tLast - s.tH <= WINDOW_H);
  const fitPoints = inWindow.length >= 3 ? inWindow : usable;

  // Earners side: fit the saturating curve to the earner counts.
  const earnersFit = fitSaturating(
    fitPoints.map((s) => s.u),
    fitPoints.map((s) => s.earners as number),
    fitPoints.map((s) => Math.pow(0.5, (tLast - s.tH) / HALF_LIFE_H)),
  );
  let medianRate =
    (earnersFit.predict(xTargetHours) /
      Math.max(playersFit.predict(xTargetHours), 1)) *
    100;

  let staleSpread = 0;
  if (stale !== null) {
    const at = series[stale.changeIndex];
    const frozenRate = ((at.earners ?? 0) / Math.max(at.players, 1)) * 100;
    if (frozenRate >= UNIVERSAL_RATE) {
      // Universal trophy: every tracked player holds it, and new arrivals
      // earn it too, so its rate is not diluted by the growing base.
      medianRate = frozenRate;
    } else {
      // Earners carry forward on their own pre-freeze trend when there is
      // one to read; with a flat frozen page they simply stay put.
      let earnerFactor = 1;
      const pre = series
        .slice(0, stale.changeIndex + 1)
        .filter((s) => s.earners !== null && (s.earners as number) > 0);
      if (pre.length >= 3) {
        const px = pre.map((s) => s.x);
        const py = pre.map((s) => Math.log(s.earners as number));
        const pw = pre.map((s) =>
          Math.pow(0.5, (tLast - s.tH) / HALF_LIFE_H),
        );
        const [eIntercept, eSlope] = weightedLine(px, py, pw);
        earnerFactor = Math.min(
          3,
          Math.max(
            1,
            Math.exp(eIntercept + eSlope * Math.log(1 + xTargetHours)) /
              Math.max(at.earners ?? 1, 1),
          ),
        );
      }
      const growHours = xTargetHours - at.tH;
      const playersAtTarget =
        at.players * Math.pow(2, growHours / PLAYER_DOUBLING_H);
      const earnersAtTarget = (at.earners ?? 0) * earnerFactor;
      medianRate = (earnersAtTarget / Math.max(playersAtTarget, 1)) * 100;
    }
    staleSpread = Math.min(0.9, STALE_SPREAD * Math.sqrt(stale.hours / 4));
  }

  const medianLogit = logit(medianRate);
  const spread = Math.min(
    2.5,
    Math.max(
      0.05,
      Math.sqrt(
        scatter * scatter +
          WALK_SIGMA * WALK_SIGMA * horizonDays +
          staleSpread * staleSpread,
      ),
    ),
  );
  const out = Z.map((z) =>
    round4(
      Math.min(100, Math.max(0, invLogitPct(medianLogit + z * spread))),
    ),
  );
  // Rounding can only tie neighbours, never invert them; enforce it anyway.
  for (let i = 1; i < out.length; i++) out[i] = Math.max(out[i], out[i - 1]);
  return out;
}

function forecastCase(c: Case): {
  trophies: Array<{ id: number; q: number[] }>;
} {
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);

  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const hoursSinceRelease = (iso: string): number =>
    Math.max(0, (Date.parse(iso) - releaseMs) / 3_600_000);
  const logTime = (tH: number): number => Math.log(1 + tH);

  const reads = c.history;
  const last = reads[reads.length - 1];

  // Where did the page last actually change? Walk back while the tracked
  // base and every rate match the previous read.
  let changeIndex = reads.length - 1;
  for (let j = reads.length - 1; j > 0; j--) {
    const a = reads[j];
    const b = reads[j - 1];
    const same =
      a.players_tracked === b.players_tracked &&
      a.rates.length === b.rates.length &&
      a.rates.every((r, i) => r === b.rates[i]);
    if (same) changeIndex = j - 1;
    else break;
  }
  const staleHours = hoursBetween(reads[changeIndex].slot, last.slot);
  const stale =
    staleHours >= STALE_AFTER_H
      ? { hours: staleHours, changeIndex }
      : null;

  const horizonDays = c.horizon_minutes / 1440;
  const xTargetHours = hoursSinceRelease(c.target);
  const tLast = hoursSinceRelease(last.slot);

  const perTrophy: Array<{ series: Point[] }> = c.trophies.map((_, i) => {
    const series: Point[] = reads.map((r) => {
      const tH = hoursSinceRelease(r.slot);
      const rate = r.rates[i] ?? null;
      return {
        x: logTime(tH),
        u: tH,
        rate,
        tH,
        players: r.players_tracked,
        earners:
          rate === null
            ? null
            : Math.round((rate / 100) * r.players_tracked),
        slotTime: r.slot,
      };
    });
    return { series };
  });

  // Tracked-base side: one saturating curve for the whole case.
  const baseUs = reads.map((r) => hoursSinceRelease(r.slot));
  const baseYs = reads.map((r) => r.players_tracked);
  const baseWs = reads.map((r) =>
    Math.pow(0.5, (tLast - hoursSinceRelease(r.slot)) / HALF_LIFE_H),
  );
  const playersFit = fitSaturating(baseUs, baseYs, baseWs);

  // First pass: each trophy's residual scatter, for pooling. The scatter
  // is the residual of the fitted rate -- fitted earners over the fitted
  // base -- in logit space, at the reads themselves.
  const scatters: number[] = perTrophy.map(({ series }) => {
    const usable = series.filter((s) => s.rate !== null);
    const inWindow = usable.filter((s) => tLast - s.tH <= WINDOW_H);
    const fitPoints = inWindow.length >= 3 ? inWindow : usable;
    if (fitPoints.length < 3) return 0;
    const earnersFit = fitSaturating(
      fitPoints.map((s) => s.u),
      fitPoints.map((s) => s.earners as number),
      fitPoints.map((s) => Math.pow(0.5, (tLast - s.tH) / HALF_LIFE_H)),
    );
    let sw = 0;
    let acc = 0;
    for (const s of fitPoints) {
      const fitted =
        (earnersFit.predict(s.u) / Math.max(playersFit.predict(s.u), 1)) *
        100;
      // Winsorized: near 0% or 100% a hair of rate is many logits, so
      // one boundary read must not blow the spread up on its own.
      const r = Math.max(
        -1,
        Math.min(1, logit(s.rate as number) - logit(fitted)),
      );
      const w = Math.pow(0.5, (tLast - s.tH) / HALF_LIFE_H);
      sw += w;
      acc += w * r * r;
    }
    return Math.sqrt(acc / Math.max(sw, 1e-9));
  });
  const pooled = (() => {
    const vals = scatters.filter((s) => s > 0).sort((a, b) => a - b);
    if (vals.length === 0) return 0.25;
    return vals[Math.floor(vals.length / 2)];
  })();

  const trophies = c.trophies.map((t, i) => {
    // Shrink this trophy's scatter toward the pooled value.
    const n = perTrophy[i].series.filter((s) => s.rate !== null).length;
    const own = scatters[i];
    const shrunk = Math.sqrt(
      (n * own * own + SHRINK_WEIGHT * pooled * pooled) /
        (n + SHRINK_WEIGHT),
    );
    return {
      id: t.id,
      q: forecastTrophy(
        perTrophy[i].series,
        playersFit,
        xTargetHours,
        horizonDays,
        stale,
        shrunk,
      ),
    };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
if (inputs.length === 0) throw new Error(`no .in cases in ${INPUT_DIR}`);
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
