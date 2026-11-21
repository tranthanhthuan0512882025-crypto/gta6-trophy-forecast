# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tranthanhthuan0512882025-crypto/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tranthanhthuan0512882025-crypto/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase. It reads one case — every read the house made of the game's trophy page before the checkpoint — and writes, for each of the 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the case's target, for the lowest pinball loss on the logit scale.

## How it works

A rate is earners over tracked players, and the program forecasts the two sides separately instead of extrapolating the bare rate. Earners are fit with a saturating curve against hours since release (players reach a trophy's content, then bend toward how many will ever hold it); the tracked base is fit the same way (launch buyers arrive, then bend toward its ceiling). The median rate at the target is fitted earners over fitted base. When the tail of the history has not changed for an hour or more, the page is treated as frozen: earners are carried forward and the base is grown at a launch-night doubling prior of 2 hours, so the median is diluted by the players who arrived while the page was frozen — except a trophy every tracked player already holds, which new arrivals earn too. The spread is built from the fit's residual scatter (shrunk toward the pooled value across trophies), a random-walk floor growing with the square root of the horizon, and a staleness term — sized from the data, never from a fixed multiplier.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` runs on Node 22, standard library only, no network, deterministic: two runs on the same case write the same bytes. It reads every `*.in` file in the input directory and writes one `*.out` file per case.

## Validation

- On the kit's practice cases (`p00`, `p01`) it writes valid forecasts for all 55 trophies in ~0.2 s, byte-identical across runs, each output well under the 65,536-byte cap.
- Against a hypothetical practice answer (earners held constant, tracked base at the refreshed read's 3,868 players, the 100% trophy held at 100%) it scores about 0.12 / 0.10 pinball loss, against the starter's ~1.69 on the same hypothetical — the dilution the refresh brings is the whole game on the practice cut. That is a hypothetical, not a measured score: this competition has no local grade, the server grades every entry.
- On synthetic final-like cases (saturating rates, log-linear growth, and an earners-over-growing-base world) it stays competitive with the starter while forecasting real movement the starter's flat median misses; its median error is lowest in the dilution world, where the two-sided model matters most.

---

<sub>Entered on [make.fun](https://make.fun/?ref=02mf7dke&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
