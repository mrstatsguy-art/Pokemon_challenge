# PTCG AI Battle Challenge (Playground) — overnight report

## Deliverables (in `/home/claude/ptcg/build/`)
| File | Agent | Deck | Use |
|---|---|---|---|
| `ptcg_mc03_climb2.tar.gz` | determinized Monte-Carlo search, 0.3 s/decision, 400 sims | hill-climbed water/Mega Abomasnow list | **primary** |
| `ptcg_mc03_sample.tar.gz` | same agent | organizer's sample deck | second active slot (Kaggle allows two) |

Both tarballs were extracted and loaded exactly the way `kaggle_environments` loads agents
(`exec` of `main.py`, last callable taken), returned a legal 60-card deck, and completed a best-of-three
inside the official `cabt` environment with statuses `DONE, DONE` and no errors. Upload via the
competition's **Submit Agent** page.

## What was built
* **Engine locally.** The simulator (`libcg.so`), card database and forward model ship inside the public
  `kaggle-environments` package; I reconstructed the small SDK shim (`cg/api.py`, ctypes signatures for
  `SearchBegin/SearchStep`) so everything runs here. 1,431 cards; sample deck legal.
* **Harness.** The MIT-licensed public harness by Kelly Mao (local BO3 runner mirroring the official
  interpreter, heuristic agent, determinized flat-MC search over the engine's forward model). Credited in
  `LICENSE-harness`. A `__main__`-guard bug in my own gauntlet script cost ~1 h early on; fixed.
* **Deck optimisation.** Heuristic-vs-heuristic matches run in <1 s, so I hill-climbed the sample deck with
  ±1 card mutations, 80–100 BO3 matches per candidate against a fixed opponent pool, confirming each gain
  with a fresh seed. Two rounds, 280 candidate decks evaluated.
* **Search budget.** Measured 0.1 / 0.3 / 1.0 s per decision.

## Results (BO3 match score, Wilson 95% CI, seats alternated)
| Test | Matches | Score |
|---|---|---|
| sample deck vs my 4 homebrew archetypes (heuristic pilots) | 30 each | 0.77–0.93 for sample |
| deck climb round 1: sample → climb1 (vs sample + Ursaluna) | 80/candidate | 0.650 → 0.731 |
| deck climb round 2: climb1 → climb2 (vs sample + Ursaluna + climb1) | 100/candidate | 0.617 → 0.723 |
| MC 0.1 s vs heuristic, climb2 mirror | 40 | 0.700 [0.55, 0.82] |
| MC 0.3 s vs heuristic, climb2 mirror | 40 | **0.850 [0.71, 0.93]** |
| MC 0.3 s (climb2) vs heuristic (sample) | 40 | **0.875 [0.74, 0.95]** |
| MC 1.0 s vs MC 0.3 s | 30 | 0.533 [0.36, 0.70] — no gain; `max_sims` cap binds |
| MC 0.3 s time use | — | ≤14 s of the 600 s pool per match |

Deck changes accepted by the climb (vs the sample list): −3 Basic Water Energy, −1 Kyogre, −1 Surfing Beach,
+1 Mega Signal, +1 Powerglass, +1 Perrin, +1 Pokégear 3.0, +1 Hilda.

## Honest assessment
* The public harness author measured his MC agent at ~0.02 vs the strongest public bot (an RL-trained
  transformer policy). I could not install torch here (PyPI blocked), so no learned policy. Expect a
  mid-ladder placement, not the top.
* The deck was optimised for *my* pilot against *my* opponent pool; ladder opponents differ. The organizer's
  sample deck is itself solid (it beat every archetype I hand-built), which is why I climbed from it.

## Next steps, in order of expected value
1. Upload both tarballs; watch the ladder rating for a day.
2. Raise `max_sims` with the 0.3–0.8 s budget (test running: `results/mc_budget3.log`).
3. With Kaggle GPU + torch: port a policy network and use it as the rollout policy / prior.
4. Pull the public episodes dataset to fit a Bradley-Terry deck meta and pick a deck that beats what people
   actually play.

## Reproduce
```bash
export PTCG_CG_DIR=/home/claude/ptcg
python3 scripts/gauntlet.py heuristic 30 sample bouffalant ursaluna hooh bellibolt
python3 scripts/deck_climb.py decks/sample.csv decks/climb.csv --n 80 --iters 80
python3 scripts/run_match.py mc heuristic -n 40 -w 2 --a-kw budget=0.3 --deck-a decks/climb2.csv
python3 scripts/build_submission.py --deck decks/climb2.csv --config '{"agent":"mc","budget":0.3,"max_sims":400}' --name ptcg_mc03_climb2
```
Logs: `results/*.log`.
