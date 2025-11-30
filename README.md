# Project Chess God

A personal, after-hours project to heve some fun and build AI-powered chess engines.
The aim is to iteratively improve engine strength, experiment with training approaches,
and learn more about engine architectures, evaluation, and tooling.
The project as you look at it is probably in the state of utter madness as I'm figuring
shit out :)

## Highlights

- Dataset: Lichess 1M games — https://www.kaggle.com/datasets/aapohermankoskinen/lichess-1-million-chess-games
- Dataset: NNUE stockfish challange, almost 2 million games — https://www.kaggle.com/competitions/train-your-own-stockfish-nnue/data?select=test.csv
- Uses JCchess (https://github.com/johncheetham/jcchess) for game UI, PGN handling and testing.
- Engines communicate via UCI (Universal Chess Interface) — engines here include a UCI translation layer.

## Repository layout (suggested)

- `engines/` — engine save directory
- `src/` — egine development
- `data/` — datasets, preprocessing scripts, and sampling utilities
- `experiments/` — training configs, logs, and evaluation scripts
- `tools/` — helpers (position generators, self-play scripts, analysis tools)

### Quick start (high level)

1. Clone the repo.
2. Inspect the engine-specific README in `engines/<engine-name>/README.md` for setup details.
3. Use JCchess to load a UCI-compatible engine (see `engines/README.md`).

### Example local workflow

1. Prepare data: place raw PGN files into `data/` and run preprocessing scripts.
2. Train or iterate on an engine under `engines/`.
3. Launch JCchess and configure it to use your engine's UCI interface for testing/play.

## License & contact

GNU GENERAL PUBLIC LICENSE v3