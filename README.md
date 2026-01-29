# Synthetic PCFG

A Python library for generating synthetic Probabilistic Context-Free Grammars (PCFGs) with desirable properties for machine learning experiments.

## Features

- Generate PCFGs in Chomsky Normal Form (CNF)
- Consistent grammars (probability of all strings sums to 1)
- Configurable length distributions (Poisson, WSJ-like, child-directed speech)
- Configurable lexical distributions (Zipfian via log-normal, Pitman-Yor, Dirichlet)
- Two grammar types:
  - **Non-trivial CFG backbone**: Sparse production rules
  - **Full CFG**: All possible productions with sampled probabilities

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate a Grammar

```python
from syntheticpcfg import PCFG, Sampler, load_pcfg_from_file

# Load an existing grammar
grammar = load_pcfg_from_file("data/manual/dyck2.pcfg")

# Sample strings from the grammar
sampler = Sampler(grammar)
for _ in range(10):
    tree = sampler.sample_tree()
    print(tree)
```

### Generate a Random Grammar

```python
from syntheticpcfg.pcfgfactory import PCFGFactory

factory = PCFGFactory()
factory.cfgfactory.number_nonterminals = 10
factory.cfgfactory.number_terminals = 1000

grammar = factory.sample()
grammar.store("my_grammar.pcfg")
```

## Command-Line Tools

### Generate a grammar with sparse CFG backbone

```bash
python -m syntheticpcfg.sample_grammar output.pcfg \
    --nonterminals 10 \
    --terminals 10000 \
    --binaryproductions 40 \
    --lexicalproductions 10000
```

### Generate a grammar with full CFG

```bash
python -m syntheticpcfg.sample_fullgrammar output.pcfg \
    --nonterminals 10 \
    --terminals 10000
```

### Sample from a grammar

```bash
# Sample trees with probabilities
python -m syntheticpcfg.sample_corpus grammar.pcfg samples.txt --samples 1000

# Sample only yields (strings)
python -m syntheticpcfg.sample_corpus grammar.pcfg samples.txt \
    --samples 1000 --yieldonly --omitprobs
```

## Grammar File Format

Grammars are stored in a simple text format:

```
# Comments start with #
1.0 S -> A B
0.5 A -> a
0.5 A -> c
1.0 B -> b
```

Each line contains: `probability LHS -> RHS1 [RHS2]`

## Key Parameters

### Length Distribution

- `--poisson LAMBDA`: Zero-truncated Poisson (default λ=5)
- `--wsjlength`: Distribution matching Wall Street Journal corpus
- `--cdslength`: Distribution matching child-directed speech

### Lexical Distribution

- `--sigma SIGMA`: Log-normal prior (default σ=3.0, higher = more Zipfian)
- `--pitmanyor`: Use Pitman-Yor process
- `--dirichletparam ALPHA`: Symmetric Dirichlet prior

## API Reference

### PCFG Class

```python
class PCFG:
    # Core attributes
    start: str                    # Start symbol (default "S")
    nonterminals: Set[str]        # Set of nonterminal symbols
    terminals: Set[str]           # Set of terminal symbols
    productions: List[Tuple]      # List of productions
    parameters: Dict[Tuple, float]  # Production probabilities
    
    # Key methods
    def expected_length(self) -> float
    def derivational_entropy(self) -> float
    def nonterminal_expectations(self) -> Dict[str, float]
    def sample_tree(self) -> Tuple  # Use Sampler class instead
    def store(self, filename: str) -> None
    def copy(self) -> PCFG
```

### Sampler Class

```python
class Sampler:
    def __init__(self, pcfg: PCFG, max_depth: int = 100)
    def sample_tree(self) -> Tuple
    def sample_string(self) -> List[str]
```

### InsideComputation Class

```python
class InsideComputation:
    def __init__(self, pcfg: PCFG)
    def inside_probability(self, sentence: List[str]) -> float
    def inside_log_probability(self, sentence: List[str]) -> float
    def viterbi_parse(self, sentence: List[str]) -> Tuple
    def count_parses(self, sentence: List[str]) -> int
```

## Example Grammars

The `data/manual/` directory contains example grammars:

- `dyck2.pcfg`: Dyck-2 language (two types of balanced brackets)
- `example1.pcfg`: Simple 3-nonterminal grammar
- `hardest.cfg`: Challenging grammar for parsing

## Development

### Running Tests

```bash
cd syntheticpcfg
python -m pytest syntheticpcfg/test_syntheticpcfg.py -v
```

### Code Style

The project uses:
- 4-space indentation
- PEP 8 naming conventions
- Relative imports within the package

## License

See LICENSE file for details.

## Citation

If you use this software in your research, please cite appropriately.
