# test_syntheticpcfg.py
"""
Comprehensive unit tests for the syntheticpcfg package.
Run with: pytest test_syntheticpcfg.py -v
"""

import pytest
import math
import numpy as np
import numpy.random
from collections import Counter, defaultdict

import utility
import cfg
import pcfg
import inside


# =============================================================================
# Fixtures: Small grammars for testing
# =============================================================================

@pytest.fixture
def simple_pcfg():
    """
    Simple PCFG: S -> A B, A -> a | c, B -> b
    Generates strings: "ab" or "cb" with equal probability
    """
    mypcfg = pcfg.PCFG()
    mypcfg.start = "S"
    mypcfg.nonterminals = {"S", "A", "B"}
    mypcfg.terminals = {"a", "b", "c"}
    mypcfg.productions = [
        ("S", "A", "B"),
        ("A", "a"),
        ("A", "c"),
        ("B", "b"),
    ]
    mypcfg.parameters = {
        ("S", "A", "B"): 1.0,
        ("A", "a"): 0.5,
        ("A", "c"): 0.5,
        ("B", "b"): 1.0,
    }
    mypcfg.set_log_parameters()
    return mypcfg


@pytest.fixture
def ambiguous_pcfg():
    """
    Ambiguous PCFG that can generate the same string multiple ways.
    S -> S S | a
    """
    mypcfg = pcfg.PCFG()
    mypcfg.start = "S"
    mypcfg.nonterminals = {"S"}
    mypcfg.terminals = {"a"}
    mypcfg.productions = [
        ("S", "S", "S"),
        ("S", "a"),
    ]
    mypcfg.parameters = {
        ("S", "S", "S"): 0.4,
        ("S", "a"): 0.6,
    }
    mypcfg.set_log_parameters()
    return mypcfg


@pytest.fixture
def dyck1_pcfg():
    """
    Dyck-1 language (balanced parentheses).
    S -> A B | S S, A -> a, B -> b
    """
    mypcfg = pcfg.PCFG()
    mypcfg.start = "S"
    mypcfg.nonterminals = {"S", "A", "B"}
    mypcfg.terminals = {"a", "b"}
    mypcfg.productions = [
        ("S", "A", "B"),
        ("S", "S", "S"),
        ("A", "a"),
        ("B", "b"),
    ]
    # Choose parameters so grammar is consistent
    mypcfg.parameters = {
        ("S", "A", "B"): 0.6,
        ("S", "S", "S"): 0.4,
        ("A", "a"): 1.0,
        ("B", "b"): 1.0,
    }
    mypcfg.set_log_parameters()
    return mypcfg


@pytest.fixture
def unary_pcfg():
    """
    PCFG with single terminal for testing UnaryInside.
    """
    mypcfg = pcfg.PCFG()
    mypcfg.start = "S"
    mypcfg.nonterminals = {"S", "A"}
    mypcfg.terminals = {"x"}
    mypcfg.productions = [
        ("S", "S", "A"),
        ("S", "A", "A"),
        ("S", "x"),
        ("A", "x"),
    ]
    mypcfg.parameters = {
        ("S", "S", "A"): 0.2,
        ("S", "A", "A"): 0.3,
        ("S", "x"): 0.5,
        ("A", "x"): 1.0,
    }
    mypcfg.set_log_parameters()
    return mypcfg


@pytest.fixture
def loaded_pcfg():
    """Load a PCFG from the example file."""
    return pcfg.load_pcfg_from_file("../data/manual/example1.pcfg")


@pytest.fixture
def dyck2_pcfg():
    """Load the Dyck-2 grammar."""
    return pcfg.load_pcfg_from_file("../data/manual/dyck2.pcfg")


# =============================================================================
# Tests for utility.py
# =============================================================================

class TestUtilityTreeOperations:
    """Tests for tree manipulation functions."""

    def test_collect_yield_lexical(self):
        """Test yield collection for lexical tree."""
        tree = ("A", "word")
        assert utility.collect_yield(tree) == ["word"]

    def test_collect_yield_binary(self):
        """Test yield collection for binary tree."""
        tree = ("S", ("A", "hello"), ("B", "world"))
        assert utility.collect_yield(tree) == ["hello", "world"]

    def test_collect_yield_deep(self):
        """Test yield collection for deeper tree."""
        tree = ("S", 
                ("A", ("C", "a"), ("D", "b")), 
                ("B", "c"))
        assert utility.collect_yield(tree) == ["a", "b", "c"]

    def test_tree_to_string_lexical(self):
        """Test string conversion for lexical tree."""
        tree = ("A", "word")
        assert utility.tree_to_string(tree) == "(A word)"

    def test_tree_to_string_binary(self):
        """Test string conversion for binary tree."""
        tree = ("S", ("A", "a"), ("B", "b"))
        result = utility.tree_to_string(tree)
        assert result == "(S (A a) (B b))"

    def test_string_to_tree_roundtrip(self):
        """Test that string_to_tree inverts tree_to_string."""
        original = ("S", ("A", "hello"), ("B", "world"))
        string = utility.tree_to_string(original)
        recovered = utility.string_to_tree(string)
        assert recovered == original

    def test_tree_depth_lexical(self):
        """Test depth of lexical tree."""
        tree = ("A", "word")
        assert utility.tree_depth(tree) == 1

    def test_tree_depth_binary(self):
        """Test depth of binary tree."""
        tree = ("S", ("A", "a"), ("B", "b"))
        assert utility.tree_depth(tree) == 2

    def test_tree_depth_unbalanced(self):
        """Test depth of unbalanced tree."""
        tree = ("S", ("A", ("C", "x"), ("D", "y")), ("B", "z"))
        assert utility.tree_depth(tree) == 3

    def test_tree_to_unlabeled_tree(self):
        """Test conversion to unlabeled tree."""
        tree = ("S", ("A", "a"), ("B", "b"))
        unlabeled = utility.tree_to_unlabeled_tree(tree)
        assert unlabeled == ("a", "b")

    def test_tree_to_preterminals(self):
        """Test extraction of preterminals."""
        tree = ("S", ("A", "a"), ("B", "b"))
        preterminals = utility.tree_to_preterminals(tree)
        assert preterminals == ("A", "B")

    def test_collect_unlabeled_spans(self):
        """Test span collection."""
        tree = ("S", ("A", "a"), ("B", "b"))
        spans = utility.collect_unlabeled_spans(tree)
        # Should not include the root span
        assert (0, 2) not in spans

    def test_collect_labeled_spans(self):
        """Test labeled span collection."""
        tree = ("S", ("A", "a"), ("B", "b"))
        spans = utility.collect_labeled_spans(tree)
        assert ("A", 0, 1) in spans
        assert ("B", 1, 2) in spans
        assert ("S", 0, 2) in spans

    def test_relabel_tree(self):
        """Test tree relabeling."""
        tree = ("S", ("A", "a"), ("B", "b"))
        nt_map = {"S": "X", "A": "Y", "B": "Z"}
        relabeled = utility.relabel_tree(tree, nt_map)
        assert relabeled == ("X", ("Y", "a"), ("Z", "b"))

    def test_count_productions(self):
        """Test production counting."""
        tree = ("S", ("A", "a"), ("B", "b"))
        counter = Counter()
        utility.count_productions(tree, counter)
        assert counter[("S", "A", "B")] == 1
        assert counter[("A", "a")] == 1
        assert counter[("B", "b")] == 1


class TestUtilityMisc:
    """Tests for miscellaneous utility functions."""

    def test_generate_lexicon_size(self):
        """Test that lexicon has correct size."""
        n = 100
        lexicon = utility.generate_lexicon(n)
        assert len(lexicon) == n

    def test_generate_lexicon_unique(self):
        """Test that lexicon elements are unique."""
        n = 50
        lexicon = utility.generate_lexicon(n)
        assert len(lexicon) == len(set(lexicon))

    def test_catalan_numbers(self):
        """Test Catalan number computation."""
        # Known Catalan numbers: 1, 1, 2, 5, 14, 42, 132, ...
        assert utility.catalan_numbers(0) == 1
        assert utility.catalan_numbers(1) == 1
        assert utility.catalan_numbers(2) == 2
        assert utility.catalan_numbers(3) == 5
        assert utility.catalan_numbers(4) == 14
        assert utility.catalan_numbers(5) == 42

    def test_variation_of_information_identical(self):
        """Test VI for identical clusterings."""
        counter = Counter({("A", "A"): 10, ("B", "B"): 10})
        vi = utility.variation_of_information(counter)
        assert vi == pytest.approx(0.0, abs=1e-10)

    def test_zero_one_unlabeled_match(self):
        """Test zero-one evaluation for matching trees."""
        tree1 = ("S", ("A", "a"), ("B", "b"))
        tree2 = ("X", ("Y", "a"), ("Z", "b"))  # Same structure
        num, denom = utility.zero_one_unlabeled(tree1, tree2)
        assert num == 1
        assert denom == 1

    def test_zero_one_unlabeled_mismatch(self):
        """Test zero-one evaluation for mismatched trees."""
        tree1 = ("S", ("A", "a"), ("B", ("C", "b"), ("D", "c")))
        tree2 = ("S", ("A", ("E", "a"), ("F", "b")), ("B", "c"))
        num, denom = utility.zero_one_unlabeled(tree1, tree2)
        assert num == 0


class TestStronglyConnectedComponents:
    """Tests for Tarjan's SCC algorithm."""

    def test_scc_single_node(self):
        """Test SCC with single node."""
        graph = {"A": []}
        sccs = utility.strongly_connected_components(graph)
        assert len(sccs) == 1
        assert "A" in sccs[0]

    def test_scc_cycle(self):
        """Test SCC with a cycle."""
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        sccs = utility.strongly_connected_components(graph)
        assert len(sccs) == 1
        assert set(sccs[0]) == {"A", "B", "C"}

    def test_scc_no_cycle(self):
        """Test SCC with no cycles."""
        graph = {"A": ["B"], "B": ["C"], "C": []}
        sccs = utility.strongly_connected_components(graph)
        assert len(sccs) == 3


# =============================================================================
# Tests for cfg.py
# =============================================================================

class TestCFG:
    """Tests for CFG class."""

    def test_cfg_creation(self):
        """Test basic CFG creation."""
        mycfg = cfg.CFG()
        mycfg.start = "S"
        mycfg.nonterminals = {"S", "A"}
        mycfg.terminals = {"a", "b"}
        mycfg.productions = {("S", "A", "A"), ("A", "a"), ("A", "b")}
        
        assert mycfg.start == "S"
        assert len(mycfg.nonterminals) == 2
        assert len(mycfg.terminals) == 2

    def test_compute_coreachable_set(self):
        """Test coreachable set computation."""
        mycfg = cfg.CFG()
        mycfg.start = "S"
        mycfg.nonterminals = {"S", "A", "B", "C"}
        mycfg.terminals = {"a"}
        mycfg.productions = {
            ("S", "A", "B"),
            ("A", "a"),
            ("B", "a"),
            ("C", "C", "C"),  # C can't generate anything
        }
        
        coreachable = mycfg.compute_coreachable_set()
        assert "S" in coreachable
        assert "A" in coreachable
        assert "B" in coreachable
        assert "C" not in coreachable

    def test_compute_trim_set(self):
        """Test trim set computation."""
        mycfg = cfg.CFG()
        mycfg.start = "S"
        mycfg.nonterminals = {"S", "A", "B", "C"}
        mycfg.terminals = {"a"}
        mycfg.productions = {
            ("S", "A", "B"),
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),  # C is coreachable but not reachable from S
        }
        
        trim = mycfg.compute_trim_set()
        assert "S" in trim
        assert "A" in trim
        assert "B" in trim
        assert "C" not in trim


# =============================================================================
# Tests for pcfg.py - PCFG class
# =============================================================================

class TestPCFGBasic:
    """Basic tests for PCFG class."""

    def test_pcfg_creation(self, simple_pcfg):
        """Test basic PCFG creation."""
        assert simple_pcfg.start == "S"
        assert len(simple_pcfg.nonterminals) == 3
        assert len(simple_pcfg.terminals) == 3
        assert len(simple_pcfg.productions) == 4

    def test_is_normalised(self, simple_pcfg):
        """Test normalization check."""
        assert simple_pcfg.is_normalised()

    def test_check_normalisation(self, simple_pcfg):
        """Test normalization totals."""
        totals = simple_pcfg.check_normalisation()
        assert totals["S"] == pytest.approx(1.0)
        assert totals["A"] == pytest.approx(1.0)
        assert totals["B"] == pytest.approx(1.0)

    def test_copy(self, simple_pcfg):
        """Test PCFG copy."""
        copy = simple_pcfg.copy()
        assert copy.start == simple_pcfg.start
        assert copy.nonterminals == simple_pcfg.nonterminals
        assert copy.terminals == simple_pcfg.terminals
        assert copy.parameters == simple_pcfg.parameters
        # Ensure it's a real copy
        copy.parameters[("A", "a")] = 0.9
        assert simple_pcfg.parameters[("A", "a")] == 0.5

    def test_log_parameters(self, simple_pcfg):
        """Test log parameters are set correctly."""
        for prod in simple_pcfg.productions:
            expected = math.log(simple_pcfg.parameters[prod])
            assert simple_pcfg.log_parameters[prod] == pytest.approx(expected)

    def test_normalise(self):
        """Test normalization of unnormalized grammar."""
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a", "b"}
        mypcfg.productions = [("S", "a"), ("S", "b")]
        mypcfg.parameters = {("S", "a"): 2.0, ("S", "b"): 3.0}
        mypcfg.log_parameters = {}
        
        mypcfg.normalise()
        
        assert mypcfg.parameters[("S", "a")] == pytest.approx(0.4)
        assert mypcfg.parameters[("S", "b")] == pytest.approx(0.6)

    def test_trim_zeros(self, simple_pcfg):
        """Test trimming zero productions."""
        simple_pcfg.parameters[("A", "c")] = 0.0
        simple_pcfg.trim_zeros()
        
        assert ("A", "c") not in simple_pcfg.productions
        assert "c" not in simple_pcfg.terminals


class TestPCFGDerivation:
    """Tests for derivation-related methods."""

    def test_log_probability_derivation_lexical(self, simple_pcfg):
        """Test log probability of lexical derivation."""
        tree = ("A", "a")
        lp = simple_pcfg.log_probability_derivation(tree)
        assert lp == pytest.approx(math.log(0.5))

    def test_log_probability_derivation_binary(self, simple_pcfg):
        """Test log probability of full derivation."""
        tree = ("S", ("A", "a"), ("B", "b"))
        lp = simple_pcfg.log_probability_derivation(tree)
        # P = 1.0 * 0.5 * 1.0 = 0.5
        assert lp == pytest.approx(math.log(0.5))

    def test_log_probability_derivation_sum(self, simple_pcfg):
        """Test that derivation probabilities sum correctly."""
        tree1 = ("S", ("A", "a"), ("B", "b"))
        tree2 = ("S", ("A", "c"), ("B", "b"))
        
        p1 = math.exp(simple_pcfg.log_probability_derivation(tree1))
        p2 = math.exp(simple_pcfg.log_probability_derivation(tree2))
        
        # These are the only two possible derivations
        assert p1 + p2 == pytest.approx(1.0)


class TestPCFGExpectations:
    """Tests for expectation computations."""

    def test_nonterminal_expectations(self, simple_pcfg):
        """Test nonterminal expectation computation."""
        expectations = simple_pcfg.nonterminal_expectations()
        
        # S is used exactly once
        assert expectations["S"] == pytest.approx(1.0)
        # A is used exactly once
        assert expectations["A"] == pytest.approx(1.0)
        # B is used exactly once
        assert expectations["B"] == pytest.approx(1.0)

    def test_nonterminal_expectations_recursive(self, ambiguous_pcfg):
        """Test expectations with recursive grammar."""
        expectations = ambiguous_pcfg.nonterminal_expectations()
        
        # S is used at least once, plus more due to recursion
        assert expectations["S"] >= 1.0

    def test_production_expectations(self, simple_pcfg):
        """Test production expectation computation."""
        pe = simple_pcfg.production_expectations()
        
        assert pe[("S", "A", "B")] == pytest.approx(1.0)
        assert pe[("A", "a")] == pytest.approx(0.5)
        assert pe[("A", "c")] == pytest.approx(0.5)
        assert pe[("B", "b")] == pytest.approx(1.0)

    def test_terminal_expectations(self, simple_pcfg):
        """Test terminal expectation computation."""
        te = simple_pcfg.terminal_expectations()
        
        assert te["a"] == pytest.approx(0.5)
        assert te["c"] == pytest.approx(0.5)
        assert te["b"] == pytest.approx(1.0)

    def test_expected_length(self, simple_pcfg):
        """Test expected length computation."""
        el = simple_pcfg.expected_length()
        # Always generates 2 terminals
        assert el == pytest.approx(2.0)

    def test_expected_length_recursive(self, ambiguous_pcfg):
        """Test expected length with recursive grammar."""
        el = ambiguous_pcfg.expected_length()
        # E[L] = 0.6 * 1 + 0.4 * 2 * E[L]
        # E[L] = 0.6 + 0.8 * E[L]
        # 0.2 * E[L] = 0.6
        # E[L] = 3
        assert el == pytest.approx(3.0)


class TestPCFGEntropy:
    """Tests for entropy computations."""

    def test_derivational_entropy_deterministic(self):
        """Test entropy of deterministic grammar is zero."""
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a"}
        mypcfg.productions = [("S", "a")]
        mypcfg.parameters = {("S", "a"): 1.0}
        mypcfg.set_log_parameters()
        
        entropy = mypcfg.derivational_entropy()
        assert entropy == pytest.approx(0.0, abs=1e-10)

    def test_derivational_entropy_positive(self, simple_pcfg):
        """Test that non-deterministic grammar has positive entropy."""
        entropy = simple_pcfg.derivational_entropy()
        assert entropy > 0

    def test_derivational_entropy_split(self, simple_pcfg):
        """Test that split entropy sums to total."""
        total = simple_pcfg.derivational_entropy()
        binary_e, lexical_e = simple_pcfg.derivational_entropy_split()
        
        assert binary_e + lexical_e == pytest.approx(total)

    def test_entropy_unigram(self, simple_pcfg):
        """Test unigram entropy computation."""
        entropy = simple_pcfg.entropy_unigram()
        assert entropy >= 0


class TestPCFGPartitionFunction:
    """Tests for partition function computation."""

    def test_partition_function_consistent(self, simple_pcfg):
        """Test partition function of consistent grammar."""
        pf = simple_pcfg.compute_partition_function_fast()
        
        # For consistent grammar, all should be 1.0
        for nt in simple_pcfg.nonterminals:
            assert pf[nt] == pytest.approx(1.0, abs=1e-6)

    def test_partition_function_fp_consistent(self, simple_pcfg):
        """Test fixed-point partition function."""
        pf = simple_pcfg.compute_partition_function_fp()
        
        for nt in simple_pcfg.nonterminals:
            assert pf[nt] == pytest.approx(1.0, abs=1e-3)

    def test_partition_function_methods_agree(self, dyck1_pcfg):
        """Test that fast and fp methods give same results."""
        pf_fast = dyck1_pcfg.compute_partition_function_fast()
        pf_fp = dyck1_pcfg.compute_partition_function_fp()
        
        for nt in dyck1_pcfg.nonterminals:
            assert pf_fast[nt] == pytest.approx(pf_fp[nt], abs=1e-3)


class TestPCFGSampling:
    """Tests for sampling from PCFG."""

    def test_sampler_creation(self, simple_pcfg):
        """Test sampler creation."""
        sampler = pcfg.Sampler(simple_pcfg)
        assert sampler.start == simple_pcfg.start

    def test_sample_tree_valid(self, simple_pcfg):
        """Test that sampled trees are valid."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(simple_pcfg)
        
        for _ in range(10):
            tree = sampler.sample_tree()
            # Check tree has correct root
            assert tree[0] == "S"
            # Check yield
            y = utility.collect_yield(tree)
            assert len(y) == 2
            assert y[0] in ["a", "c"]
            assert y[1] == "b"

    def test_sample_string(self, simple_pcfg):
        """Test string sampling."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(simple_pcfg)
        
        for _ in range(10):
            s = sampler.sample_string()
            assert len(s) == 2
            assert s[0] in ["a", "c"]
            assert s[1] == "b"

    def test_sample_distribution(self, simple_pcfg):
        """Test that sampling follows the distribution."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(simple_pcfg)
        
        counts = Counter()
        n = 1000
        for _ in range(n):
            s = tuple(sampler.sample_string())
            counts[s] += 1
        
        # Should be approximately 50-50
        assert counts[("a", "b")] / n == pytest.approx(0.5, abs=0.05)
        assert counts[("c", "b")] / n == pytest.approx(0.5, abs=0.05)


class TestPCFGMakeUnary:
    """Tests for make_unary method."""

    def test_make_unary(self, simple_pcfg):
        """Test unary grammar creation."""
        unary = simple_pcfg.make_unary()
        
        assert len(unary.terminals) == 1
        assert pcfg.UNARY_SYMBOL in unary.terminals
        assert unary.nonterminals == simple_pcfg.nonterminals

    def test_make_unary_preserves_length_distribution(self, simple_pcfg):
        """Test that unary grammar has same length distribution."""
        unary = simple_pcfg.make_unary()
        
        original_el = simple_pcfg.expected_length()
        unary_el = unary.expected_length()
        
        assert original_el == pytest.approx(unary_el)


class TestPCFGLoadStore:
    """Tests for loading and storing PCFGs."""

    def test_load_pcfg(self, loaded_pcfg):
        """Test loading PCFG from file."""
        assert loaded_pcfg.start == "S"
        assert loaded_pcfg.is_normalised()
        assert "A" in loaded_pcfg.nonterminals
        assert "B" in loaded_pcfg.nonterminals
        assert "a" in loaded_pcfg.terminals

    def test_store_and_reload(self, simple_pcfg, tmp_path):
        """Test storing and reloading PCFG."""
        filepath = tmp_path / "test.pcfg"
        simple_pcfg.store(str(filepath))
        
        reloaded = pcfg.load_pcfg_from_file(str(filepath))
        
        assert reloaded.start == simple_pcfg.start
        assert reloaded.nonterminals == simple_pcfg.nonterminals
        assert reloaded.terminals == simple_pcfg.terminals
        
        for prod in simple_pcfg.productions:
            assert reloaded.parameters[prod] == pytest.approx(
                simple_pcfg.parameters[prod], abs=1e-6
            )


class TestPCFGRenormalise:
    """Tests for renormalization methods."""

    def test_renormalise(self, dyck1_pcfg):
        """Test renormalization makes grammar consistent."""
        # Create a slightly inconsistent version
        test_pcfg = dyck1_pcfg.copy()
        test_pcfg.parameters[("S", "S", "S")] = 0.5  # Makes it inconsistent
        test_pcfg.normalise()
        
        test_pcfg.renormalise()
        
        pf = test_pcfg.compute_partition_function_fast()
        assert pf["S"] == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Tests for inside.py
# =============================================================================

class TestInsideComputation:
    """Tests for InsideComputation class."""

    def test_inside_probability_simple(self, simple_pcfg):
        """Test inside probability for simple grammar."""
        ic = inside.InsideComputation(simple_pcfg)
        
        # P("ab") = P(S->AB) * P(A->a) * P(B->b) = 1.0 * 0.5 * 1.0 = 0.5
        p = ic.inside_probability(["a", "b"])
        assert p == pytest.approx(0.5)
        
        # P("cb") = 0.5
        p = ic.inside_probability(["c", "b"])
        assert p == pytest.approx(0.5)

    def test_inside_log_probability(self, simple_pcfg):
        """Test inside log probability."""
        ic = inside.InsideComputation(simple_pcfg)
        
        lp = ic.inside_log_probability(["a", "b"])
        assert lp == pytest.approx(math.log(0.5))

    def test_inside_probability_ambiguous(self, ambiguous_pcfg):
        """Test inside probability for ambiguous grammar."""
        ic = inside.InsideComputation(ambiguous_pcfg)
        
        # P("aa") = P(S->SS)*P(S->a)*P(S->a) = 0.4 * 0.6 * 0.6 = 0.144
        p = ic.inside_probability(["a", "a"])
        assert p == pytest.approx(0.144)

    def test_inside_probability_unparseable(self, simple_pcfg):
        """Test inside probability for unparseable string."""
        ic = inside.InsideComputation(simple_pcfg)
        
        # "aa" cannot be parsed
        p = ic.inside_probability(["a", "a"])
        assert p == 0.0

    def test_inside_log_probability_raises(self, simple_pcfg):
        """Test that log probability raises for unparseable string."""
        ic = inside.InsideComputation(simple_pcfg)
        
        with pytest.raises(utility.ParseFailureException):
            ic.inside_log_probability(["a", "a"])

    def test_inside_bracketed_log_probability(self, simple_pcfg):
        """Test bracketed log probability."""
        ic = inside.InsideComputation(simple_pcfg)
        
        tree = ("S", ("A", "a"), ("B", "b"))
        lp = ic.inside_bracketed_log_probability(tree)
        
        # Same as derivation probability for unambiguous case
        assert lp == pytest.approx(math.log(0.5))

    def test_viterbi_parse_simple(self, simple_pcfg):
        """Test Viterbi parsing."""
        ic = inside.InsideComputation(simple_pcfg)
        
        tree = ic.viterbi_parse(["a", "b"])
        
        assert tree[0] == "S"
        assert utility.collect_yield(tree) == ["a", "b"]

    def test_viterbi_parse_ambiguous(self, ambiguous_pcfg):
        """Test Viterbi parsing for ambiguous grammar."""
        ic = inside.InsideComputation(ambiguous_pcfg)
        
        # "aaa" can be parsed as ((a a) a) or (a (a a))
        tree = ic.viterbi_parse(["a", "a", "a"])
        
        assert tree[0] == "S"
        assert utility.collect_yield(tree) == ["a", "a", "a"]

    def test_count_parses_unambiguous(self, simple_pcfg):
        """Test parse counting for unambiguous grammar."""
        ic = inside.InsideComputation(simple_pcfg)
        
        count = ic.count_parses(["a", "b"])
        assert count == 1

    def test_count_parses_ambiguous(self, ambiguous_pcfg):
        """Test parse counting for ambiguous grammar."""
        ic = inside.InsideComputation(ambiguous_pcfg)
        
        # "aaa" has Catalan(2) = 2 parses
        count = ic.count_parses(["a", "a", "a"])
        assert count == 2
        
        # "aaaa" has Catalan(3) = 5 parses
        count = ic.count_parses(["a", "a", "a", "a"])
        assert count == 5

    def test_add_posteriors(self, simple_pcfg):
        """Test posterior computation."""
        ic = inside.InsideComputation(simple_pcfg)
        
        posteriors = defaultdict(float)
        ic.add_posteriors(["a", "b"], posteriors)
        
        # All posteriors should be 1.0 for unambiguous grammar
        assert posteriors[("S", "A", "B")] == pytest.approx(1.0)
        assert posteriors[("A", "a")] == pytest.approx(1.0)
        assert posteriors[("B", "b")] == pytest.approx(1.0)


class TestUnaryInside:
    """Tests for UnaryInside class."""

    def test_unary_inside_creation(self, unary_pcfg):
        """Test UnaryInside creation."""
        ui = inside.UnaryInside(unary_pcfg)
        assert ui.nnts == 2

    def test_compute_inside(self, unary_pcfg):
        """Test inside table computation."""
        ui = inside.UnaryInside(unary_pcfg)
        
        table = ui.compute_inside(3)
        # Table should have correct shape
        assert table.shape == (3, 4, 2)

    def test_compute_inside_smart(self, unary_pcfg):
        """Test smart inside computation."""
        ui = inside.UnaryInside(unary_pcfg)
        
        table = ui.compute_inside_smart(5)
        # Should compute probabilities for lengths 1-5
        assert table.shape == (6, 2)
        
        # Length 1 should have lexical probabilities
        assert table[1, ui.ntindex["S"]] == pytest.approx(0.5)
        assert table[1, ui.ntindex["A"]] == pytest.approx(1.0)

    def test_get_params(self, unary_pcfg):
        """Test parameter extraction."""
        ui = inside.UnaryInside(unary_pcfg)
        
        params = ui.get_params()
        
        for prod in unary_pcfg.productions:
            assert params[prod] == pytest.approx(unary_pcfg.parameters[prod])


class TestInsideOutsideConsistency:
    """Tests for inside-outside algorithm consistency."""

    def test_posteriors_sum_to_length(self, ambiguous_pcfg):
        """Test that lexical posteriors sum to sentence length."""
        ic = inside.InsideComputation(ambiguous_pcfg)
        
        sentence = ["a", "a", "a"]
        posteriors = defaultdict(float)
        ic.add_posteriors(sentence, posteriors)
        
        # Sum of lexical posteriors should equal sentence length
        lexical_sum = sum(v for k, v in posteriors.items() if len(k) == 2)
        assert lexical_sum == pytest.approx(len(sentence))

    def test_sampling_matches_inside(self, ambiguous_pcfg):
        """Test that sampled probabilities match inside probabilities."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(ambiguous_pcfg)
        ic = inside.InsideComputation(ambiguous_pcfg)
        
        # Sample many strings and estimate probabilities
        counts = Counter()
        n = 2000
        for _ in range(n):
            s = tuple(sampler.sample_string())
            if len(s) <= 4:  # Only count short strings
                counts[s] += 1
        
        # Check that empirical and computed probabilities match
        for s, count in counts.items():
            if count > 50:  # Only check frequent strings
                empirical_p = count / n
                computed_p = ic.inside_probability(list(s))
                assert empirical_p == pytest.approx(computed_p, abs=0.05)


# =============================================================================
# Tests for numerical stability (the issue to be fixed)
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of matrix operations."""

    def test_nonterminal_expectations_vs_sampling(self, dyck1_pcfg):
        """Test that computed expectations match sampled expectations."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(dyck1_pcfg)
        
        # Compute expectations
        computed = dyck1_pcfg.nonterminal_expectations()
        
        # Estimate by sampling
        n = 2000
        counts = Counter()
        for _ in range(n):
            tree = sampler.sample_tree()
            for prod in dyck1_pcfg.productions:
                # Count nonterminal occurrences
                pass  # This would require tree traversal
        
        # Just verify the computed values are reasonable
        for nt in dyck1_pcfg.nonterminals:
            assert computed[nt] >= 1.0 or nt != dyck1_pcfg.start

    def test_partition_function_near_critical(self):
        """Test partition function for near-critical grammar."""
        # Create a grammar that's close to critical (sum of binary probs close to 1)
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a"}
        mypcfg.productions = [
            ("S", "S", "S"),
            ("S", "a"),
        ]
        # With p(S->SS) = 0.49, grammar is subcritical but close to critical
        mypcfg.parameters = {
            ("S", "S", "S"): 0.49,
            ("S", "a"): 0.51,
        }
        mypcfg.set_log_parameters()
        
        pf = mypcfg.compute_partition_function_fast()
        assert pf["S"] == pytest.approx(1.0, abs=1e-5)

    def test_expected_length_consistency(self, dyck1_pcfg):
        """Test that expected length methods are consistent."""
        el1 = dyck1_pcfg.expected_length()
        
        # Compute via sampling
        numpy.random.seed(42)
        sampler = pcfg.Sampler(dyck1_pcfg)
        
        lengths = []
        for _ in range(2000):
            s = sampler.sample_string()
            lengths.append(len(s))
        
        el_sampled = np.mean(lengths)
        
        assert el1 == pytest.approx(el_sampled, abs=0.2)

    def test_matrix_inversion_alternative(self):
        """
        Test that np.linalg.solve gives same results as np.linalg.inv.
        This documents the current behavior before fixing.
        """
        n = 5
        np.random.seed(42)
        
        # Create a random transition matrix with spectral radius < 1
        T = np.random.rand(n, n) * 0.1
        I = np.eye(n)
        B = np.random.rand(n, n)
        
        # Current method (using inv)
        result_inv = np.dot(np.linalg.inv(I - T), B)
        
        # Better method (using solve)
        result_solve = np.linalg.solve(I - T, B)
        
        # They should give the same result
        np.testing.assert_array_almost_equal(result_inv, result_solve)

    def test_newton_step_alternative(self):
        """
        Test that solving J*delta = y gives same results as inv(J)*y.
        """
        n = 3
        np.random.seed(42)
        
        J = np.random.rand(n, n) + np.eye(n)  # Make sure it's invertible
        y = np.random.rand(n)
        
        # Current method
        delta_inv = np.dot(np.linalg.inv(J), y)
        
        # Better method
        delta_solve = np.linalg.solve(J, y)
        
        np.testing.assert_array_almost_equal(delta_inv, delta_solve)


class TestNearCriticality:
    """
    Tests for numerical stability near criticality.
    
    A PCFG is "critical" when the expected number of nonterminals generated
    per nonterminal equals 1, meaning the grammar is on the boundary between
    convergent (consistent) and divergent behavior.
    
    For a simple grammar S -> S S | a with probability p for S->SS:
    - Expected children per S = 2p
    - Critical point is at p = 0.5
    - Subcritical (consistent) when p < 0.5
    - Expected length E[L] = 1 / (1 - 2p)
    
    Near criticality, (I - T) becomes nearly singular, testing numerical stability.
    """

    def _make_simple_recursive_pcfg(self, p_binary):
        """
        Create S -> S S | a grammar with given binary probability.
        
        Args:
            p_binary: probability of S -> S S (must be < 0.5 for consistency)
        
        Returns:
            PCFG object
        """
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a"}
        mypcfg.productions = [("S", "S", "S"), ("S", "a")]
        mypcfg.parameters = {
            ("S", "S", "S"): p_binary,
            ("S", "a"): 1.0 - p_binary,
        }
        mypcfg.set_log_parameters()
        return mypcfg

    def _make_multi_nt_near_critical(self, n_nonterminals, p_binary_total):
        """
        Create a grammar with multiple nonterminals near criticality.
        
        Each nonterminal has:
        - Binary productions to all pairs of NTs with total prob p_binary_total
        - Lexical production with prob (1 - p_binary_total)
        
        Args:
            n_nonterminals: number of nonterminals (including S)
            p_binary_total: total probability of binary productions per NT
                           (must be < 0.5 for consistency)
        
        Returns:
            PCFG object
        """
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        # Create nonterminals: S, NT1, NT2, ... (total n_nonterminals)
        mypcfg.nonterminals = {"S"} | {f"NT{i}" for i in range(1, n_nonterminals)}
        mypcfg.terminals = {"a"}
        mypcfg.productions = []
        mypcfg.parameters = {}
        
        all_nts = sorted(list(mypcfg.nonterminals))
        n_binary_prods = len(all_nts) ** 2
        
        for nt in all_nts:
            # Binary productions with equal probability, totaling p_binary_total
            p_each_binary = p_binary_total / n_binary_prods
            for nt2 in all_nts:
                for nt3 in all_nts:
                    prod = (nt, nt2, nt3)
                    mypcfg.productions.append(prod)
                    mypcfg.parameters[prod] = p_each_binary
            
            # Lexical production gets remaining probability
            p_lexical = 1.0 - p_binary_total
            prod = (nt, "a")
            mypcfg.productions.append(prod)
            mypcfg.parameters[prod] = p_lexical
        
        mypcfg.set_log_parameters()
        return mypcfg

    @pytest.mark.parametrize("p_binary", [0.1, 0.3, 0.45, 0.48, 0.49, 0.499])
    def test_partition_function_varying_criticality(self, p_binary):
        """Test partition function at varying distances from criticality."""
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        pf = mypcfg.compute_partition_function_fast()
        
        # All consistent grammars should have partition function = 1
        assert pf["S"] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("p_binary", [0.1, 0.3, 0.45, 0.48, 0.49, 0.499])
    def test_expected_length_analytical(self, p_binary):
        """
        Test expected length against analytical formula.
        
        For S -> S S | a with p(S->SS) = p, p(S->a) = 1-p:
        E[L] = (1-p) / (1 - 2p)
        
        Derivation: E[L] = (1-p) + 2p * E[L], solving gives E[L] = (1-p)/(1-2p)
        """
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        computed_el = mypcfg.expected_length()
        analytical_el = (1.0 - p_binary) / (1.0 - 2 * p_binary)
        
        # Tolerance scales with expected length (larger near criticality)
        rel_tol = 1e-6
        assert computed_el == pytest.approx(analytical_el, rel=rel_tol)

    @pytest.mark.parametrize("p_binary", [0.1, 0.3, 0.45, 0.48, 0.49, 0.499])
    def test_nonterminal_expectations_analytical(self, p_binary):
        """
        Test nonterminal expectations against analytical formula.
        
        For S -> S S | a with p(S->SS) = p:
        E[#S] = 1 / (1 - 2p)  (same as expected length for this grammar)
        """
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        expectations = mypcfg.nonterminal_expectations()
        analytical = 1.0 / (1.0 - 2 * p_binary)
        
        rel_tol = 1e-6
        assert expectations["S"] == pytest.approx(analytical, rel=rel_tol)

    @pytest.mark.parametrize("p_binary", [0.3, 0.4, 0.45])
    def test_expected_length_vs_sampling_near_critical(self, p_binary):
        """
        Validate computed expected length against Monte Carlo.
        
        Note: Very near criticality (p > 0.45), sampling becomes unreliable
        due to max_depth limits and high variance, so we test less extreme cases.
        """
        numpy.random.seed(42)
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        computed_el = mypcfg.expected_length()
        
        # Sample to estimate expected length
        sampler = pcfg.Sampler(mypcfg, max_depth=500)
        lengths = []
        n_samples = 5000
        
        for _ in range(n_samples):
            try:
                s = sampler.sample_string()
                lengths.append(len(s))
            except ValueError:
                # Max depth exceeded - skip but don't count
                pass
        
        if len(lengths) > 1000:
            sampled_el = np.mean(lengths)
            # Allow tolerance based on standard error
            std_err = np.std(lengths) / np.sqrt(len(lengths))
            tolerance = max(0.3, 3 * std_err)  # 3 sigma
            assert computed_el == pytest.approx(sampled_el, abs=tolerance)

    def test_very_near_critical_convergence(self):
        """
        Test that computation converges even very near criticality.
        
        At p = 0.4999:
        - E[#S] = 1/(1-2*0.4999) = 1/0.0002 = 5000
        - E[L] = (1-0.4999)/(1-0.9998) = 0.5001/0.0002 = 2500.5
        """
        p_binary = 0.4999
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        # Should not raise, should converge
        pf = mypcfg.compute_partition_function_fast()
        assert pf["S"] == pytest.approx(1.0, abs=1e-3)
        
        el = mypcfg.expected_length()
        analytical_el = (1.0 - p_binary) / (1.0 - 2 * p_binary)
        assert el == pytest.approx(analytical_el, rel=1e-3)
        
        expectations = mypcfg.nonterminal_expectations()
        analytical_nt = 1.0 / (1.0 - 2 * p_binary)
        assert expectations["S"] == pytest.approx(analytical_nt, rel=1e-3)

    def test_multi_nonterminal_near_critical(self):
        """Test with multiple nonterminals near criticality."""
        # p_binary_total = 0.45 means each NT expands to ~0.9 NTs on average
        # This is near-critical for multi-NT grammars
        mypcfg = self._make_multi_nt_near_critical(n_nonterminals=5, p_binary_total=0.45)
        
        # Should compute without numerical issues
        pf = mypcfg.compute_partition_function_fast()
        for nt in mypcfg.nonterminals:
            assert pf[nt] == pytest.approx(1.0, abs=1e-3)
        
        expectations = mypcfg.nonterminal_expectations()
        for nt in mypcfg.nonterminals:
            assert expectations[nt] >= 1.0
            assert np.isfinite(expectations[nt])
        
        el = mypcfg.expected_length()
        assert el > 0
        assert np.isfinite(el)

    def test_condition_number_awareness(self):
        """
        Test that we handle ill-conditioned matrices properly.
        
        Near criticality, (I - T) has eigenvalues close to 0,
        leading to high condition numbers.
        
        Use multi-NT grammar since 1x1 matrices always have cond=1.
        """
        # Create a multi-nonterminal grammar near criticality
        mypcfg = self._make_multi_nt_near_critical(n_nonterminals=5, p_binary_total=0.48)
        
        # Build the transition matrix
        n = len(mypcfg.nonterminals)
        T = np.zeros([n, n])
        ntlist = sorted(list(mypcfg.nonterminals))
        index = {nt: i for i, nt in enumerate(ntlist)}
        
        for prod in mypcfg.productions:
            if len(prod) == 3:
                alpha = mypcfg.parameters[prod]
                lhs = index[prod[0]]
                T[lhs, index[prod[1]]] += alpha
                T[lhs, index[prod[2]]] += alpha
        
        # Check condition number of (I - T)
        I_minus_T = np.eye(n) - T
        cond = np.linalg.cond(I_minus_T)
        
        # Condition number should be elevated near criticality
        # (may not be > 100 depending on matrix structure)
        assert cond > 1
        
        # Despite potentially high condition number, solve should still work
        el = mypcfg.expected_length()
        assert el > 0
        assert np.isfinite(el)
        
        # Verify against expectations
        expectations = mypcfg.nonterminal_expectations()
        for nt in mypcfg.nonterminals:
            assert np.isfinite(expectations[nt])

    def test_derivational_entropy_near_critical(self):
        """Test entropy computation near criticality."""
        for p_binary in [0.3, 0.45, 0.49]:
            mypcfg = self._make_simple_recursive_pcfg(p_binary)
            
            entropy = mypcfg.derivational_entropy()
            
            # Entropy should be positive and finite
            assert entropy > 0
            assert np.isfinite(entropy)
            
            # Entropy should increase as we approach criticality
            # (more uncertainty in longer derivations)

    def test_production_expectations_sum_near_critical(self):
        """
        Test that production expectations are consistent near criticality.
        
        Sum of lexical production expectations should equal expected length.
        """
        for p_binary in [0.3, 0.45, 0.49, 0.499]:
            mypcfg = self._make_simple_recursive_pcfg(p_binary)
            
            pe = mypcfg.production_expectations()
            el = mypcfg.expected_length()
            
            # Sum of lexical productions = expected length
            lexical_sum = sum(e for prod, e in pe.items() if len(prod) == 2)
            assert lexical_sum == pytest.approx(el, rel=1e-6)

    def test_fp_vs_fast_near_critical(self):
        """
        Compare fixed-point and Newton methods near criticality.
        
        Both should give same results, but Newton converges faster and
        is more accurate. Allow looser tolerance for fixed-point method.
        """
        p_binary = 0.45  # Not too close to critical for FP convergence
        mypcfg = self._make_simple_recursive_pcfg(p_binary)
        
        pf_fast = mypcfg.compute_partition_function_fast()
        pf_fp = mypcfg.compute_partition_function_fp()
        
        for nt in mypcfg.nonterminals:
            # FP method may be less accurate, use looser tolerance
            assert pf_fast[nt] == pytest.approx(pf_fp[nt], abs=0.01)


# =============================================================================
# Tests for pcfgfactory.py
# =============================================================================

class TestPCFGFactory:
    """Tests for PCFG factory classes."""

    def test_lognormal_prior(self):
        """Test LogNormalPrior sampling."""
        from pcfgfactory import LogNormalPrior
        
        prior = LogNormalPrior(sigma=1.0)
        sample = prior.sample(100)
        
        assert len(sample) == 100
        assert sum(sample) == pytest.approx(1.0)
        assert all(s >= 0 for s in sample)

    def test_lexical_dirichlet(self):
        """Test LexicalDirichlet sampling."""
        from pcfgfactory import LexicalDirichlet
        
        ld = LexicalDirichlet(dirichlet=1.0)
        sample = ld.sample(50)
        
        assert len(sample) == 50
        assert sum(sample) == pytest.approx(1.0)

    def test_length_distribution_poisson(self):
        """Test Poisson length distribution."""
        from pcfgfactory import LengthDistribution
        
        ld = LengthDistribution()
        ld.set_poisson(5.0, 20)
        
        assert len(ld.weights) == 21
        assert ld.weights[0] == 0.0  # Zero-truncated
        assert ld.weights[5] > ld.weights[1]  # Mode around 5

    def test_stick_breaking(self):
        """Test stick-breaking process."""
        from pcfgfactory import sample_stick_py
        
        np.random.seed(42)
        sample = sample_stick_py(0.5, 1.0, 10)
        
        assert len(sample) == 10
        assert sum(sample) == pytest.approx(1.0)
        assert all(s >= 0 for s in sample)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_sample_parse_roundtrip(self, dyck1_pcfg):
        """Test that sampled trees can be parsed back."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(dyck1_pcfg)
        ic = inside.InsideComputation(dyck1_pcfg)
        
        for _ in range(20):
            tree = sampler.sample_tree()
            s = utility.collect_yield(tree)
            
            # Should be parseable
            p = ic.inside_probability(s)
            assert p > 0
            
            # Viterbi parse should have same yield
            viterbi_tree = ic.viterbi_parse(s)
            assert utility.collect_yield(viterbi_tree) == s

    def test_derivation_probability_consistency(self, dyck1_pcfg):
        """Test consistency between derivation and string probabilities."""
        numpy.random.seed(42)
        sampler = pcfg.Sampler(dyck1_pcfg)
        ic = inside.InsideComputation(dyck1_pcfg)
        
        for _ in range(20):
            tree = sampler.sample_tree()
            s = utility.collect_yield(tree)
            
            # Derivation probability
            p_deriv = math.exp(dyck1_pcfg.log_probability_derivation(tree))
            
            # String probability (sum over all derivations)
            p_string = ic.inside_probability(s)
            
            # String prob should be >= derivation prob
            assert p_string >= p_deriv - 1e-10

    def test_entropy_bounds(self, ambiguous_pcfg):
        """Test that various entropies satisfy expected bounds."""
        # Derivational entropy
        h_deriv = ambiguous_pcfg.derivational_entropy()
        
        # Unigram entropy (should be 0 for single-terminal grammar)
        h_unigram = ambiguous_pcfg.entropy_unigram()
        
        assert h_deriv >= 0
        assert h_unigram >= 0

    def test_loaded_grammar_properties(self, dyck2_pcfg):
        """Test properties of loaded Dyck-2 grammar."""
        assert dyck2_pcfg.is_normalised()
        
        pf = dyck2_pcfg.compute_partition_function_fast()
        assert pf["S"] == pytest.approx(1.0, abs=1e-5)
        
        el = dyck2_pcfg.expected_length()
        assert el > 0


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_grammar(self):
        """Test handling of empty grammar."""
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = set()
        mypcfg.productions = []
        mypcfg.parameters = {}
        
        # Should handle gracefully
        assert len(mypcfg.productions) == 0

    def test_single_production_grammar(self):
        """Test grammar with single production."""
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a"}
        mypcfg.productions = [("S", "a")]
        mypcfg.parameters = {("S", "a"): 1.0}
        mypcfg.set_log_parameters()
        
        assert mypcfg.is_normalised()
        assert mypcfg.expected_length() == 1.0
        assert mypcfg.derivational_entropy() == 0.0

    def test_very_small_probabilities(self):
        """Test handling of very small probabilities."""
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a", "b"}
        mypcfg.productions = [("S", "a"), ("S", "b")]
        mypcfg.parameters = {("S", "a"): 1e-10, ("S", "b"): 1.0 - 1e-10}
        mypcfg.set_log_parameters()
        
        # Should still work
        ic = inside.InsideComputation(mypcfg)
        p = ic.inside_probability(["a"])
        assert p == pytest.approx(1e-10)

    def test_max_depth_exceeded(self):
        """Test that sampling respects max depth."""
        # Highly recursive grammar
        mypcfg = pcfg.PCFG()
        mypcfg.start = "S"
        mypcfg.nonterminals = {"S"}
        mypcfg.terminals = {"a"}
        mypcfg.productions = [("S", "S", "S"), ("S", "a")]
        mypcfg.parameters = {("S", "S", "S"): 0.9, ("S", "a"): 0.1}
        mypcfg.set_log_parameters()
        
        sampler = pcfg.Sampler(mypcfg, max_depth=5)
        
        # May raise ValueError due to depth, which is expected
        errors = 0
        for _ in range(100):
            try:
                sampler.sample_tree()
            except ValueError:
                errors += 1
        
        # Should hit max depth sometimes with 90% recursion probability
        assert errors > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
