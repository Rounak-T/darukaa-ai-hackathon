import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph_reasoner import load_graph, trace_pathways
from core.retriever import get_retriever, load_cards
from core.schemas import Recommendation, SiteProfile, TimeHorizon
from core.slots import derive_conditions, readiness, regex_extract
from core.verifier import _is_generic


def test_knowledge_cards_wellformed():
    cards = load_cards()
    assert len(cards) >= 20
    required = {
        "id", "title", "domain", "intervention", "mechanism", "quantified_effect",
        "metrics_improved", "preconditions", "time_horizon", "source", "source_tier",
    }
    ids = set()
    for c in cards:
        assert required.issubset(c.keys()), f"{c.get('id')} missing fields"
        assert c["id"] not in ids, "duplicate card id"
        ids.add(c["id"])
        assert c["time_horizon"] in {"short", "medium", "long"}
        assert len(c["metrics_improved"]) >= 2


def test_causal_graph_integrity():
    g = load_graph()
    assert g.number_of_nodes() >= 10
    assert g.number_of_edges() >= 20
    for _, _, d in g.edges(data=True):
        assert d["sign"] in {"+", "-"}
        assert 0 < d["weight"] <= 1
        assert d["mechanism"]


def test_multi_hop_pathways_exist():
    paths = trace_pathways(["soil_organic_carbon"])
    assert paths, "expected at least one causal pathway"
    assert any(p["n_hops"] >= 2 for p in paths), "need genuine multi-hop chains"


def test_readiness_gate():
    thin = SiteProfile(observed_problem="biodiversity is declining")
    ready, missing = readiness(thin)
    assert not ready and len(missing) >= 3

    full = SiteProfile(
        soil_organic_carbon_pct=0.4, rainfall_pattern="low",
        land_use="monoculture", region_type="semi_arid",
    )
    ready2, _ = readiness(full)
    assert ready2


def test_regex_numeric_extraction():
    got = regex_extract("SOC is 0.42 % and pH 5.3, we get about 480 mm across 12 hectares")
    assert got["soil_organic_carbon_pct"] == 0.42
    assert got["soil_ph"] == 5.3
    assert got["rainfall_mm"] == 480.0
    assert got["area_hectares"] == 12.0


def test_hybrid_retrieval_returns_cards():
    r = get_retriever()
    hits = r.search("low soil organic carbon in semi-arid monoculture wheat")
    assert len(hits) == 5
    assert all("retrieval_score" in h for h in hits)


def test_generic_advice_is_rejected():
    bad = Recommendation(
        action="Use sustainable practices",
        scientific_reasoning="It is good for the environment.",
        metrics_improved=["soil_organic_carbon"],
        time_horizon=TimeHorizon.MEDIUM,
    )
    assert _is_generic(bad)

    good = Recommendation(
        action="Sow Sesbania cover crop in the post-harvest fallow window",
        scientific_reasoning=(
            "Biological nitrogen fixation plus high root turnover raises soil organic "
            "carbon, which increases plant-available water capacity, which extends the "
            "microbially active window into the dry season."
        ),
        metrics_improved=["soil_organic_carbon", "soil_moisture_retention", "microbial_diversity"],
        time_horizon=TimeHorizon.MEDIUM,
    )
    assert not _is_generic(good)