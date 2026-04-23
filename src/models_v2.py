from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin
from enum import Enum
import types
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


def typeinfo(annotation: Any) -> str:
    """Render a readable type name from a Pydantic FieldInfo annotation."""

    if annotation is None or annotation is type(None):  # noqa: E721
        return "null"
    if annotation is Any:
        return "any"

    # Primitive normalization (match requested output)
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "bool"
    if annotation is str:
        return "str"

    origin = get_origin(annotation)

    # `Annotated[T, ...]` -> unwrap to `T`
    if origin is getattr(types, "AnnotatedAlias", object()) or str(origin) == "typing.Annotated":
        args = get_args(annotation)
        return typeinfo(args[0]) if args else "any"
    if origin is getattr(__import__("typing"), "Annotated", object()):
        args = get_args(annotation)
        return typeinfo(args[0]) if args else "any"

    # `T | U` (py3.10+) and `Union[T, U]`
    if origin is Union or isinstance(annotation, types.UnionType):
        args = list(get_args(annotation))
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) != len(args):
            if len(non_none) == 1:
                return f"{typeinfo(non_none[0])}|exclude empty"
            return "|".join(typeinfo(a) for a in non_none) + "|exclude empty"
        return "|".join(typeinfo(a) for a in args)

    if origin in (list, List):
        (item_t,) = get_args(annotation) or (Any,)
        return f"list[{typeinfo(item_t)}]"

    if isinstance(annotation, type):
        # Normalize common typing-ish names while keeping unknowns readable
        return getattr(annotation, "__name__", str(annotation))

    # Fallback for uncommon typing constructs
    return str(annotation).replace("typing.", "")

# ────────────────────────────────────────────────
# Base class — updated version
# ────────────────────────────────────────────────
class Digest(BaseModel):
    # parsed data
    key_points: List[str] = Field(
        default_factory=list, 
        description=(
            "List of specified key events, facts, datapoints. "
            "Format=complete-sentences. Include=who,what,where,when,how,impact. Limit=3to8. "
            "Examples:\n"
            "- Hedge funds shifted large positions in Asian markets after selling in the US and Europe\n"
            "- US markets dropped 9.3% from all-time highs\n"
            "- Kia unveiled new EV4 and Concept EV2 models at EV Day 2025\n"
            "- Mozilla revised Firefox terms amid user backlash\n"
        )
    )        
    macro_driver: Optional[str] = Field(
        None,
        description=(
            "Primary geopolitical, trade, economic or technological driver of events. Short phrase."
            "Examples: "
            "us_iran_conflict, red_sea_disruption, tariff_volatility, "
            "rare_earth_controls, arctic_shipping_rivalry, lithium_supply_ban, "
            "africa_mineral_conflict, cyber_arms_race_escalation etc."
        ),
    )
    # "model_release, agent_launch, enterprise_adoption_case, safety_regulation_update, multimodal_breakthrough\n"
    # "ransomware_attack, zero_day_disclosure, supply_chain_breach, ai_enhanced_exploit, state_sponsored_campaign\n"
    # "chip_launch, platform_announcement, chip_shortage, foundry_partnership\n"
    # "humanoid_demo, warehouse_deployment, drone_swarm_test, regulation_change\n"
    # "series_a, acquisition_announced, merger_completed, strategic_partnership, ipo_filing\n"
    # "earnings_beat, stock_reaction, analyst_upgrade, sector_rotation, sec_filing_update\n"
    # "route_disruption, freight_rate_spike, aircraft_order, supply_chain_bottleneck, cyber_incident_on_cargo\n"
    # "oil_price_shock, gdp_forecast_revision, inflation_spike, rate_cut_signal, commodity_demand_shift\n"
    event_type: Optional[str] = Field(
        None,
        description=(
            "Primary aggregated event_type/resultant. "
            "Examples: "
            "model_release, agent_launch, regulation_update, multimodal_breakthrough, "
            "ransomware_attack, zero_day_disclosure, supply_chain_breach, "
            "chip_launch, platform_announcement, "
            "humanoid_demo, warehouse_deployment, "
            "series_a, acquisition_announced, ipo_filing, "
            "route_disruption, freight_rate_spike, supply_chain_bottleneck, "
            "oil_price_shock, gdp_forecast_revision, rate_cut_signal etc."
        ),
    )
    impact_level: Optional[str] = Field(
        None,
        description="Specified impact of the events on primary domain/context. "
        "ALLOWED: null, low, medium, high, critical, transformative",
    )
    cross_domain_impacts: List[str] = Field(
        default_factory=list,
        description=(
            "List of secondary domains and associated impacts. Format=[domain]:[1-sentence impact]. "
            "Examples:\n"
            "- cybersecurity: Increased risk of data breaches due to new vulnerabilities\n"
            "- aviation: Flight delays and cancellations due to air traffic control issues\n"
            "- hardware: Supply chain disruptions affecting chip production\n"
            "- startups: Emerging companies facing funding challenges\n"
        ),
    )    
    future_outlook: Optional[str] = Field(default=None, description="1-sentence specifying future outlook/trajectory.")
    headline: str = Field(description="1-sentence headline. Include=core_subjects,geographic_anchors,primary_stats,timescale,key_actor_driver,risk_reward. Length<=25words.")

    # search keywords
    regions: List[str] = Field(default_factory=list, description="List of specified geographic regions/locations. Exclude=grouped/aggregated qualifications - 30 countries.")
    people: List[str] = Field(default_factory=list, description="List of specified key people (ex: CEOs, political leaders). Exclude=generic,grouped/aggregated qualifications - 14 leaders.")
    companies: List[str] = Field(default_factory=list, description="List of specified companies/organizations. Exclude=generic,grouped/aggregated qualifications - 5 companies.")
    stock_tickers: List[str] = Field(default_factory=list, description="List of specified stock tickers.")
    
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "List of keywords usable for search,classification,clustering. "
            "Examples: agentic_ai, sovereign_compute, defense_tech etc.",
        )
    )

    @classmethod
    def schema(cls):
        return "\n".join(
            f"{fname}: {typeinfo(finfo.annotation)}|{finfo.description}"
            for fname, finfo in cls.model_fields.items()
        )

    def __str__(self):
        return self.model_dump_json()

# ────────────────────────────────────────────────
# Domain-specific models (inherit from base)
# ────────────────────────────────────────────────


class AINewsDigest(Digest):
    models_or_agents: List[str] = Field(default_factory=list, description="List of names/codenames of AI models, agents or frameworks mentioned (e.g. 'Grok-4', 'o3-mini', 'Perplexity PC agent')")
    researchers: List[str] = Field(default_factory=list, description="List of names of key researchers, authors or quoted experts")
    benchmark_scores: List[str] = Field(default_factory=list, description="List of reported performance numbers on standard benchmarks (key: benchmark name, value: score)")
    claimed_productivity_lift_pct: Optional[str] = Field(None, description="Reported % productivity/efficiency gain claimed for users or enterprises")
    enterprise_adoption_rate_pct: Optional[str] = Field(None, description="Reported % companies (especially Fortune 500) already using or piloting the technology")
    pricing_monthly_usd: Optional[str] = Field(None, description="Reported monthly subscription price in USD.")
    valuation_or_market_size: Optional[str] = Field(None, description="Company valuation or projected market size in USD")


class CyberNewsDigest(Digest):
    # malware information
    threat_actors: List[str] = Field(default_factory=list, description="List of named or categorized attackers. Examples: LockBit, nation state, etc.",)
    vulnerabilities: List[str] = Field(default_factory=list, description="List of CVE IDs, product names or zero-day descriptions mentioned")
    malware_family: Optional[str] = Field(None, description="Name of the malware family or ransomware strain if applicable")
    attack_speed: Optional[str] = Field(None)
    incident_type: Optional[str] = Field(None, description="Primary category of the cybersecurity event. Allowed: ransomware, supply_chain, zero_day, ai_enhanced, state_sponsored, shadow_ai, critical_infra etc.")
    # impact information
    affected_entities: List[str] = Field(default_factory=list, description="List of organizations, sectors or number of users impacted")
    records_breached: Optional[str] = Field(None)
    financial_impact: Optional[str] = Field(None, description="Specified financial damage or cost of recovery in USD.")    
    business_impact: Optional[str] = Field(None, description="Specified operational, financial, reputational or regulatory consequences")
    # remediation
    recommended_actions: List[str] = Field(default_factory=list, description="List of specified defensive or mitigation steps" )
    compliance_triggers: List[str] = Field(default_factory=list, description="Policies,standards,laws,regulations mentioned as being triggered or relevant (SEC, CIRCIA, GDPR, etc.)")


class HardwareNewsDigest(Digest):
    """Summary focused on chips, accelerators, compute infrastructure"""

    products: List[str] = Field(default_factory=list, description="List of specified products/chips (e.g. 'H100', 'NVIDIA GH200')")
    manufacturer: str = Field()
    use_cases: List[str] = Field(default_factory=list, description="List of intended/demonstrated uses. Examples: AI training, inference, HPC, edge computing. Limit=5",)
    performance_improvement_x: Optional[str] = Field(
        None, description="Speedup factor compared to previous generation. Include unit in value (e.g. '2.8x')."
    )
    power_efficiency_gain_pct: Optional[str] = Field(
        None, description="Performance-per-watt improvement. Include unit in value (e.g. '15%')."
    )
    capex_investment_usd_billions: Optional[str] = Field(
        None,
        description="CapEx for data centers/fabs. Include unit in value (e.g. '$2.5 billion'). Only if >$1B announced.",
    )
    price_per_unit_usd: Optional[str] = Field(
        None, description="Estimated or announced price per chip/unit. Include currency in value (e.g. '$10,000')."
    )


class RoboticsAVDronesNewsSummary(Digest):
    """Summary focused on robotics systems, autonomous vehicles, drones"""

    product_system: str = Field(
        ..., description="Name/model of robot/AV/drone (e.g. 'Tesla Cybercab', 'Boston Dynamics Stretch')"
    )
    manufacturer: str = Field(
        ..., description="Company. Format: official name. If startup: include founding year."
    )
    category: Optional[str] = Field(
        None,
        description="Broad category of the embodied system. Allowed: industrial_cobot, humanoid, autonomous_vehicle, drone_swarm, warehouse_agv.",
    )
    speed_or_payload_improvement: Optional[str] = Field(
        None,
        description="Key upgrade vs prior gen. Include unit in value. Examples: '2.5 m/s', '50% faster'.",
    )
    deployment_sites_count: Optional[str] = Field(
        None,
        description="Real-world deployments/customers mentioned. Include count in value (e.g. '12 sites').",
    )
    funding_raised_usd_millions: Optional[str] = Field(
        None,
        description="Funding amount raised. Include currency and magnitude in value (e.g. '$50 million').",
    )
    cost_per_unit_usd: Optional[str] = Field(
        None, description="Estimated cost per robot / vehicle. Include currency in value (e.g. '$250,000')."
    )
    real_world_limitation_noted: List[str] = Field(
        default_factory=list,
        description="Practical limitations noted. Examples: weather sensitivity, edge cases, cost barriers. Exclude speculation. Limit: 5.",
    )


class StartupCorpNewsSummary(Digest):
    """Summary focused on startups, corporate moves, funding, M&A"""

    main_company: str = Field(..., description="Primary company (official name). No qualifiers.")
    other_companies: List[str] = Field(
        default_factory=list,
        description="Co-parties: acquirers, investors, partners. Exclude=unrelated mentions. Limit: 5.",
    )
    lead_investors: List[str] = Field(
        default_factory=list, description="Lead/prominent investors (official names/fund names). Exclude=passive stakeholders. Limit: 5."
    )
    acquirer: Optional[str] = Field(
        None, description="Name of the acquiring company in M&A deals"
    )
    funding_amount_usd_millions: Optional[str] = Field(
        None, description="Amount raised in the funding round. Include currency in value (e.g. '$25 million')."
    )
    round_type: Optional[str] = Field(
        None, description="Stage of funding (Seed, Series A, Late, Debt, etc.)"
    )
    pre_post_valuation_usd_billions: Optional[str] = Field(
        None, description="Pre-money and/or post-money valuation. Include currency in value (e.g. 'pre: $500M, post: $1B')."
    )
    deal_value_usd_millions: Optional[str] = Field(
        None, description="Transaction value in M&A deals. Include currency in value (e.g. '$500 million')."
    )
    yoy_funding_growth_pct: Optional[str] = Field(
        None,
        description="Year-over-year change in funding volume. Include unit in value (e.g. '+15%').",
    )
    strategic_rationale: str = Field(
        ...,
        description="1-sentence rationale. Format: [Actor] seeks [capability/market] via [deal type].",
    )
    use_of_funds: Optional[str] = Field(
        None, description="Stated use of capital. Categories: R&D, expansion, acquisition, operations, debt repayment. Be explicit."
    )


class FinancialMarketsNewsSummary(Digest):
    """Summary focused on stocks, earnings, filings, market movements"""

    ticker_or_index: str = Field(
        ..., description="Primary ticker(s) or index (format: 'AAPL' or 'S&P500'). Limit: 3 tickers max."
    )
    companies_mentioned: List[str] = Field(
        default_factory=list,
        description="Companies discussed. Format: official ticker+name. Exclude=tangential mentions. Limit: 5.",
    )
    stock_reaction_pct: Optional[str] = Field(
        None,
        description="Stock price change after news. Include unit in value (e.g. '+2.5%' or '-1.3%').",
    )
    earnings_beat_miss_pct: Optional[str] = Field(
        None, description="EPS/revenue beat or miss vs consensus. Include unit and direction in value (e.g. '+3% beat' or '-2% miss')."
    )
    revenue_or_ebitda_usd_millions: Optional[str] = Field(
        None,
        description="Reported or forecasted revenue / EBITDA. Include currency and magnitude in value (e.g. '$1,200 million' or '$1.2B').",
    )
    forward_guidance_change_pct: Optional[str] = Field(
        None, description="FY guidance change vs prior. Include unit in value (e.g. '+5%' if raised, '-10%' if lowered, or 'reaffirmed')."
    )
    valuation_multiple: Optional[str] = Field(
        None,
        description="Forward-looking valuation metric (e.g. '32x revenue', '18x EBITDA')",
    )
    financial_analysis_summary: str = Field(
        ...,
        description="1-sentence summary. Format: [Company] [beat/miss] due to [reason], implying [outlook].",
    )
    sector_rotation_signal: str = Field(
        default="",
        description="Money flow signal. Format: 'from [sector] to [sector]' (e.g. 'from Mag7 to cyclicals').",
    )


class LogisticsDigest(Digest):
    transportation_mode: str = Field(description="Allowed: N/A, air, ocean, truck, multimodal")
    affected_routes: List[str] = Field(default_factory=list, description="Examples: Red Sea, Suez, Transpacific. Limit=5")
    freight_rate_change: Optional[str] = Field(None, description="Include=value,unit,context. Example: +8% in 2 years")
    order_quantity: Optional[str] = Field(None, description="Include=value,unit. Example: 30 tons")
    shipping_delay: Optional[str] = Field(None, description="Include=value,unit,context. Example: 5 days for Transpacific route.")
    financial_impact: Optional[str] = Field(None, description="Specified financial damage or cost of recovery in USD.")    
    business_impact: Optional[str] = Field(None, description="Specified operational, reputational or regulatory consequences")
    mitigations: List[str] = Field(
        default_factory=list,
        description="List of specified mitigations. Examples: rerouting, inventory buildup, alternative suppliers. Limit=5.",
    )


class MacroEconomyDigest(Digest):
    """Summary focused on global economy, macro indicators, forecasts"""

    gdp_growth_forecast: Optional[str] = Field(None, description="GDP growth forecast. Include=value,unit,timeframe.")
    inflation_impact: Optional[str] = Field(None, description="Inflation impact. Include=value,unit,timeframe.")
    oil_price: Optional[str] = Field(None, description="Oil price scenario. Include=value,unit,timeframe.")
    gold_demand: Optional[str] = Field(None, description="Gold demand. Include=value,unit,timeframe.")
    macro_signal: str = Field(...,description="Format: [Event] signals [risk/opportunity] for [sector/macro].",)
    market_significance: str = Field(description="Format: affects [equities/credit/funding] via [mechanism].",)


# ────────────────────────────────────────────────
# New shared financial core – extracted from overlapping fields
# ────────────────────────────────────────────────
class FinancialCoreMetrics(BaseModel):
    """Reusable core quantitative financial metrics common to earnings releases and SEC filings"""

    revenue_usd_millions: Optional[float] = Field(
        None, description="Total revenue reported (millions USD)"
    )
    revenue_growth_yoy_pct: Optional[float] = Field(
        None, description="Year-over-year revenue growth percentage"
    )
    net_income_usd_millions: Optional[float] = Field(
        None,
        description="Net income attributable to common shareholders (millions USD)",
    )
    eps_basic: Optional[float] = Field(
        None, description="Basic earnings per share (GAAP)"
    )
    eps_diluted: Optional[float] = Field(None, description="Diluted EPS (GAAP)")
    operating_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Net cash provided by operating activities (millions USD)"
    )
    capex_usd_millions: Optional[float] = Field(
        None, description="Capital expenditures (millions USD)"
    )
    cash_equivalents_usd_millions: Optional[float] = Field(
        None,
        description="Cash, cash equivalents & short-term investments at period end",
    )
    total_debt_usd_millions: Optional[float] = Field(
        None, description="Total short + long-term debt"
    )
    key_financial_ratios: Dict[str, str] = Field(
        default_factory=dict,
        description="Important ratios (e.g. {'gross_margin_pct': '42.1', 'operating_margin_pct': '18.7', 'net_debt_to_ebitda': '1.8x'})",
    )


# ────────────────────────────────────────────────
# Derived: Earnings Report Summary
# ────────────────────────────────────────────────
class EarningsReportSummary(Digest):
    """Summary for earnings press releases, call transcripts, and related materials"""

    fiscal_period: str = Field(
        ..., description="Reporting period (e.g. 'Q4 2025', 'FY 2025')"
    )
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")

    # Compose shared financial block
    financials: FinancialCoreMetrics = Field(
        ..., description="Core quantitative financial results"
    )

    # Earnings-specific extensions
    revenue_beat_miss_pct: Optional[float] = Field(
        None, description="Revenue beat/miss vs consensus (%)"
    )
    eps_beat_miss_pct: Optional[float] = Field(
        None, description="EPS beat/miss vs consensus (%)"
    )
    eps_adjusted: Optional[float] = Field(None, description="Non-GAAP / adjusted EPS")
    gross_margin_pct: Optional[float] = Field(
        None, description="Gross margin percentage"
    )
    operating_margin_pct: Optional[float] = Field(
        None, description="Operating margin percentage"
    )
    free_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Free cash flow (operating CF – capex)"
    )

    next_quarter_guidance: Dict[str, str] = Field(
        default_factory=dict,
        description="Key next-quarter guidance ranges or narrative",
    )
    full_year_guidance_update: Optional[str] = Field(
        None, description="Full-year outlook change (raised/lowered/reaffirmed)"
    )
    guidance_tone: Literal["bullish", "cautious", "neutral", "mixed"] = Field("neutral")

    key_segments_performance: Dict[str, str] = Field(
        default_factory=dict, description="Segment/product/region highlights"
    )
    main_drivers: List[str] = Field(
        default_factory=list, description="Primary drivers of results"
    )
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Key MD&A paraphrases"
    )
    strategic_priorities: List[str] = Field(
        default_factory=list, description="Strategic / capital allocation themes"
    )
    risks_updated: List[str] = Field(
        default_factory=list, description="Updated or emphasized risks"
    )
    one_time_items: List[str] = Field(
        default_factory=list, description="Significant non-recurring items"
    )

    management_tone: Literal[
        "confident", "cautious", "defensive", "optimistic", "mixed"
    ] = Field("neutral")
    qna_hot_topics: List[str] = Field(
        default_factory=list, description="Most discussed Q&A themes"
    )
    call_sentiment_score: Optional[float] = Field(
        None, description="Optional numeric sentiment (-1.0 to +1.0)"
    )


# ────────────────────────────────────────────────
# Derived: SEC Filing Summary
# ────────────────────────────────────────────────
class SECFilingSummary(Digest):
    """Summary for SEC filings (10-K, 10-Q, 8-K, etc.)"""

    filing_type: Literal["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K"] = Field(
        ..., description="SEC form type"
    )
    accession_number: str = Field(..., description="EDGAR accession number")
    filing_date: date = Field(..., description="Date filed with SEC")
    period_end_date: date = Field(..., description="End of reporting period")
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name as filer")

    # Compose shared financial block
    financials: FinancialCoreMetrics = Field(
        ..., description="Core quantitative financial results from statements"
    )

    # SEC-specific extensions
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Most material MD&A points"
    )
    material_trends_uncertainties: List[str] = Field(
        default_factory=list,
        description="Trends/uncertainties likely to impact future results",
    )
    critical_accounting_estimates: List[str] = Field(
        default_factory=list,
        description="Significant accounting judgments/sensitivities",
    )
    top_risk_factors: List[str] = Field(
        default_factory=list, description="Top / newly emphasized risk factors"
    )
    legal_proceedings_status: List[str] = Field(
        default_factory=list, description="Material litigation / contingencies summary"
    )
    business_overview_highlights: List[str] = Field(
        default_factory=list, description="Key Item 1 Business points"
    )
    strategic_initiatives: List[str] = Field(
        default_factory=list, description="Major strategic priorities / shifts"
    )
    key_exhibits_filed: List[str] = Field(
        default_factory=list,
        description="Notable exhibits (e.g. insider policy, material contracts)",
    )
    material_events_8k: Optional[str] = Field(
        None, description="For 8-K: description of triggering event"
    )


# ────────────────────────────────────────────────
# Merged model: FinancialDocumentSummary
# ────────────────────────────────────────────────
class FinancialDocumentSummary(Digest):
    """
    Unified model covering earnings releases, call transcripts, 10-K, 10-Q, 8-K and combinations.
    Use 'document_subtype' to distinguish primary focus.
    """

    document_subtype: Literal[
        "earnings_release",  # Press release only
        "earnings_call_transcript",  # Transcript / call summary
        "earnings_package",  # Release + call + slides
        "10-K",  # Annual report
        "10-Q",  # Quarterly report
        "8-K",  # Current report / material event
        "hybrid_earnings_filing",  # e.g. 10-Q + earnings release on same day
    ] = Field(..., description="Primary nature / most important part of the document")

    fiscal_period: str = Field(
        ..., description="Reporting period e.g. 'Q4 2025', 'FY 2025'"
    )
    ticker: str = Field(..., description="Stock ticker")
    company_name: str = Field(..., description="Company name as reported")
    period_end_date: date = Field(..., description="End date of the fiscal period")
    filing_date: Optional[date] = Field(
        None, description="SEC filing/submission date (if applicable)"
    )

    # Core financials (shared)
    financials: FinancialCoreMetrics = Field(..., description="Quantitative backbone")

    # Earnings-specific extensions (optional – often null for pure 10-K/8-K)
    revenue_beat_miss_pct: Optional[float] = Field(
        None, description="Revenue vs consensus (%)"
    )
    eps_beat_miss_pct: Optional[float] = Field(None, description="EPS vs consensus (%)")
    eps_adjusted: Optional[float] = Field(None, description="Non-GAAP/adjusted EPS")
    gross_margin_pct: Optional[float] = Field(None, description="Gross margin %")
    operating_margin_pct: Optional[float] = Field(
        None, description="Operating margin %"
    )
    free_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Free cash flow"
    )

    next_quarter_guidance: Dict[str, str] = Field(
        default_factory=dict, description="Next-quarter ranges or narrative"
    )
    full_year_guidance_update: Optional[str] = Field(
        None, description="Full-year outlook change"
    )
    guidance_tone: Optional[Literal["bullish", "cautious", "neutral", "mixed"]] = Field(
        None
    )

    # Call-specific (optional – usually null for pure filings)
    management_tone: Optional[
        Literal["confident", "cautious", "defensive", "optimistic", "mixed"]
    ] = Field(None)
    qna_hot_topics: List[str] = Field(
        default_factory=list, description="Recurring or heated Q&A themes"
    )
    call_sentiment_score: Optional[float] = Field(
        None, description="Numeric sentiment if analyzed (-1.0 to +1.0)"
    )

    # MD&A / Narrative – shared but emphasized differently
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Most important MD&A points / explanations"
    )
    main_drivers: List[str] = Field(
        default_factory=list,
        description="Primary drivers of results (volume, price, costs, FX…)",
    )

    # Risk & Legal – more prominent in filings
    top_risk_factors: List[str] = Field(
        default_factory=list, description="Key / newly updated risks"
    )
    legal_proceedings_status: List[str] = Field(
        default_factory=list, description="Material litigation or contingencies"
    )

    # Business / Strategy – stronger in 10-K
    business_overview_highlights: List[str] = Field(
        default_factory=list, description="Core business description points"
    )
    strategic_initiatives: List[str] = Field(
        default_factory=list, description="Strategic priorities, investments, shifts"
    )

    # Other material items
    one_time_items: List[str] = Field(
        default_factory=list, description="Non-recurring charges/gains"
    )
    material_trends_uncertainties: List[str] = Field(
        default_factory=list,
        description="Trends likely to materially affect future results",
    )
    key_exhibits_filed: List[str] = Field(
        default_factory=list, description="Notable exhibits (policies, contracts…)"
    )

    # 8-K specific
    material_event_description: Optional[str] = Field(
        None, description="For 8-K: what triggered the filing"
    )

