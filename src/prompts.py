DIGEST_SYSTEM_PROMPT = """
TASK:Create compressed-digest,token-efficient,lossless-format;
INPUT=news/blog;
OUTPUT=P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;
INSTRUCTIONS:
1=Extract:KeyPoints,KeyEvents,DataPoints,GeographicRegions,NamedEntities[Person,Company,Organization,Product];
2=Retain all data for >98% recovery;
OUTPUT_FORMAT:
1=Semicolon-separated keyvalue pairs with single-letter prefixes for each section (P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities);
2=Pipe-separate values within sections (P:KeyPoint1|KeyPoint2);
3=Skip empty/null values;
4=Avoid JSON nesting;
EXAMPLE_OUTPUT=P:KeyPoint1|KeyPoint2;E:Event1|Event2;D:Data1|Data2;R:Country|City|Continent;N:Person|Company|Product;
"""

TOPICS_SYSTEM_PROMPT="""
TASK:
INPUT=Domain:String,Articles:List<ArticleString>;ArticleString=Format<U:YYYY-MM-DD;P:Summary|...;E:Events|...;D:Datapoints|...;R:Regions|...;N:Entities|...;C:Categories|...;S:Sentiment|...>
OUTPUT=Dict<TopicTitle,Dict<frequency:Int,keywords:List<String>>>:JSON
INSTRUCTIONS:
1=AnalyzeArticles;UseFields=U,P,E,D,N;GenerateTopics=Dynamic,Specific,Granular;Cluster=SemanticSimilarity;Avoid=GenericCategoriesFromC;AllowMultiTagging=True
2=CountFrequency;Frequency=NumArticlesPerTopic
3=FilterFrequency=Min2;KeepTopics=Frequency>=2
4=GenerateKeywords;Keywords=Specific,Searchable;From=N,R;MinimizeFalsePositives=True;Include=Entities,Phrases
5=OutputFormat=Dict;Key=TopicTitle;Value=Dict;ValueFormat=frequency:Int,keywords:List<String>
EXAMPLE_OUTPUT={"TopicTitle1":{"frequency":4,"keywords":["kw1","kw2"]},"TopicTitle2":{"frequency":2,"keywords":["kw3","kw4"]}}
"""

# OPINION_SYSTEM_PROMPT="""
# TASK:INPUT=Topic:String,Articles:List<ArticleString>;ArticleString=Format<U:YYYY-MM-DD;P:Summary|...;N:Entities|...;E:Events|...;C:Categories|...;S:Sentiment|...>;OUTPUT=OpinionPiece:Markdown;"
# INSTRUCTIONS:
# 1=AnalyzeArticles;UseFields=P,N,E,S;Identify=Patterns,Themes,Insights;Grounding=Normative,MultiArticle;Focus=TopicRelevance;
# 2=GenerateOpinionPiece;Structure=Introduction,Analysis,Takeaways,Verdict;Introduction=Context,TopicOverview;Analysis=SynthesizePatterns,ReportEntitiesEvents,PresentSentiment;Takeaways=KeyInsights,Implications;Verdict=TechnicalSummary;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual;Length=400-600Words;Avoid=Speculation,Narrative,EmotiveLanguage;VerdictLength=10-20Words;
# 3=OutputFormat=Markdown;Sections=#Introduction,##Analysis,##KeyTakeaways,##Verdict;Include=TopicInTitle;
# EXAMPLE_OUTPUT=# Title\n## Introduction\nContext...\n## Analysis\nPatterns...\n## KeyTakeaways\n- Insight1\n- Insight2\n## Verdict\nSummary...
# """

OPINION_SYSTEM_PROMPT = """
TASK:WriteOpinionPiece;
INPUT=Topic:String\n\nList<Datastream>;Datastream=Format<U:DateReported;P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;C:Categories;S:Sentiments;>
OUTPUT=Analysis,Takeaways,Verdict,Title,Keywords;
STEPS:
1.AnalyzeDatastreams;UseFields=P,E,D,R,N;Identify=Patterns,Themes,Insights;Sentiments;Grounding=Normative,MultiNews;Focus=TopicRelevance;
2.GenerateOpinionPiece=Analysis,Takeaways,Verdict;Analysis=SynthesizePatterns,ReportEntitiesEvents;Takeaways=KeyInsights,Implications;Verdict=TechnicalSummary;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual;Avoid=Speculation,Narrative,EmotiveLanguage;
3.GenerateSynthesis=Title,Keywords;Title=Highlight[Who,Action,What,Object,Where];Keywords=People,Organizations,GeographicRegions;
4.RefineOutput;TotalLength=500-700Words;VerdictLength=50-80Words;TitleLength=10-20Words;Keywords=CommaSeparated;
EXAMPLE_OUTPUT=## Title\nWhoDidWhatToWhomInWhere...\n## Analysis\nPatterns...\n## Takeaways\n- Insight1\n- Insight2...\n## Verdict\nSummary...\n## Keywords\nkw1,kw2,...
"""

NEWSRECAP_SYSTEM_PROMPT = """
TASK:WriteIntelligenceBriefing;
INPUT=Topic:String\n\nList<Datastream>;Datastream=Format<U:DateReported;P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;C:Categories;S:Sentiments;>
OUTPUT=Analysis,Datapoints,Predictions,Verdict,Title,Keywords;
STEPS:
1.AnalyzeDatastreams;UseFields=P,E,D,R,N;Identify=Patterns,Themes,Insights,DataTrends;Sentiments;Grounding=Normative,MultiNews;Focus=TopicRelevance;
2.GenerateIntelligenceBriefing=Analysis,Datapoints,Predictions;Analysis=SynthesizePatterns,ReportEntitiesEvents,SentimentTrend;Datapoints=KeyData,EmergingTrends,Implications;Predictions=PotentialFutureOutcomesOfContinuingPattern;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual,DataCentric;Avoid=Speculation,Narrative,EmotiveLanguage;Tone=DrySarcastic;
3.GenerateSynthesis=Verdict,Title,Keywords;Verdict=TechnicalSummary;Title=Highlight[Who,Action,What,Object,Where];Keywords=People,Organizations,GeographicRegions;
4.RefineOutput;TotalLength=500-700Words;VerdictLength=50-80Words;TitleLength=10-20Words;Keywords=CommaSeparated;
EXAMPLE_OUTPUT=## Title\nWhoDidWhatToWhomInWhere...\n## Analysis\nObservablePatterns...\n## Datapoints\n- KeyData 1\n- KeyData 2...\n## Verdict\nSummaryVerdict...\n## Predictions\n- Potential Outcome 1 1\n- Potential Outcome 2\n## Keywords\nkw1,kw2,...
"""

OPINION_SYSTEM_PROMPT_JSON = """
TASK:WriteOpinionPiece;
INPUT=Topic:String\n\nList<Datastream>;Datastream=Format<U:DateReported;P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;C:Categories;S:Sentiments;>
OUTPUT_FORMAT=JSON;
{
    "title": string,
    "analysis": list<string>,
    "takeaways": list<string>,
    "verdict": string,
    "keywords": list<string>
}
STEPS:
1.AnalyzeDatastreams;UseFields=P,E,D,R,N;Identify=Patterns,Themes,Insights;Sentiments;Grounding=Normative,MultiNews;Focus=TopicRelevance;
2.GenerateOpinionPiece=Analysis,Takeaways,Verdict;Analysis=SynthesizePatterns,ReportEntitiesEvents;Takeaways=KeyInsights,Implications;Verdict=TechnicalSummary;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual;Avoid=Speculation,Narrative,EmotiveLanguage;AnalysisLength=300-400Words;TakawaysLength=100-200Words;VerdictLength=50-70Words;
3.GenerateSynthesis=Title,Keywords;Title=Highlight[Who,Action,What,Object,Where];TitleLength=10-20Words;Keywords=People,Organizations,GeographicRegions;
EXAMPLE_OUTPUT:
{
    "title": "Title",
    "analysis": ["Analysis", "Analysis2"],
    "takeaways": ["Insight1", "Insight2"],
    "verdict": "Summary",
    "keywords": ["kw1", "kw2"]
}
"""

NEWSRECAP_SYSTEM_PROMPT_JSON = """
TASK:WriteIntelligenceBriefing;
INPUT=Topic:String\n\nList<Datastream>;Datastream=Format<U:DateReported;P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;C:Categories;S:Sentiments;>
OUTPUT_FORMAT=JSON;
{
    "title": string,
    "analysis": list<string>,
    "datapoints": list<string>,
    "verdict": string,
    "predictions": list<string>,
    "keywords": list<string>
}
STEPS:
1.AnalyzeDatastreams;UseFields=P,E,D,R,N;Identify=Patterns,Themes,Insights,DataTrends;Sentiments;Grounding=Normative,MultiNews;Focus=TopicRelevance;
2.GenerateIntelligenceBriefing=Analysis,Datapoints,Predictions;Analysis=SynthesizePatterns,ReportEntitiesEvents,PresentSentiment;Datapoints=KeyData,EmergingTrends,Implications;Predictions=PotentialFutureOutcomesOfContinuingPattern;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual,DataCentric;Avoid=Speculation,Narrative,EmotiveLanguage;Tone=DrySarcastic;AnalysisLength=300-400Words;DatapointsLength=100-200Words;
3.GenerateSynthesis=Verdict,Title,Keywords;Verdict=TechnicalSummary;Title=Highlight[Who,Action,What,Object,Where];VerdictLength=50-70Words;TitleLength=10-20Words;Keywords=People,Organizations,GeographicRegions;
EXAMPLE_OUTPUT:
{
    "title": "Title",
    "analysis": ["Analysis", "Analysis2"],
    "datapoints": ["Datapoint1", "Datapoint2"],
    "verdict": "SummaryVerdict",
    "predictions": ["PotentialOutcome1", "PotentialOutcome2"],
    "keywords": ["kw1", "kw2"]
}
"""

BANNER_IMAGE_SYSTEM_PROMPT = """
STEP 1. Generate a banner image for a headline news article based on: {user_input}
STEP 2. Refine step 1 by removing all texts
"""