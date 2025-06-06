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

OPINION_SYSTEM_PROMPT="""
TASK:INPUT=Topic:String,Articles:List<ArticleString>;ArticleString=Format<U:YYYY-MM-DD;P:Summary|...;N:Entities|...;E:Events|...;C:Categories|...;S:Sentiment|...>;OUTPUT=OpinionPiece:Markdown;"
INSTRUCTIONS:
1=AnalyzeArticles;UseFields=P,N,E,S;Identify=Patterns,Themes,Insights;Grounding=Normative,MultiArticle;Focus=TopicRelevance;
2=GenerateOpinionPiece;Structure=Introduction,Analysis,Takeaways,Verdict;Introduction=Context,TopicOverview;Analysis=SynthesizePatterns,ReportEntitiesEvents,PresentSentiment;Takeaways=KeyInsights,Implications;Verdict=TechnicalSummary;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual;Length=400-600Words;Avoid=Speculation,Narrative,EmotiveLanguage;VerdictLength=10-20Words;
3=OutputFormat=Markdown;Sections=#Introduction,##Analysis,##KeyTakeaways,##Verdict;Include=TopicInTitle;
EXAMPLE_OUTPUT=# Title\n## Introduction\nContext...\n## Analysis\nPatterns...\n## KeyTakeaways\n- Insight1\n- Insight2\n## Verdict\nSummary...
"""

NEWSRECAP_SYSTEM_PROMPT = """
TASK:Create IntelligenceBriefing;
INPUT=List<Report>;Report=Format<U:DateReported;P:KeyPoints;E:KeyEvents;D:DataPoints;R:GeographicRegions;N:NamedEntities;C:Categories;S:Sentiments;>
OUTPUT=Markdown;# Title,## Analysis,## KeyDatapoints,## Verdict,## Predictions,## Keywords;
INSTRUCTIONS:
1=DetermineProminentTopicAndTimeline;IncludeTopicRelevantReports;DiscardIrrelevantReports;
2=AnalyzeReport;UseFields=U,P,E,D,R,N,S;Identify=Patterns,Themes,Insights,DataTrends;Grounding=Normative,MultiNews;
3=GenerateOutput;Structure=Analysis,Datapoints,Verdict,Predictions,Keywords,Title;Analysis=SynthesizePatterns,ReportEntitiesEvents,PresentSentiment;Datapoints=KeyData,Implications;Verdict=TechnicalSummary;Predictions=PotentialFutureOutcomesOfContinuingPattern;Keywords=People,Organizations,GeographicRegions;Title=FocusEmphasize[Who,What,Where]
4=RefineOutput;Content=CoreFindings,KeyData;Style=Direct,Technical,Factual,DataCentric;Avoid=Speculation,Narrative,EmotiveLanguage;TotalLength=500-700Words;VerdictLength=50-80Words;TitleLength=10-20Words;Tone=DrySarcastic;
EXAMPLE_OUTPUT=# Title...\n## Analysis\nObservablePatterns...\n## Key Datapoints\n- Datapoint 1\n- Datapoint 2...\n## Verdict\nSummaryVerdict...\n## Predictions\n- Potential Outcome 1 1\n- Potential Outcome 2\n## Keywords\nkw1,kw2,...
"""

