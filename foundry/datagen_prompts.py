SYSTEM_INST = """
INIT_LLM[MODE:CLAIM_EXTRACTION_ONLY, FILTER:RHETORIC_REMOVAL, STYLE:DRY, OUTPUT:PROSE_SUMMARY]  
RULES:  
- REMOVE[Pathos, Loaded Language, False Equivalence, Straw Man, Slippery Slope, Appeal to Fear, Appeal to Envy, Weasel Words, Ad Hominem, Historical Revisionism, Bandwagon Effect, Persuasive Questions, Moral High Ground, Overgeneralization, Simplification, Framing, Hyperbole]  
- EXTRACT[Explicit Facts, Directly Attributable Policy Proposals, Verifiable Economic Data, Official Statements, Confirmed Legislative Actions, Numerical Statistics with Sources]  
- SUMMARIZE[Unique Claims Only, No Redundant Assertions, No Unverified Assumptions, No Speculation]  
- FORMAT[Sequential Sentences, Single Dense Paragraph, No Headers, No Lists, Compressed Language]  
- OUTPUT[Concise Prose Summary]  
ACTIVATE
"""

SUMMARY_INST = """
TASK: Your input is a scrapped news, blog or social media post. Rewrite it to make it a concise intelligence briefing based the relevant contents.

STEP 1: Analyze the input and disregard the irrelevant contents
  - The input is a scrapped webpage and hence may contain text chunks that are irrelevant such as: Pay-wall message, advertisements, related articles, website biolerplate
  - Remove the irrelevant contents from further processing.

STEP 2: Extract the relevant contents presented by the core of the article
  - Extract the events and actions that have happened or will take place
  - Extract the key data points that influence the actions, events, outcomes and questions
  - Extract the key takeways presented by the article
  - Extract the key entities such as person, organization, company, product, stock ticker, phenomenon mentioned in the article
  - Extract the geographic location of the key events if mentioned.
  - Extract the domain/genre of the content subject matter such as: Government & Politics, Business & Finance, Cybersecurity, Crypto & Blockchain, Software Engineering, DevOps, Artificial Intelligence, IoT and Gadgets, Physics, Chemistry, Economy, Environment, Clean Energy, Health, Arts & Culture, Startups & Entrepreneurship, Entertainment, Sports etc.

STEP 3: Formulate your response
  - Include a one-line gist using less than 16 words 
    - The gist should be formatted as [Who] [Action] [What] [Where - optional]
      - Who - the main entity or person involved.
      - Action - What has happened or what is happening or what is being done.
      - What - The core object of the article or the impacted entity
      - Where - the action took place (if mentioned).
    - Examples of gist
      - Example 1: NASA Launches Mars Mission in 2025 to Study Planet’s Atmosphere.
      - Example 2: Congress Debates Healthcare Reform This Week to Address Rising Costs
      - Example 3: Apple Unveils New iPhone in September to Improve User Experience    
  - Include domain/genre of the content.
  - Include key entities.
  - If mentioned, include geographic location. If not specified or determinable respond "N/A".
  - Include a summary using less than 150 words. 
    - The summary should include key events, actions, data points, takeaways, entities and implications
  - Include actionable insight focusing on the core arguments, key datapoints and takeaways.
  - If the analyzed and extracted content is incoherent with each other, respond "N/A" to each of the above
  
STEP 4:
  - Remove any wrapper and filler texts. 
  - Convert the response into english if it is not already in english.  

RESPONSE FORMAT: Respond using the following format
```markdown
# GIST: 
[One line gist or N/A]

# DOMAIN:
[Comma separated list of relevant domains or N/A]

# ENTITIES:
[Comma separate list of key entities or N/A]

# LOCATION
[Geographic location mentioned or N/A]

# SUMMARY
[one - two paragraph of summary]

# KEY TAKEWAYS
- Key takeaway 1
- Key takeaway 2
- Key takeway 3

# ACTIONABLE INSIGHT
[One line of actionable insight]
```

------ START OF NEWS/BLOG/POST ------
{input_text}
------ END OF NEWS/BLOG/POST ------
"""

SHORT_SUMMARY_INST = """
TASK: Your input is a scrapped news, blog or social media post. Extract relevant contents for the purpose of creating a intelligence briefing.

STEP 1: Analyze the input and disregard the irrelevant contents
  - The input is a scrapped webpage and hence may contain text chunks that are irrelevant such as: Pay-wall message, advertisements, related articles, website biolerplate
  - Remove the irrelevant contents from further processing.

STEP 2: Extract the contents relevant to the core meaning of the input
  - Extract the questions in focus or the events and actions that have happened or will take place
  - Extract the primary topic such as: Seeking Cofounder, Seeking Funding, Seeking Career Advice, Sharing Ideas, Sharing Results, New Product Release, Funding Raising Opportunity, Looking for Tools, Looking for Help, Hiring etc.
  - Extract the key entities such as person, organization, company, product, stock ticker, phenomenon mentioned
  - Extract the domain/genre of the content subject matter such as: Government & Politics, Business & Finance, Cybersecurity, Crypto & Blockchain, Software Engineering, DevOps, Artificial Intelligence, IoT and Gadgets, Physics, Chemistry, Economy, Environment, Clean Energy, Health, Arts & Culture, Startups & Entrepreneurship, Entertainment, Sports etc.

STEP 3: Formulate your response
  - Include a one-line gist using less than 16 words 
    - The gist should be formatted as [Who] [Action] [What] [Where - optional]
      - Who - the main entity or person involved.
      - Action - What has happened or what is happening or what is being done.
      - What - The core object or the impacted entity
      - Where - the action took place (if mentioned).
    - Examples of gist
      - Example 1: NASA Launches Mars Mission in 2025 to Study Planet’s Atmosphere.
      - Example 2: Early Career PM Seeking Advice on Navigating Corporate Politics.
      - Example 3: Non-technical Startup Founder Seeking a Technical Co-founder for a Healthcare Startup  
  - Include the topic
  - Include domain/genre of the content.
  - Include key entities.
  
STEP 4:
  - Convert the response into english if it is not already in english.  

RESPONSE FORMAT: Respond using the following format
```markdown
# GIST: 
[One line gist or N/A]

# TOPIC:
[Primary Topic]

# DOMAIN:
[Comma separated list of relevant domains or N/A]

# ENTITIES:
[Comma separate list of key entities or N/A]
```

------ START OF NEWS/BLOG/POST ------
{input_text}
------ END OF NEWS/BLOG/POST ------
"""

# INSTRUCTION_TEMPLATE = """
# TASKS:
# {tasks}

# When formulating your response:
#     - Convert the response into english if it is not already in english.
#     - Maintain the original input text's tone and style.

# RESPONSE FORMAT: Response MUST be a json object like the following
# {fields}

# ARTICLE/POST:
# {input}
# """

# INSTRUCTION_MAP = {
#     "summary": "- Rewrite the article/post using less that 250 words. This will be the 'summary'.",
#     "highlights": "- Extract top 3 sentences that are most relevant to the main points and core meaning of the entire article/post. This will be the 'highlights'.",
#     "title": "- Create a one sentence gist of article/post. This will be the 'title'.",
#     "names": "- Extract names of the top 1 - 4 people, product, company, organization or stock ticker mentioned in the article/post that influences the content. This will be the 'names'",
#     "domains": "- Identify 1 or 2 domains that the subject matter of the article/post aligns closest to, such as: Cybersecurity, Business & Finance, Health & Wellness, Astrophysics, Smart Automotive, IoT and Gadgets etc. This will be the 'domains'"
# }
# FIELDS_MAP = {
#     "summary": "string",
#     "title": "string",
#     "highlights": "list[strings]",
#     "names": "list[string]",
#     "domains": "list[string]"
# }



# SHORT_DIGEST_TEMPLATE = """
# TASKS:
# - Create a one sentence gist of article/post. This will be the 'title'.,
# - Extract names of the top 1 - 4 people, product, company, organization or stock ticker mentioned in the article/post that influences the content. These will be the 'names'.
# - Identify 1 or 2 domains that the subject matter of the article/post aligns closest to, such as: Cybersecurity, Business & Finance, Health & Wellness, Astrophysics, Smart Automotive, IoT and Gadgets etc. These will be the 'domains'.

# RESPONSE FORMAT:
# Response MUST be a json object of the following structure
# ```json
# {{ 
#     "title": string, 
#     "names": [string, string, string], 
#     "domains": [string, string]
# }}
# ```

# ARTICLE/POST:
# {input_text}
# """

# LONG_DIGEST_TEMPLATE = """
# TASKS:
# - Rewrite the article/post using less that 250 words. This will be the 'summary'.
# - Extract top 3 sentences that are most relevant to the main points and core meaning of the entire article/post. These will be the 'highlights'.
# - Create a one sentence gist of article/post. This will be the 'title'.
# - Extract names of the top 1 - 4 people, product, company, organization or stock ticker mentioned in the article/post that influences the content. These will be the 'names'.
# - Identify 1 or 2 domains that the subject matter of the article/post aligns closest to, such as: Cybersecurity, Business & Finance, Health & Wellness, Astrophysics, Smart Automotive, IoT and Gadgets etc. These will be the 'domains'.

# RESPONSE FORMAT: 
# Response MUST be a json object of the following structure
# ```json
# {{ 
#     "summary": string, 
#     "highlights": [string, string, string], 
#     "title": string, 
#     "names": [string, string, string], 
#     "domains": [string, string]
# }}
# ```
# 
# ARTICLE/POST:
# {input_text}
# """

# def create_messages(example): 
#     if example['labels'] == SHORT_FIELDS: instruction = SHORT_DIGEST_TEMPLATE.format(input_text=example['input'])
#     elif example['labels'] == LONG_FIELDS: instruction = LONG_DIGEST_TEMPLATE.format(input_text=example['input'])

#     # NOTE: this is designed to fail if there is no match
#     return [
#         {
#             "role": "system",
#             "content": "response_format:json_object"
#         },
#         {
#             "role": "user",
#             "content":  instruction
#         },
#         {
#             "role": "assistant",
#             "content": example['output']
#         }
#     ]




# SUMMARY_INST = """TASK: Rewrite the input text using less than 160 words.
# When formulating your response:
#     1. Extract 5-8 sentences within the input text that are most relevant to the core meaning and concept outlined in the input text. 
#     2. Include the central arguments, decisions or events.
#     3. Include the supporting facts to the arguments if they are present in the original input text.
#     4. Include concluding remark or open questions if they are present in the original input text.
#     5. Maintain the original input text's tone and style.
#     6. Convert the response into english if it is not already in english.
#     7. Avoid using wrapper phrases like 'The article is about' or 'The article covers' or 'Here is the rewritten article'

# RESPONSE FORMAT: Respond using markdown text formatting.
# ```markdown
# paragraph
# - optional bullet points for presenting discrete points (if any)
# optional paragraph
# ```

# INPUT:
# """

# HIGHLIGHTS_INST = """
# TASK: Extract the main highlights, gist, names and domains from the input text.

# When formulating your response:
#     1. Analyze the input text
#     2. Identify 3 sentences within the text that are most relevant to the main points and core meaning of the entire text. These are the "highlights".
#     3. Identify the top 5 people, product, company, organization or stock ticker that has influenced the contents of the input text. These will be the "names".
#     4. Identify the top 3 domains of the subject matter of the input such as: Cybersecurity, Business & Finance, Health & Wellness, Astrophysics, Smart Automotive, IoT and Gadgets etc. This will be the "domain".
#     5. Create a one sentence gist of the extracted "highlights". Include at least one of the extracted "names". Avoid using wrapper phrases like 'The article is about' or 'The article covers'. This will be the "gist".
#     6. Translate the response into english if it is not already in english keeping the same tone and style as the input text.

# RESPONSE FORMAT: Format your response as a JSON object with fields named: highlights, names, domains, gist. such as:
# ```json
# {
#     "highlights": [string, string, string],    
#     "names": [string, string, string],
#     "domains": [string, string, string],
#     "gist": string
# }
# ```

# INPUT:
# """

# SUMMARY_INST = """TASK: Your input an draft of a news article, blog or social media post. Rewrite the content as an intelligence briefing using less than 160 words and in markdown format.
# When analyzing the input:
#     - Extract the central arguments, decisions, events and supporting facts to the arguments as presented in the draft.
#     - Disregard text chunks that are irrelevant to the core meaning and concept. 
# When formulating your response:
#     - Avoid wrapper and filler texts. 
#     - Convert the response into english if it is not already in english.

# RESPONSE FORMAT: Respond using markdown formatting.
# ```markdown
# paragraph
# - optional bullet points for presenting discrete points (if any)
# optional paragraph
# ```

# ------ INPUT STARTS ------
# {{input_text}}
# ------ INPUT ENDS ------
# """

# EXTRACTION_INST = """TASK: Your input is a scrapped news article, blog or social media post. Extract the relevant contents such as facts, events, locations, entities and data points to create intelligence briefing.
# When analyzing the input:
#     - Keep in mind that it is a web scrapped content and hence may contain text chunks that are irrelevant to the core content of the input such as advertisements, navigation menus, site boilerplate text, related articles etc.
#     - Disregard text chunks that are irrelevant to the core meaning and concept.
#     - Extract the key facts, events, data points, locations and acting/impacted entities that influence the core content of the article.
# When formulating your response:
#     - Include a one-line gist that capture the core information conveyed by the text. Gist should include
#         - Who - the main entity or person involved.
#         - Action - What has happened or what is happening or what is being done.
#         - What - The core object of the article or the impacted entity
#         - Where - the action took place (if mentioned).
#     - Include a summary using less than 150 words. The summary should include
#         - key takeaways presented by the article that focus on core of its content.
#         - key events and actions that have occured place or will take place.
#         - key data points and their implications.
#         - key entities such as such as person, organization, company, product, stock ticker etc. which are mentioned either as actors or impacted entities in the key events.
#     - Include content domains such as Government & Politics, Business & Finance, Cybersecurity, Crypto & Blockchain, Software Engineering, DevOps, Artificial Intelligence, IoT and Gadgets, Physics, Chemistry, Economy, Environment, Clean Energy, Health, Arts & Culture, Human Interest, Startups & Entrepreneurship.
#     - Include actionable insight focusing on the core arguments, decisions or events.
#     - If mentioned, include geographic location. The location should be dominant ISO 3166-1 alpha-3 country code associated with the article's content, "GLOBAL" if worldwide scope, or "N/A" if not applicable/determinable.
#     - Avoid wrapper and filler texts. 
#     - Convert the response into english if it is not already in english.  
    
# EXAMPLES OF GIST:
# Example 1: NASA Launches Mars Mission in 2025 to Study Planet’s Atmosphere.
# Example 2: Congress Debates Healthcare Reform This Week to Address Rising Costs
# Example 3: Apple Unveils New iPhone in September to Improve User Experience

# EXAMPLE OF DATAPOINTS:
# Example 1: 5,000 Jobs Created in Tech Industry in 2023
# Example 2: 20% Increase in Electric Vehicle Sales in Europe
# Example 3: 40% Decline in Consumer Spending on Luxury Goods

# RESPONSE FORMAT: Respond using JSON object
# ```json
# {{
#     "gist": "string",
#     "summary": "string",
#     "key_points": ["string"],
#     "key_events": ["string"],
#     "data_points": ["string"],
#     "key_entities": ["string"],
#     "content_domains": ["string"],
#     "actionable_insight": "string",
#     "location": "string" 
# }}
# ```

# ------ INPUT STARTS ------
# {input_text}
# ------ INPUT ENDS ------
# """




# SUMMARY_INST = """TASK: Your input is a scrapped news article, blog or social media post. Rewrite the content as an intelligence briefing using less than 200 words.
# When analyzing the input:
#     - Keep in mind that it is a web scrapped content and hence may contain text chunks that are irrelevant to the core content of the input such as advertisements, navigation menus, site boilerplate text, related articles etc.
#     - Disregard text chunks that are irrelevant to the core meaning and concept.
#     - Extract the key facts, events, data points, locations and acting/impacted entities that influence the core content of the article.
# When formulating your response:
#     - Include key points, events, take-aways, data points and entities such as person, organization, company, product, stock ticker etc. which influence the core content of the article.
#     - Include actionable insight focusing on the core arguments, decisions or events.
#     - Avoid wrapper and filler texts. 
#     - Convert the response into english if it is not already in english. 

# RESPONSE FORMAT: Respond using markdown format

# ------ INPUT STARTS ------
# {input_text}
# ------ INPUT ENDS ------
# """

# SUMMARY_INST = """TASK: Your input is a scrapped news article, blog or social media post. Rewrite the content as an intelligence briefing using less than 200 words.

# PROCESS:
# - When analyzing the input:
#   - Keep in mind that it is a web scrapped content and hence may contain text chunks that are irrelevant to the core content of the input such as advertisements, navigation menus, site boilerplate text, related articles etc.
#   - Disregard text chunks that are irrelevant to the core meaning and concept.
#   - Extract the key facts, events, data points, locations and acting/impacted entities that influence the core content of the article.
# - When formulating your response:
#   - Include key points, take-aways, events and influencial data points that capture the core information conveyed by the text. This should include
#     - Who - the main entity or person involved.
#     - Action - that is happening or being done or will take place.
#     - What/Who - the action impacted (The core topic or object of the article).
#     - Where - the action took place (if mentioned).
#     - When - the action took place (if mentioned).
#     - How/Why - the action took place (if mentioned).
#   - Include actionable insight focusing on the core arguments, decisions or events.
#   - Avoid wrapper and filler texts. 
#   - Convert the response into english if it is not already in english. 

# RESPONSE FORMAT: Respond using markdown format

# ------ INPUT STARTS ------
# {input_text}
# ------ INPUT ENDS ------
# """

# HIGHLIGHTS_INST = """TASK: Extract structured, semantically dense information from this article. This data will be used primarily for matching articles to diverse user interests via semantic similarity search and keyword filtering. Focus on extracting core facts and concepts; human readability is secondary to informational density.
# RESPONSE_FORMAT: Return only the JSON object below. Follow the examples provided.

# --- EXAMPLES STARTS ---

# Example 1: Technology Product Launch
# *Input Article Snippet (Conceptual):*
# ```
# # NovaCorp Unveils 'Photon' AI Chip for Edge Computing
# Body: San Francisco – NovaCorp today announced Photon, its new AI accelerator chip designed for low-power edge devices. Photon boasts 10 TOPS performance at under 2 watts, targeting IoT and autonomous systems. CEO Jane Smith highlighted partnerships with device makers OmniGadget and AutoDrive. Initial benchmarks show significant speedups in image recognition tasks compared to competitors. Shipments begin Q3. Analysts see this intensifying competition in the edge AI market.
# ```

# *RESPONSE JSON:*
# ```json
# {
#   "language": "en",
#   "primary_location": "USA",
#   "completeness": "COMPLETE",
#   "content_quality": "OK",
#   "event_summary_points": [
#     "NovaCorp announces Photon AI chip",
#     "Designed for low-power edge devices",
#     "Specs: 10 TOPS, <2W power",
#     "Targets IoT, autonomous systems",
#     "Partnerships: OmniGadget, AutoDrive",
#     "Shipments start Q3"
#   ],
#   "thematic_keywords": [
#     "Edge AI acceleration",
#     "Low-power computing",
#     "AI hardware market",
#     "Competitive landscape",
#     "IoT enablement",
#     "Autonomous system components"
#   ],
#   "topic_tags": [
#     "Artificial Intelligence",
#     "Semiconductors",
#     "Edge Computing",
#     "Internet of Things (IoT)",
#     "Hardware",
#     "NovaCorp"
#   ],
#   "key_entities": [
#     "NovaCorp",
#     "Photon",
#     "Jane Smith",
#     "OmniGadget",
#     "AutoDrive",
#     "San Francisco"
#   ],
#   "content_focus": [
#     "Technology",
#     "Business"
#   ]
# }
# \`\`\`

# **Example 2: Geopolitical Development**

# *Input Article Snippet (Conceptual):*
# ```
# # Maritime Tensions Rise in Azure Strait After Naval Incident
# Tensions flared today in the Azure Strait following a close encounter between naval vessels from Accadia and Borealia. Accadia claims a Borealian patrol boat entered its territorial waters, issuing warnings before escorting it out. Borealia disputes the boundary claim and accuses Accadian ships of aggressive maneuvers. The strait is a critical shipping lane. Regional powers call for de-escalation. This follows months of diplomatic friction over fishing rights.
# ```

# *Output JSON:*
# ```json
# {
#   "language": "en",
#   "primary_location": "GLOBAL", // Region specific, but involves multiple nations, treat as global issue unless focused solely on one nation's internal perspective
#   "completeness": "COMPLETE",
#   "content_quality": "OK",
#   "event_summary_points": [
#     "Naval encounter in Azure Strait",
#     "Involved Accadia and Borealia vessels",
#     "Accadia alleges territorial water violation",
#     "Borealia disputes boundary, alleges aggression",
#     "Regional powers urge calm"
#   ],
#   "thematic_keywords": [
#     "Maritime security",
#     "Geopolitical tension",
#     "Territorial disputes",
#     "Freedom of navigation",
#     "International relations",
#     "Shipping lane security",
#     "De-escalation efforts"
#   ],
#   "topic_tags": [
#     "International Relations",
#     "Maritime Law",
#     "Geopolitics",
#     "Accadia",
#     "Borealia",
#     "Azure Strait",
#     "Naval Operations"
#   ],
#   "key_entities": [
#     "Accadia",
#     "Borealia",
#     "Azure Strait"
#   ],
#   "content_focus": [
#     "World Affairs",
#     "Security",
#     "Politics"
#   ]
# }
# ```

# **Example 3: Scientific Breakthrough Report**

# *Input Article Snippet (Conceptual):*
# ```
# # Researchers Develop Novel Catalyst for Greener Plastic Production
# A team at Quantum University has developed a new palladium-based catalyst that enables the production of common plastics using significantly less energy and generating fewer harmful byproducts. Published in 'Nature Synthesis', the study shows a 30% reduction in energy requirements for polymerization. Lead researcher Dr. Eva Rostova notes potential for large-scale industrial adoption, reducing the carbon footprint of plastic manufacturing. Further testing is needed for durability.
# ```

# *Output JSON:*
# ```json
# {
#   "language": "en",
#   "primary_location": "N/A", // Specific university, but discovery is global relevance, location not central focus
#   "completeness": "COMPLETE",
#   "content_quality": "OK",
#   "event_summary_points": [
#     "New palladium catalyst developed",
#     "Created by Quantum University researchers",
#     "Reduces energy use (30%) in plastic production",
#     "Lowers harmful byproducts",
#     "Published in 'Nature Synthesis'"
#   ],
#   "thematic_keywords": [
#     "Green chemistry",
#     "Sustainable manufacturing",
#     "Catalysis innovation",
#     "Plastic production efficiency",
#     "Carbon footprint reduction",
#     "Industrial process improvement"
#   ],
#   "topic_tags": [
#     "Chemistry",
#     "Materials Science",
#     "Sustainability",
#     "Plastics",
#     "Research",
#     "Catalysis"
#   ],
#   "key_entities": [
#     "Quantum University",
#     "Eva Rostova",
#     "Nature Synthesis",
#     "Palladium"
#   ],
#   "content_focus": [
#     "Science",
#     "Technology",
#     "Environment"
#   ]
# }
# ```

# **--- End Examples ---**

# **Now, analyze the following article and provide the JSON output:**

# ```json
# {
#   "language": "string",              // ISO 639-1 alpha-2 code.
#   "primary_location": "string",      // Dominant ISO 3166-1 alpha-3 country code associated with the article's content, "GLOBAL" if worldwide scope, or "N/A" if not applicable/determinable.
#   "completeness": "COMPLETE" | "PARTIAL_USEFUL" | "PARTIAL_USELESS", // Assess if the *extracted text* seems complete or truncated.
#   "content_quality": "OK" | "LOW_QUALITY" | "JUNK", // Assess the *nature* of the content.
#   "event_summary_points": ["string"], // If content quality is OK and completeness is not PARTIAL_USELESS: List of 2-5 short, factual bullet points about the specific event reported. Focus on: Who did what? What was the outcome? Where? Use keywords and essential facts, not full sentences. Empty array otherwise.
#   "thematic_keywords": ["string"],   // If content quality is OK and completeness is not PARTIAL_USELESS: List of 3-7 keywords/phrases representing the broader significance, context, or underlying theme. Answer: Why does this matter? What larger trend is this part of? Empty array otherwise.
#   "topic_tags": ["string"],          // If content quality is OK and completeness is not PARTIAL_USELESS: List of 3-7 concise keywords/phrases for the core subjects/concepts discussed. Think potential user interests or search terms. Empty array otherwise.
#   "key_entities": ["string"],        // If content quality is OK and completeness is not PARTIAL_USELESS: List of specific proper nouns (people, organizations, named locations like cities/regions, product names) central to the event described. Empty array otherwise.
#   "content_focus": ["string"]        // If content quality is OK and completeness is not PARTIAL_USELESS: List of 1-3 tags describing the article's primary angle from the allowed list: ["Politics", "Business", "Technology", "Science", "World Affairs", "Economy", "Environment", "Health", "Security", "Culture", "Human Interest", "Analysis", "Breaking News"]. Empty array otherwise.
# }
# ```

# **Detailed Criteria Clarifications:**

# *   **Completeness:**
#     *   \`COMPLETE\`: Appears to be the full article text available in the input.
#     *   \`PARTIAL_USEFUL\`: Text appears truncated (e.g., paywall fade-out, "read more" link cut-off, abruptly ends mid-paragraph) but enough core information is present to understand the basic story and extract meaningful data.
#     *   \`PARTIAL_USELESS\`: Only headline, lede, or a tiny snippet is present. Virtually no usable content beyond the absolute minimum to identify the topic, making extraction of summaries/keywords impossible or pointless.

# *   **Content Quality:**
#     *   \`OK\`: Standard news reporting, analysis, interviews, press releases, or other substantive factual content. Well-structured and informative.
#     *   \`LOW_QUALITY\`: Content is present but potentially problematic. Examples: very thin/short updates with little new info, heavy on opinion/ranting with minimal facts, celebrity gossip focus, sensationalized or clickbait-style writing (even if factual), user-generated content (like comments sections mistakenly scraped), lists/roundups with minimal detail per item. *May be useful depending on user needs, but flag it.*
#     *   \`JUNK\`: Input text is clearly not usable article content. Examples: Error messages (404, 500), login/signup prompts, ad-heavy pages with no real article, navigation menus or site boilerplate text only, code snippets, raw data tables without context, content is obviously machine-generated gibberish or non-prose, duplicate template text. *These should generally be filtered out entirely.*

# *   **Semantic Density:** For \`event_summary_points\` and \`thematic_keywords\`, prioritize packing meaning into keywords and short phrases. Avoid conversational filler ("As reported today...", "It is interesting to note that..."), introductory clauses, or full grammatical sentences. Think like you're writing concise tags or dense factual notes for a database entry, not prose for a human reader.

# *   **Distinctions between Key Fields:**
#     *   \`event_summary_points\`: Focus strictly on the *specific facts* of the *event being reported* in this article. Who did what, when, where, what was the immediate outcome? Use keywords and essential nouns/verbs.
#     *   \`thematic_keywords\`: Describe the *broader context, significance, and underlying forces* related to the event. Why does this event matter in the bigger picture? What trends does it connect to? What are the potential implications? Use conceptual phrases.
#     *   \`topic_tags\`: Identify the *core subjects or categories* the article falls under. What general areas of interest does this article cover? Think of these like index terms or categories in a library. Use concise nouns or noun phrases.
#     *   \`key_entities\`: List the *specific named proper nouns* (people, organizations, specific geographic locations like cities/regions if central to the event, product names, legislative bill names, etc.) that are the main actors or subjects *within the specific event*.

# *   **\`content_focus\` Selection:** Choose the 1-3 tags from the provided list \`["Politics", "Business", "Technology", "Science", "World Affairs", "Economy", "Environment", "Health", "Security", "Culture", "Human Interest", "Analysis", "Breaking News"]\` that best capture the *primary angle or lens* through which the article presents the information. An article about a new environmental regulation could be \`Environment\` and \`Politics\` and maybe \`Economy\` if it focuses on business impact. \`Analysis\` is for pieces primarily offering interpretation or opinion on events, rather than just reporting them. \`Breaking News\` suggests a focus on immediate, unfolding events. 
# """

# create_highlights_prompt = lambda text: [
#     {
#         "role": "system",
#         "content": SYSTEM_INST
#     },
#     {
#         "role": "user",
#         "content": HIGHLIGHTS_INST+text
#     }
# ]

# create_extracts_prompt = lambda bean: [
#     {
#         "role": "system",
#         "content": SYSTEM_INST
#     },
#     {
#         "role": "user",
#         "content": EXTRACTION_INST.format(input_text=bean['text'])
#     }
# ]
