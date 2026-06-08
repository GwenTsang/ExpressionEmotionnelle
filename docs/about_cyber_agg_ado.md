# CyberAgressionAdo-Large
 Dataset containing annotated scenarios mimicking cyber aggression situations tat may occur on private instant messaging platforms. It is an extended version of the dataset CyberAgressionAdo-v2 introduced in the paper: https://hal.science/hal-04514689.
v 3.1: October 16 2024
https://github.com/aollagnier/CyberAgression-Large


1) DESCRIPTION

This is the README file for the CyberAgressionAdo-Large dataset

CyberAgressionAdo-Large contains 36 scenarios including 11 about ethnicity, 9 about homophobia, 8 about obesity and 8 about religion. Messages are annotated using a fine-grained hierarchical annotation scheme.

The dataset was annotated by 3 expert annotators, with background in computational linguistics. The gold labels were assigned taking the agreement of the two annotators into consideration. 

Each conversation contains up to 7 labels each corresponding to one of the following levels:

Role: This label corresponds to the type of active roles involved in aggressions. It helps categorize comments based on the roles individuals play in online interactions.

Hate: This label is based on Salminen et al. (2020) and is used for comments containing language that includes hate speech targeted towards individuals or groups, profanity, offensive language, or toxicity. In simpler terms, it identifies comments that are rude, disrespectful, and have the potential to lead to negative consequences both online and offline, affecting individuals, communities, and society as a whole.

Target: This label signifies the role of the individual(s) who are the recipients of hate speech and/or verbal abuse within a conversation.

Verbal Abuse: This label is used to identify instances of verbal abuse in written language. It includes five frequently occurring types of abuse: blaming, name-calling, threats, denigration, and other forms of verbal mistreatment.

Intention: This label is employed to understand the purpose or message conveyed through the comments. It helps categorize comments based on the underlying intentions of the individuals posting them.

Context: This label is used to establish the context within which these messages are situated, particularly as responses in a conversation. It provides insights into the surrounding circumstances and helps in better understanding the comments.

Sentiment: This label is used to establish the emotional tone of a message, categorizing it as positive, negative, or neutral based on the feelings or attitudes expressed.

Moreover, this resource incorporates annotations contributed by three annotators, facilitating the adoption of computational methodologies aligned with the emerging perspectivist trend in machine learning. Each tab within every file corresponds to the annotations provided by an individual annotator.

***Majority vote* This folder contains Excel files where a majority voting strategy was applied to the labels provided by three annotators. In this strategy, the option that appears in more than half of the annotations is selected as the final choice. When no clear majority was found, discrepancies were addressed manually.

2) FORMAT

Instances are included in CSV format as follows:

ID	TIME    NAME    TEXT    ROLE    TARGET  HATE    VERBAL_ABUSE    INTENTION   CONTEXT   SENTIMENT

Whenever a label is not given, a value NULL is inserted (e.g. INSTANCE	NOT	NULL	NULL)

The column names in the file are the following:

The labels used in the annotation are listed below.

3) FRHATE ANNOTATION SCHEME

(A) Level Role: Active roles involved in aggressions.

- (victim) victim - person who is harassed.
- (bully) bully - person who initiates the harassment.
- (victim_support) bystander-defender: person who helps the victim and discourages the harasser.
- (bully_support) bystander-assistant: person who takes part in the actions of the harasser.
- (conciiator) conciliator: a common friend of the bully and the victim mediating the disagreement among active participants

In our annotation, we label a post based on the role of the author within the context of the scenario being discussed.

(B) Level HATE: Identifies offensive and damaging language. 

- (OAG) overtly aggressive: explicitly aggressive communication involves the use of offensive, hostile, or threatening language, as well as hate speech or insults, and can also include subtle elements that convey aggression through context and perception.
- (CAG) covertly aggressive: linguistic strategies to conceal aggression avoiding explicit threats or insults, relying on tactics like figurative language, rhetorical questions, and euphemisms, among others.
- (NAG) non aggressive:  messages free from explicit derogatory language, threats, direct harm, or subtle linguistic strategies that might subtly imply an ambiguous intentions to harm or intimidate.

(C) Level Target: the role of the individual(s) who are the targets of online hate within a conversation or discussion.

- (victim) victim - person who is harassed.
- (bully) bully - person who initiates the harassment.
- (victim_support) bystander-defender: person who helps the victim and discourages the harasser.
- (bully_support) bystander-assistant: person who takes part in the actions of the harasser.
- (conciiator) conciliator: a common friend of the bully and the victim mediating the disagreement among active participants 

In our annotation, we label the target(s) whether the message is overt or covert aggression. 

(D) Level Verbal Abuse: verbal abuse types

- (BLM) Blaming - making the victim believe they are responsible for the abusive behavior or that they bring the verbal abuse upon themselves.
- (NCG) name calling - abusive, derogatory language or insults that chip away at the target's self-esteem, sense of self-worth, and selfconcept.
- (THR) threat - statements meant to frighten, control, and manipulate the victim into compliance.
- (DNG) denigration - harsh remarks that are meant to make the person feel bad about themselves and are not constructive, but deliberate and hurtful.
- (OTH) aggression other - content corresponding to other types of abusive, derogatory language or insults deliberately harmful not fitting in the other categories.

Posts containing overt or covert aggression.

(E) Level Intention: unveil the authors' intentions (what they aim to achieve or convey through their messages)

- (ATK) Attack - Communication demonstrating deliberate aggression toward victims, their supporters, or mediators, including insults, threats, mockery, exclusion, taunting, and discrediting, is characteristic of bullies and their allies, serving as a tool to harm or escalate violence.
- (DFN) Defend - Communication refers to impulsive, non-deliberate responses aimed at protecting oneself or others from perceived attacks, which can be in retaliation for real or perceived attacks. These responses, exclusive to victims, their supporters, or conciliators.
- (CNS) Counterspeech - Any non-aggressive response to harmful speech, aiming to undermine it. It employs strategies like presenting facts, highlighting contradictions, warning of consequences, and denouncing hate. It's initiated by victims, supporters, or conciliators.
- (AIN) Abet/Instigate - Messages that support, encourage, or validate previous negative messages or incite aggression, either before or during the act, often in reaction to bullies and their supporters.
- (GSL) Gaslightning - text or speech used by bullies and their supporters to minimize, distort, or manipulate another person's trauma or memory, utilizing tactics such as denying harm, victim-blaming, questioning memory, invalidating feelings, and using group consensus to create doubt in the victim's perception of reality.
- (EMP) Empathy - Messages that show understanding, compassion, and support for those affected, including self-empathy by acknowledging one's own distress, and is typically expressed by victims, their supporters, or conciliators.
- (CR) conflict-resolution - Communication, typically initiated by victim supporters and conciliators, focuses on resolving conflicts and de-escalating situations through non-aggressive means such as mediation, mitigation, and education to promote positive online behavior.
- (OTH) other - Messages with unclear intent, this involves tagging in cases like neutral utterances, non-standard language (e.g., incomplete sentences, one-word responses, or sentence fragments), and annotating emoticons and emojis used to convey emotions, attitudes, and reactions.

Messages are labeled whether they are aggressive or not.

(F) Context : establish the context within which these messages are situated as responses

- (ATK) Attack - Communication demonstrating deliberate aggression toward victims, their supporters, or mediators, including insults, threats, mockery, exclusion, taunting, and discrediting, is characteristic of bullies and their allies, serving as a tool to harm or escalate violence.
- (DFN) Defend - Communication refers to impulsive, non-deliberate responses aimed at protecting oneself or others from perceived attacks, which can be in retaliation for real or perceived attacks. These responses, exclusive to victims, their supporters, or conciliators.
- (CNS) Counterspeech - Any non-aggressive response to harmful speech, aiming to undermine it. It employs strategies like presenting facts, highlighting contradictions, warning of consequences, and denouncing hate. It's initiated by victims, supporters, or conciliators.
- (AIN) Abet/Instigate - Messages that support, encourage, or validate previous negative messages or incite aggression, either before or during the act, often in reaction to bullies and their supporters.
- (GSL) Gaslightning - text or speech used by bullies and their supporters to minimize, distort, or manipulate another person's trauma or memory, utilizing tactics such as denying harm, victim-blaming, questioning memory, invalidating feelings, and using group consensus to create doubt in the victim's perception of reality.
- (EMP) Empathy - Messages that show understanding, compassion, and support for those affected, including self-empathy by acknowledging one's own distress, and is typically expressed by victims, their supporters, or conciliators.
- (CR) conflict-resolution - Communication, typically initiated by victim supporters and conciliators, focuses on resolving conflicts and de-escalating situations through non-aggressive means such as mediation, mitigation, and education to promote positive online behavior.
- (OTH) other - Messages with unclear intent, this involves tagging in cases like neutral utterances, non-standard language (e.g., incomplete sentences, one-word responses, or sentence fragments), and annotating emoticons and emojis used to convey emotions, attitudes, and reactions.

Messages are labeled whether they are aggressive or not.

(G) Sentiment : determine the emotional tone behind a body of text.

(POS) Positive Sentiment - express joy, love, support, or appreciation in a non-aggressive tone, promoting respect, kindness, and encouragement, often including casual or humorous comments without intent to offend.
(NEG) Negative Sentiment - express emotions like anger, frustration, or disapproval, often containing aggression or hostility, including insulting or offensive language, and can promote violence, mockery, or exclusion without necessarily using profanity.
(NEU) Neutral Sentiment - convey facts, objective statements, or inquiries without expressing strong emotions, and does not reveal the author's emotional state.
Messages are labeled whether they are aggressive or not.

Here are the possible label combinations in the FRHATE annotation.

-	NAG NULL NULL DFN ATK NEU
-	OAG victim_support THR ATK EMP POS
-	CAG victim BLM GSL NULL NEU

4) CITING THE DATASET

If you use this dataset we kindly ask you to include a reference to the paper below. 

@inproceedings{DBLP:conf/coling/Ollagnier24,
  author       = {Ana{\"{\i}}s Ollagnier},
  editor       = {Nicoletta Calzolari and
                  Min{-}Yen Kan and
                  V{\'{e}}ronique Hoste and
                  Alessandro Lenci and
                  Sakriani Sakti and
                  Nianwen Xue},
  title        = {CyberAgressionAdo-v2: Leveraging Pragmatic-Level Information to Decipher
                  Online Hate in French Multiparty Chats},
  booktitle    = {Proceedings of the 2024 Joint International Conference on Computational
                  Linguistics, Language Resources and Evaluation, {LREC/COLING} 2024,
                  20-25 May, 2024, Torino, Italy},
  pages        = {4287--4298},
  publisher    = {{ELRA} and {ICCL}},
  year         = {2024},
  url          = {https://aclanthology.org/2024.lrec-main.383},
  biburl       = {https://dblp.org/rec/conf/coling/Ollagnier24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}