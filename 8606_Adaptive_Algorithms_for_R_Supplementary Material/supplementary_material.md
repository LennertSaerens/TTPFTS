|                       | Adaptive     | Algorithms | for                           | Relaxed |     |
| --------------------- | ------------ | ---------- | ----------------------------- | ------- | --- |
|                       |              | Pareto Set | Identification                |         |     |
|                       | CyrilleKone1 |            | EmilieKaufmann1               |         |     |
| cyrille.kone@inria.fr |              |            | emilie.kaufmann@univ-lille.fr |         |     |
LauraRichert2
laura.richert@u-bordeaux.fr
1Univ. Lille,Inria,CNRS,CentraleLille,UMR9198-CRIStAL,F-59000Lille,France
2Univ. Bordeaux,Inserm,Inria,BPH,U1219,Sistm,F-33000Bordeaux,France
Abstract
Inthispaperwerevisitthefixed-confidenceidentificationoftheParetooptimal
| setinamulti-objectivemulti-armedbanditmodel.  |           |                   | Asthesamplecomplexityto  |                        |           |
| --------------------------------------------- | --------- | ----------------- | ------------------------ | ---------------------- | --------- |
| identify                                      | the exact | Pareto set can be | very large, a relaxation | allowing               | to output |
| someadditionalnear-optimalarmshasbeenstudied. |           |                   |                          | Inthisworkwealsotackle |           |
alternativerelaxationsthatallowinsteadtoidentifyarelevantsubsetofthePareto
set. Notably,weproposeasinglesamplingstrategy,calledAdaptiveParetoExplo-
ration,thatcanbeusedinconjunctionwithdifferentstoppingrulestotakeinto
| accountdifferentrelaxationsoftheParetoSetIdentificationproblem. |     |     |     | Weanalyze |     |
| --------------------------------------------------------------- | --- | --- | --- | --------- | --- |
thesamplecomplexityofthesedifferentcombinations,quantifyinginparticular
thereductioninsamplecomplexitythatoccurswhenoneseekstoidentifyatmost
| kParetooptimalarms. |     | WeshowcasethegoodpracticalperformanceofAdaptive |     |     |     |
| ------------------- | --- | ----------------------------------------------- | --- | --- | --- |
ParetoExplorationonareal-worldscenario,inwhichweadaptivelyexploreseveral
| vaccination | strategies | against Covid-19 | in order to | find the optimal ones | when |
| ----------- | ---------- | ---------------- | ----------- | --------------------- | ---- |
multipleimmunogenicitycriteriaaretakenintoaccount.
1 Introduction
Inamulti-armedbanditmodel,anagentsequentiallycollectssamplesfromseveralunknowndistri-
butions,calledarms,inordertolearnaboutthesedistributions(pureexploration),possiblyunder
theconstrainttomaximizethesamplescollected,viewedasrewards(regretminimization). These
objectiveshavebeenextensivelystudiedfordifferenttypesofunivariatearmsdistributions[21]. In
thispaper,weconsiderthelesscommonsettinginwhicharmsaremulti-variatedistributions. We
areinterestedintheParetoSetIdentification(PSI)problem. Inthispureexplorationproblem,the
agentseekstoidentifythearmsthatare(Pareto)optimal,i.e. suchthattheirexpectedvaluesforall
objectivesarenotuniformlyworsethanthoseofanotherarm.
Weformalizethisasafixed-confidenceidentificationproblem:ineachroundttheagentselectsanarm
A usinganadaptivesamplingruleandobservesasampleX ∈RD fromtheassociateddistribution.
| t   |     |     | t   |     |     |
| --- | --- | --- | --- | --- | --- |
Itfurtherusesanadaptivestoppingruleτ todecidewhentostopsamplingandoutputasetofarmsS(cid:98)τ
whichisherguessfor(anapproximationof)thetrueParetosetS⋆. Givenariskparameterδ ∈(0,1),
|     |     |     | satisfyP(S(cid:98)τ | =S⋆)≥1−δforexactPareto |     |
| --- | --- | --- | ------------------- | ---------------------- | --- |
thisguessshouldbecorrectwithhighprobability,e.g.
setidentification,whilerequiringasmallsamplecomplexityτ. Thisgeneralizesthewell-studied
fixed-confidenceBestArmIdentification(BAI)problem[8,15,10]tomultipleobjectives.
37thConferenceonNeuralInformationProcessingSystems(NeurIPS2023).

Ourmotivationtostudymulti-objectiveadaptiveidentificationstemsfromthedesignofadaptiveearly-
phaseclinicaltrials. InphaseI/IItrials,theeffectsofamoleculeinhumansareexplored,andseveral
biologicalendpointsmaybeassessedatthesametimeasindicativemarkersofefficacy. Inparticular,
inthecontextofvaccinedevelopment,early-phasetrialsusuallyassessmultipleimmunogenicity
endpoints(i.e. variousmarkersoftheeffectsofthevaccineontheimmunesystem,suchasdifferent
facetsofantibodyresponsesorotherimmuneparameters). Intheabsenceofaknowncorrelateof
protectionduringearlyclinicaldevelopment,theseendpointsmaynothaveaclearapriorihierarchy,
maynotallbecorrelated,whichmakesanexaminationoftheParetosetofdifferentvaccinalstrategies
particularlyrelevant. Inaddition,giventheavailabilityofvariousvaccineplatforms(suchasmRNA
vaccines,viral-vectorvaccines,proteinvaccines),asexemplifiedbyCovid-19vaccines,theremay
beaneedtoadaptivelyscreenthevariousresultingvaccinestrategiestoselectthemostpromising
ones. Apartfromclinicaltrials,activeParetoSetIdentificationcanbemeaningfulinmanyreal-word
contexts, and we refer the reader to the various examples given by [28], such as in hardware or
software design. Other applications include A/B/n testing for marketing or online recommender
systemsinwhichitiscommontojointlyoptimizemultiple(possiblyconflicting)objectivessuch
asuserbehavioralmetrics(e.g. clicks,streams,dwelltime,etc),supplierexposureobjectives(e.g.
diversity)andplatformcentricobjectives(e.g. promotions)[22].
Formanyapplications,thesamplecomplexityofexactPSIcanbeprohibitive,eitherwhenthereare
manyclosetooptimalarmsorwhentheParetosetisverylarge,anddifferentrelaxationshavebeen
consideredintheliterature[4,27]. Goingbacktoourmotivation,inanadaptivetrialthataimsat
pre-selectingacertainnumberoftreatmentsorvaccinestrategiesforfurtherinvestigationsinclinical
trials,practicalconstraints(thecostandfeasibilityofthetrials)imposeaconstraintonthemaximal
numberofinterestingarmsthatcanbeidentified. Thismotivatestheintroductionofanewsetting
wheretheagentisaskedtoidentifyatmostkParetooptimalarms. Interestinglythesamplingrule
that we propose for this setting can be used to solve (some generalizations of) other relaxations
consideredintheliterature.
Related work The work most closely related to ours is that of Auer et al. [4], who propose a
relaxation,whichwerefertoasε
1
-PSI:theiralgorithmreturnsasetS(cid:98)thatcontainsw.h.p. allthe
Paretooptimalarmsandpossiblysomesub-optimalarms,whichwhenincreasedbyε coordinate-
1
wisebecomeParetooptimal. Forarmsthathavesub-Gaussianmarginals,theyprovideaninstance-
dependentsamplecomplexityboundscalingwithsomenotionofsub-optimalitygapforeacharm.
TheworkofZaluagaetal. [28,27]studiesastructuredvariantoffixed-confidencePSIinwhichthe
meansareregularfunctionsofarms’descriptors. TheyuseGaussianprocessmodelingandobtain
worse-casesamplecomplexitybounds. Inparticular[27]considerstheidentificationofanε-cover
oftheParetoset, whichisarepresentativesubsetofthe(ε)-Paretosetthatwillberelatedtoour
(ε ,ε )-PSIcriterion. Thealgorithmsof[4]andthoseof[28,27]intheunstructuredsetting1havethe
1 2
sameflavor: theysampleuniformlyfromasetofactivearmsandremovearmsthathavebeenfound
sub-optimal(ornotrepresentative). Aueretal.[4]furtheraddsanacceptationmechanismtostop
samplingsomeofthearmsthathavebeenfound(nearly-)optimalandareguaranteednottodominate
anarmoftheactiveset. Inthispaper,weproposeinsteadamoreadaptiveexplorationstrategy,which
departsfromsuchaccept/rejectmechanismsandissuitedfordifferenttypesofrelaxation,including
ournovelk-relaxation.
AdaptiveParetoExploration(APE)leveragesconfidenceintervalsonthedifferencesofarms’coor-
dinatesinordertoidentifyasinglearmtoexplore,inthespiritoftheLUCB[15]orUGapEc[9]
algorithmsforTop-midentificationin(one-dimensional)banditmodels. Thesealgorithmshavebeen
foundouttobepreferableinpracticetotheircompetitorsbasedonuniformsamplingandeliminations
[19],anobservationthatwillcarryovertoAPE.Besidesthemulti-dimensionalobservations,we
emphasizethatamajorchallengeofthePSIproblemwithrespecttoe.g. Topmidentificationisthat
thenumberofarmstoidentifyisnotknowninadvance. Moreover,whenrelaxationsareconsidered,
therearemultiplecorrectanswers. Intheone-dimensionalsettings,findingoptimalalgorithmsin
thepresenceofmultiplecorrectanswersisnotoriouslyhardasdiscussedbytheauthorsof[5],and
theirlower-boundbasedapproachbecomesimpracticalinourmulti-dimensionalsetting. Finally,we
remarkthatthek-relaxationcanbeviewedasanextensionoftheproblemofidentifyinganyk-sized
subsetoutofthebestmarmsinastandardbandit[25].
1PALreliesonconfidenceintervalsthatfollowfromGaussianprocessregression,butcanalsobeinstantiated
withsimplerun-structuredconfidenceintervalsasthoseusedinourworkandinAuer’s
2

Beyond Pareto set identification, other interesting multi-objective bandit identification problems
havebeenstudiedintheliterature. Forexample[6]proposeanalgorithmtoidentifysomeparticular
armsintheParetosetthroughascalarizationtechnique[23]. Theideaistoturnthemulti-objective
pure-explorationproblemintoasingle-objectiveone(uniqueoptimalarm)byusingareal-valued
preferencefunctionwhichisonlymaximizedbyParetooptimalarms(seee.g[23]forsomeexamples
ofthesefunctions). Inpractice,afamilyofthosefunctionscanbeusedtoidentifymanyarmsof
theParetosetbutitisnotalwayspossibletoidentifytheentireParetosetusingthistechnique(see
e.g[7]forweightedsumwithafamilyofweightsvectors). Inadifferentdirection,theauthorsof
[16]introducethefeasiblearmidentificationproblem,inwhichthegoalistoidentifythesetofarms
whosemeanvectorsbelongtoaknownpolyhedronP ⊂RD. Inafollowupwork[17],theypropose
afixed-confidencealgorithmforfindingfeasiblearmsthatfurthermaximizeagivenweightedsum
oftheobjectives. Inclinicaltrials,thiscouldbeusedtofindtreatmentsmaximizingefficacy(ora
weightedsumofdifferentefficacyindicators),undertheconstraintthatthetoxicityremainsbelowa
threshold. However,inthepresenceofmultipleindicatorsofbiologicalefficacy,choosingtheweights
maybedifficult,andanexaminationoftheParetosetcouldbemoresuitable. Finally,somepapers
considerextensionsoftheParetooptimalitycondition. Theauthorsof[1]tackletheidentificationof
thesetofnon-dominatedarmsofanypartialorderdefinedbyanRD polyhedralorderingcone(the
usualParetodominancecorrespondstousingtheconedefinedbythepositiveorthantRD),andthey
+
provideworst-casesamplecomplexityinthePACsetting. Theworkof[3]studiestheidentification
ofthesetofnon-dominatedelementsinapartiallyorderedsetundertheduelingbanditsetting,in
whichtheobservationsconsistsinpairwisecomparisonbetweenarms.
Outlineandcontributions First,weformalizeinSection2differentrelaxationsofthePSIproblem:
ε -PSI,asintroducedby[4],ε ,ε -PSI,ofwhichaparticularcasewasstudiedby[27]andε -PSI-k,
1 1 2 1
a novel relaxation that takes as input an upper bound k on the maximum number of ε -optimal
1
armsthatcanbereturned. Then,weintroduceinSection3AdaptiveParetoExploration,asimple,
adaptivesamplingrulewhichcansimultaneouslytackleallthreerelaxations,whencoupledwithan
appropriatestoppingrulethatwedefineforeachofthem. InSection4,weprovehigh-probability
upperboundsonthesamplecomplexityofAPEunderdifferentstoppingrules. Forε -PSI,ourbound
1
slightlyimprovesuponthestate-of-the-art. Ourstrongestresultistheboundforε -PSI-k,which
1
leadstoanewnotionofsub-optimalitygap,quantifyingthereductioninsamplecomplexitythatis
obtained. Then,Section5presentstheresultofanumericalstudyonsyntheticdatasets,oneofthem
beinginspiredbyaCovid-19vaccineclinicaltrial. Itshowcasesthegoodempiricalperformanceof
APEcomparedtoexistingalgorithms,andillustratestheimpactofthedifferentrelaxations.
2 ProblemSetting
In this section, we introduce the Pareto Set Identification (PSI) problem and its relaxations. Fix
K,D ∈ N⋆. Let ν ,...,ν be distributions over RD with means µ ,...,µ ∈ RD. Let A :=
1 K 1 K
[K]:={1,...,K}denotethesetofarms. Letν :=(ν ,...,ν )andX :=(µ ,...,µ ). Weuse
1 K 1 K
boldfacedsymbolsforRD elements. LettingX∈RD,u∈R,foranyd∈{1,...,D},Xddenotes
thed-thcoordinateofXandX+u:=(X1+u,...,XD+u). Inthesequel,wewillassumethat
ν ,...,ν have1-subgaussianmarginals2.
1 K
Definition1. Giventwoarmsi,j ∈A,iisweakly(Pareto)dominatedbyj (denotedbyµ ≤µ )
i j
ifforanyd ∈ {1,...,D},µd ≤ µd. Thearmiis(Pareto)dominatedbyj (µ ⪯ µ ori ⪯ j)ifi
i j i j
isweaklydominatedbyj andthereexistsd∈{1,...,D}suchthatµd <µd. Thearmiisstrictly
i j
(Pareto)dominatedbyj (µ ≺µ ori≺j)ifforanyd∈{1,...,D},µd <µd.
i j i j
Forε∈RD,theε-ParetosetS⋆(X)isthesetofε-Paretooptimalarms,thatis:
+ ε
S⋆(X):={i∈As.t∄j ∈A:µ +ε≺µ }.
ε i j
Inparticular,S⋆(X)iscalledtheParetosetandwewillsimplywriteS⋆(X)todenoteS⋆(X). When
0 0
itisclearfromthecontext,wewriteS⋆(orS⋆)todenoteS⋆(X)(orS⋆(X)). Byabuseofnotation
ε ε
wewriteS⋆whenε∈R+todenoteS⋆,withε:=(ε,...,ε).
ε ε
Ineachroundt=1,2,...,theagentchoosesanarmA andobservesanindependentdrawX ∼ν
t t At
withE(X ) = µ . WedenotebyP thelawofthestochasticprocess(X ) andbyE ,the
At At ν t t≥1 ν
2ArandomvariableXisσ−subgaussianifforanyλ∈R,E(exp(λ(X−E(X)))≤exp(λ2σ2).
2
3

expectationunderP . LetF := σ(A ,X ,...,A ,X )theσ-algebrarepresentingthehistoryof
|     |     | ν   |     | t 1 1 |     | t t |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
theprocess. AnalgorithmforPSIconsistsin: i)asamplingrulewhichdetermineswhicharmto
sampleattimetbasedonhistoryuptotimet−1,ii)astoppingruleτ whichisastoppingtime
w.r.tthefiltration(F ) andiii)arecommendationrulewhichisaF -measurablerandomsetS(cid:98)τ
|     |     | t   | t≥1 |     |     |     | τ   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
representingtheguessofthelearner. Thegoalofthelearneristomakeacorrectguesswithhigh
probability,usingasfewsamplesτ aspossible. Beforeformalizingthis,weintroducethedifferent
notionofcorrectnessconsideredinthiswork,dependingonparametersε ≥,ε ≥0andk ∈[K].
|     |     |     |     |     |     |     |     | 1 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Ourfirstcriterionistheoneconsideredby[4].
| Definition2. |     | S(cid:98)⊂Aiscorrectforε |     | -PSIifS⋆ | ⊂S(cid:98)⊂S | ⋆ . |     |     |     |
| ------------ | --- | ------------------------ | --- | -------- | ------------ | --- | --- | --- | --- |
|              |     |                          |     | 1        |              | ε 1 |     |     |     |
Tointroduceoursecondcriterion,weneedthefollowingdefinition.
Definition3. Letε ,ε ≥0. AsubsetS ⊂Aisan(ε ,ε )-coveroftheParetosetif: S ⊂S⋆ and
|     |     | 1   | 2   |     |     | 1 2 |     |     | ε1  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
S⋆or∃j
| foranyi∈/ | S eitheri∈/ |     |     | ∈S suchthatµ |     | ≺µ +ε 2 | .   |     |     |
| --------- | ----------- | --- | --- | ------------ | --- | ------- | --- | --- | --- |
i j
Theε-accuratesetof[27]isaparticularcaseof(ε ,ε )-coverforwhichε = ε = ε. Allowing
|     |     |     |     |     |     | 1 2 |     | 1 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ε ̸=ε generalizesthenotionofε-correctsetandcanbeuseful,e.g.,inscenarioswhenwewantto
1 2
identifytheexactParetoset(settingε =0)butallowsomeoptimalarmstobediscardediftheyare
1
tooclose(parameterizedbyε 2 )toanotheroptimalarmalreadyreturned. Wenotehoweverthatthe
sparsecoverof[4]isan(ε,ε)-coverwithandadditionalconditionthatthearmsinthereturnedset
shouldnotbetooclosetoeachover. Identifyingasparsecoverfromsamplesrequiresinparticularto
| identifyS⋆ | henceitcannotbeseenasarelaxationofε |     |     |     |     | -PSI. |     |     |     |
| ---------- | ----------------------------------- | --- | --- | --- | --- | ----- | --- | --- | --- |
|            | ε1                                  |     |     |     |     | 1     |     |     |     |
Definition4. S(cid:98)⊂Aiscorrectfor(ε ,ε )-PSIifitisan(ε ,ε )-coveroftheParetoset.
|     |     |     |     | 1 2 |     | 1   | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
S(cid:98) ⊂ A is correct for ε -PSI-k if either |S(cid:98)|= k and S(cid:98) ⊂ S ⋆ or |S(cid:98)|< k and
| Definition | 5.  |     |     | 1   |     | i)  |     | ε ii) |     |
| ---------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
1
| S⋆ ⊂S(cid:98)⊂S⋆ |     | holds. |     |     |     |     |     |     |     |
| ---------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
ε1
Givenaspecifiedobjective(ε 1 -PSI,(ε 1 ,ε 2 )-PSIorε 1 -PSI-k),andatargetriskparameterδ ∈(0,1),
thegoaloftheagentistobuildaδ-correctalgorithm,thatistoguaranteethatwithprobabilitylarger
than1−δ,herguessS(cid:98)τ iscorrectforthegivenobjective,whileminimizingthenumberofsamplesτ
neededtomaketheguess,calledthesamplecomplexity.
Wenowintroducetwoimportantquantitiestocharacterizethe(Pareto)optimalityorsub-optimality
| ofthearms. | Foranytwoarmsi,j,welet |        |     |              |     |          |              |     |     |
| ---------- | ---------------------- | ------ | --- | ------------ | --- | -------- | ------------ | --- | --- |
|            |                        | m(i,j) | :=  | min (µd−µd), | and | M(i,j):= | max (µd−µd), |     |     |
|            |                        |        |     | j            | i   |          |              | i j |     |
|            |                        |        |     | 1≤d≤D        |     |          | 1≤d≤D        |     |     |
whichhavethefollowinginterpretation. Ifi⪯j,m(i,j)istheminimalquantityα≥0thatshould
beaddedcomponent-wisetoµ sothatµ +α ⊀ µ , α := (α,...,α). Moreover, m(i,j) > 0
|     |     |     |     | i i |     | j   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
⊀
if and only if i ≺ j. Then, for any arms i,j, if i j, M(i,j) is the minimum quantity α′ such
µ ≤µ +α′,α′ :=(α′,...,α′). WeremarkthatM(i,j)<0ifandonlyifi≺j. Ouralgorithms,
i j
presentedinthenextsection,relyonconfidenceintervalsonthesequantities.
3 AdaptiveParetoExploration
Wedescribeinthissectionoursamplingrule,AdaptiveParetoExploration,andpresentthreestopping
andrecommendationrulestowhichitcanbecombinedtosolveeachoftheproposedrelaxation.
(cid:80)t−11(A
LetT k (t) := s = k)bethenumberoftimesarmk hasbeenpulleduptoroundtand
s=1
µ (t):=T (t)−1(cid:80)Tk(t)X theempiricalmeanofthisarmattimet,whereX denotesthes-th
| (cid:98)k        | k                          | s=1        | k,s |                   |          |                             |     | k,s |     |
| ---------------- | -------------------------- | ---------- | --- | ----------------- | -------- | --------------------------- | --- | --- | --- |
| observationdrawn |                            | i.i.dfromν |     | k . Foranyarmsi,j | ∈A,welet |                             |     |     |     |
|                  | m(i,j,t):=min(µd(t)−µd(t)) |            |     |                   |          | M(i,j,t):=max(µd(t)−µd(t)). |     |     |     |
and
|     |     |     | d   | (cid:98)j (cid:98)i |     |     | d (cid:98)i | j   |     |
| --- | --- | --- | --- | ------------------- | --- | --- | ----------- | --- | --- |
TheempiricalParetosetisdefinedas
|     |     |     | S(t) | := {i∈A:∄j | ∈A:µ                | (t)≺µ     | (t)},     |     |     |
| --- | --- | --- | ---- | ---------- | ------------------- | --------- | --------- | --- | --- |
|     |     |     |      |            |                     | (cid:98)i | (cid:98)j |     |     |
|     |     |     |      | = {i∈A:∀j  | ∈A\{i},M(i,j,t)>0}. |           |           |     |     |
4

3.1 Genericalgorithm(s)
AdaptiveParetoExplorationreliesonalower/upperconfidenceboundapproach,similartosingle-
objectiveBAIalgorithmslikeUGapEc[9],LUCB[15]andLUCB++[26]. Thesethreealgorithms
identifyineach roundtwocontentiousarms: b : acurrent guessfortheoptimalarm (defined as
t
theempiricalbestarmorthearmwiththesmallestupperboundonitssub-optimalitygap),c : a
t
contenderofthisarm;thearmwhichisthemostlikelytooutperformb t (inallthreealgorithms,it
isthearmwiththelargestupperconfidenceboundin[K]\{b }). Then,eitherbotharmsarepulled
t
(LUCB,LUCB++)ortheleastexploredamongb andc ispulled(UGapEc). Theoriginalityofour
|     |     |     |     | t t |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
samplingruleliesinhowtoappropriatelydefineb andc forthemulti-objectivesetting. Todefine
|     |     |     |     | t t |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
those,wesupposethatthereexistsconfidenceintervals[Ld (t,δ),Ud (t,δ)]onthedifferenceof
i,j i,j
expectedvaluesforeachpairofarms(i,j)andeachobjectived∈D,suchthatintroducing
|     | K                 | D                |     |           |         |     | ∞        |     |
| --- | ----------------- | ---------------- | --- | --------- | ------- | --- | -------- | --- |
|     | (cid:92) (cid:92) | (cid:92) (cid:8) |     |           | (cid:9) |     | (cid:92) |     |
| E   | :=                | Ld (t,δ)≤µd−µd   |     | ≤Ud (t,δ) | and     | E = | E ,      | (1) |
| t   |                   | i,j              | i   | j i,j     |         |     | t        |     |
|     | i=1j̸=id=1        |                  |     |           |         |     | t=1      |     |
wehaveP(E)≥1−δ.ConcretechoicesoftheseconfidenceintervalswillbediscussedinSection3.2.
Toeasethenotation,wedropthedependencyinδintheconfidenceintervalsandfurtherdefine
|     | M−(i,j,t) | := maxLd   | (t) | M+(i,j,t):=maxUd       |     |     | (t) |     |
| --- | --------- | ---------- | --- | ---------------------- | --- | --- | --- | --- |
|     |           |            | i,j | and                    |     |     | i,j | (2) |
|     |           |            | d   |                        |     | d   |     |     |
|     | m−(i,j,t) | −M+(i,j,t) |     | m+(i,j,t):=−M−(i,j,t). |     |     |     |     |
|     |           | :=         |     | and                    |     |     |     | (3) |
M−(i,j,t)
Lemma 1. For any round t ≥ 1, if E t holds, then for any i,j ∈ A, ≤ M(i,j) ≤
M+(i,j,t)andm−(i,j,t)≤m(i,j)≤m+(i,j,t).
NotingthatS⋆ ={i∈A:∀j ̸=i,M(i,j)+ε >0},wedefinethefollowingsetofarmsthatare
|     | ε1  |     |     | 1   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
likelytobeε -Paretooptimal:
1
|     | OPTε1(t):={i∈A:∀j |     |     | ∈A\{i},M−(i,j,t)+ε |     | >0}. |     |     |
| --- | ----------------- | --- | --- | ------------------ | --- | ---- | --- | --- |
1
Samplingrule Inroundt,AdaptiveParetoExplorationsamplesa ,theleastpulledarmamong
t
| twocandidatearmsb | andc | givenby |        |               |     |     |     |     |
| ----------------- | ---- | ------- | ------ | ------------- | --- | --- | --- | --- |
|                   | t    | t       |        |               |     |     |     |     |
|                   |      | b :=    | argmax | minM+(i,j,t), |     |     |     |     |
t
|     |     |      | i∈A\OPTε1(t) | j̸=i  |     |     |     |     |
| --- | --- | ---- | ------------ | ----- | --- | --- | --- | --- |
|     |     | c := | argminM−(b   | ,j,t) |     |     |     |     |
|     |     | t    |              | t     |     |     |     |     |
j̸=bt
Theintuitionfortheirdefinitionisthefollowing. Lettingibeafixedarm, notethatM(i,j) > 0
forsomej,ifandonlyifthereexistsacomponentdsuchthatµd > µd i.eiisnotdominatedby
i j
j. Moreover,thelargerM(i,j),themoreiisnon-dominatedbyj inthesensethatthereexistsd
such that µd ≫ µd. Therefore, i is strictly optimal if and only if for all j ̸= i, M(i,j) > 0 i.e
i j
α :=min M(i,j)>0. Andthelargerα ,themoreilooksoptimalinthesensethatforeacharm
| i j̸=i |     |     | i   |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
j ̸=i,thereexistsacomponentd j forwhichiiswaybetterthanj. Astheα i areunknown,wedefine
b asthemaximizerofanoptimisticestimateoftheα ’s. Wefurtherrestrictthemaximizationtoarms
t i
forwhichwearenotalreadyconvincedthattheyareoptimal(byLemma1,thearmsinOPTε1(t)
are(nearly)ParetooptimalontheeventE). Then,wenotethatforafixedarmi,M(i,j)<0ifand
onlyifiisstrictlydominatedbyj. AndthesmallerM(i,j),themorej isclosetodominatei(or
foranycomponentd,µd−µdissmall(ornegative).
largelydominatesit): Thus,forafixedarmi,
|     |     |     |     | i j |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
argmin M(i,j)canbeseenasthearmwhichistheclosesttodominatei(orwhichdominatesit
j̸=i
bythelargestmargin). ByminimizingalowerconfidenceboundontheunknownquantityM(b t ,j),
ourcontenderc canbeinterpretedasthearmwhichisthemostlikelytobe(closeto)dominatingb .
|     | t   |     |     |     |     |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Gatheringinformationonbothb andc canbeusefultocheckwhetherb canindeedbeoptimal.
|     |     | t   | t   |     |     | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Interestingly,weshowinAppendixEthatforD =1,oursamplingruleisclosebutnotidenticalto
thesamplingrulesusedbyexistingconfidence-basedbestarmidentificationalgorithms.
5

Stoppingcondition Recommendation Objective
τ Zε1(t)>0 ∧Zε1(t)>0 O(τ ) ε -PSI
τ ε1 Zε1, 1 ε2(t)>0 ∧Z 2 ε1,ε2(t)>0 OPTε1( ε τ 1 ) (ε , 1 ε )-PSI
ε1,ε2 1 2 ε1,ε2 1 2
τk |OPTε1(t)|≥k OPTε1(τk) ε -PSI-k
1
Table1: Stoppingconditionsandassociatedrecommendation
Stoppingandrecommendationrule(s) Dependingontheobjective,AdaptiveParetoExploration
canbepluggedinwithdifferentstoppingrules,thataresummarizedinTable1withtheirassociated
recommendations. Todefinethose,wedefineforalli∈A,ε ,ε ≥0,
1 2
gε2(t):=maxm−(i,j,t)+ε 1{j ∈OPTε1(t)} and hε1(t):=minM−(i,j,t)+ε .
i 2 i 1
j̸=i j̸=i
andletg (t):=g0(t). Introducing
i i
Zε1(t):= min hε1(t), and Zε1(t):= min max(g (t),hε1(t)),
1 i 2 i i
i∈S(t) i∈S(t)c
forε -PSI,ourstoppingruleisτ := inf{t ≥ K : Zε1(t) > 0∧Zε1(t) > 0}andtheassociated
1 ε1 1 2
recommendationisO(τ )where
ε1
O(t):=S(t)∪{i∈S(t)c :∄j ̸=i:m−(i,j,t)>0}
consistsofthecurrentempiricalParetosetplussomeadditionalarmsthathavenotyetbeenformally
identifiedassub-optimal. Thosearmsshouldbe(ε )-Paretooptimal.
1
For(ε ,ε )-PSIwedefineasimilarstoppingruleτ wherethestoppingstatisticsarerespectively
1 2 ε1,ε2
replacedwith
Zε1,ε2(t):= min max(gε2(t),hε1(t)) andZε1,ε2(t):= min max(gε2(t),hε1(t))
1 i i 2 i i
i∈S(t) i∈S(t)c
withtheconventionmin =+∞,andtherecommendationisOPTε1(τ ).
∅ ε1,ε2
Totackletheε -PSI-krelaxation,weproposetocoupleτ withanadditionalstoppingcondition
checkingwheth 1 erOPTε1(t)alreadycontainskarms. That ε is 1 ,westopatτk :=min (cid:0) τ ,τk(cid:1) where
ε1 ε1
τk :=inf{t≥K :|OPTε1(t)|≥k}withassociatedrecommendationOPTε1(τk). Dependingof
thereasonforstopping(τ orτk),wefollowthecorrespondingrecommendation.
ε1
Lemma2. AssumeE holds. Forε -PSI(resp. (ε ,ε )-PSI,ε -PSI-k),AdaptiveParetoExploration
1 1 2 1
combinedwiththestoppingruleτ (resp. τ ,resp. τk)outputsacorrectsubset.
ε1 ε1,ε2 ε1
Remark1. Wedecoupledthepresentationofthesamplingruletothatofthe“sequentialtesting”
aspect(stoppingandrecommendation). Wecouldevengofurtherandobservethatmultipletests
couldactuallyberuninparallel,forfree. IfwecollectsampleswithAPE(whichonlydependsonε ),
1
wheneveroneofthethreestoppingconditionsgiveninTable1triggers,foranyvaluesofε ork,we
2
candecidetostopandmakethecorrespondingrecommendationorcontinueandwaitforanother
“moreinteresting”stoppingconditiontobesatisfied. IfE holds,arecommendationmadeatanysuch
timewillbecorrectfortheobjectiveassociatedtothestoppingcriterion(thirdcolumninTable1).
3.2 Ourinstantiation
Weproposetoinstantiatethealgorithmswithconfidenceintervalonthedifferenceofpairofarms.For
anypairi,j ∈A,wedefineafunctionβ suchthatforanyd∈[D],Ud (t)=µd(t)−µd(t)+β (t)
i,j i,j (cid:98)i (cid:98)j i,j
andLd (t)=µd(t)−µd(t)−β (t). Wetakefrom[20]thefollowingconfidencebonusfortime-
i,j (cid:98)i (cid:98)j i,j
uniformconcentration:
(cid:118)
(cid:117)
(cid:117)
 (cid:32)
log
(cid:0)K1 (cid:1)(cid:33)
(cid:88)

(cid:88) 1

β
i,j
(t):=2(cid:117) (cid:116)Cg
2
δ + log(4+log(T
a
(t)))
T (t)
, (4)
a
a∈{i,j} a∈{i,j}
where K := K(K −1)D/2 and Cg ≈ x+log(x) is a calibration function. They result in the
1
simpleexpressionsM±(i,j,t) = M(i,j,t)±β (t)andm±(i,j,t) = m(i,j,t)±β (t). Asan
i,j i,j
6

Algorithm1:ε -APE-k: AdaptiveParetoExplorationforε -PSI-k
1 1
Data: parameterε ≥0,k ∈[K]
1
initialize: sampleeacharmonce,sett=K,T (K)=1foranyi∈A
i
fort=K+1,...,do
S(t)={i∈A:∀j ∈A\{i},M(i,j,t)>0};
OPTε1(t)={i∈A:∀j ∈A\{i},M(i,j,t)−β (t)+ε >0};
i,j 1
if|OPTε1(t)|≥kthen
breakandoutputOPTε1(t)
ifZε1(t)>0 ∧Zε1(t)>0then
b 1 reakandoutp 2 utO(t)=S(t) (cid:83) {i∈S(t)c :∄j ̸=i:m(i,j,t)−β (t)>0}
i,j
b :=argmax min [M(i,j,t)+β (t)];
t i∈A\OPTε1(t) j̸=i i,j
c :=argmin [M(b ,j,t)−β (t)];
t i̸=bt t bt,j
samplea :=argmin T (t);
t i∈{bt,ct} i
example,westateinAlgorithm1thepseudo-codeofAPEcombinedthestoppingrulesuitedforthe
k-relaxationofε -PSI,whichwerefertoasε -APE-k.
1 1
In Appendix F, we also study a different instantiation based on confidence bounds of the form
U (t) = U (t)−L (t)where[L (t),U (t)]isaconfidenceintervalonµ . Thisistheapproach
i,j i j i i i
followedbyLUCBforD =1andpriorworkonParetoidentification[4,27]. Inpracticeweadvocate
theuseofthepairwiseconfidenceintervalsdefinedabove,evenifourcurrentanalysisdoesnotallow
toquantifytheirimprovement. FortheLUCB-likeinstantiation,wefurtherderiveinAppendixFan
upperboundontheexpectedstoppingtimeofAPEforthedifferentstoppingrules.
4 Theoreticalanalysis
Inthissection,westateourmaintheoremonthesamplecomplexityofouralgorithmsandgivea
sketchofitsproof.
Firstletusintroducesomequantitiesthatareneededtostatethetheorem. Thesamplecomplexity
ofthealgorithmproposedby[4]for(ε )-Paretosetidentificationscalesasasumoverthearmsi
1
of1/(∆ ∨ε )2where∆ iscalledthesub-optimalitygapofarmiandisdefinedasfollows. Fora
i 1 i
sub-optimalarmi∈/ S⋆(X),
∆ :=maxm(i,j),
i
j∈S⋆
whichisthesmallestquantitythatshouldbeaddedcomponent-wisetoµ tomakeiappearPareto
i
optimalw.r.t{µ :i∈A}. ForaParetooptimalarmi∈S⋆(X),thedefinitionismoreinvolved:
i
(cid:26) min ∆ ifS⋆ :={i}
∆ := j∈A\{i} j
i min(δ+,δ−) else,
i i
where
δ+ := min min(M(i,j),M(j,i))and δ− := min {(M(j,i))++∆ }.
i i j
j∈S⋆\{i} j∈A\S⋆
Forx ∈ R,(x)+ := max(x,0). Wealsointroducesomeadditionalnotionneededtoexpressthe
contributionofthek-relaxation. Let1≤k ≤K. Foranyarmi,letω =min M(i,j)anddefine
i j̸=i
ωk :=m k axω , S⋆,k :=ar 1 g . m ..k axω ,
i i
i∈A i∈A
withthek-thmaxandfirsttok-thargmaxoperators. Observethatωk >0ifandonlyif|S⋆(X)|≥k.
Theorem1. Fixariskparameterδ ∈(0,1),ε ≥0,letk ≤K andν abanditwith1-subgaussian
1
marginals. Withprobabilityatleast1−δ, ε -APE-k recommendsacorrectsetfortheε -PSI-k
1 1
objectiveandstopsafteratmost
(cid:18) (cid:18) (cid:19)(cid:19)
(cid:88) 88 2K(K−1)D 12e
log log ,
a∈A
∆(cid:101)2
a
δ ∆(cid:101)a
samples,whereforeacha∈A,∆(cid:101)a :=max(∆
a
,ε
1
,ωk).
7

First,whenk =K,observingthatε -APE-K providesaδ-correctalgorithmforε -PSI,ourbound
|     |     |     |     | 1   |     |     |     |     | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
improves the result of [4] for the ε -PSI problem in terms of constant multiplicative factors and
1
loglog∆−1 terms instead of log∆−2. It nearly matches the lower bound of Auer et al.[4] for
the ε -PSI problem (Theorem 17 therein). It also shows the impact of the k-relaxation on the
1
Inparticular,wecanremarkthatforanyarmi∈S⋆\S⋆,k,max(∆
| samplecomplexity. |     |     |     |     |     |     |     |     | i ,ω k | )=ω k . |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------- |
Intuitively, it says that we shouldn’t pay more than the cost of identifying the k-th optimal arm,
ordered by the ω′s. A similar result has been obtained for the any k−sized subset of the best m
i
problem[25]. Buttheauthorshaveshowntherelaxationonlyforthebestmarmswhileourresult
showsthateventhesub-optimalarmsshouldbesampledless.
InAppendixD,weprovethefollowinglowerboundshowingthatinsomescenarios,ε -APE-kis
1
optimalforε 1 -PSI-k,uptoDlog(K)andconstantmultiplicativeterms. Wenotethatforε 1 -PSIa
| lowerboundfeaturingthegaps∆ |     |     |     | andε | wasalreadyderivedbyAueretal. |     |     |     | [4]. |     |
| --------------------------- | --- | --- | --- | ---- | ---------------------------- | --- | --- | --- | ---- | --- |
a 1
with|S⋆|=p≥3suchthatfork
| Theorem2. | Thereexistsabanditinstanceν |     |     |     |     |     |     |     | ∈{p,p−1,p−2} |     |
| --------- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- |
anyδ-correctalgorithmfor0-PSI-kverifies
|     |     |     |     |       | (cid:18) | (cid:19) K |     |     |     |     |
| --- | --- | --- | --- | ----- | -------- | ---------- | --- | --- | --- | --- |
|     |     |     |     |       | 1        | 1 (cid:88) | 1   |     |     |     |
|     |     |     | E   | (τ )≥ | log      |            |     | ,   |     |     |
ν δ
|         |     |          |                                  |     | D   | δ   | (∆k)2 |     |     |     |
| ------- | --- | -------- | -------------------------------- | --- | --- | --- | ----- | --- | --- | --- |
|         |     |          |                                  |     |     | a=1 | a     |     |     |     |
| where∆k | :=∆ | +ωk andτ | isthestoppingtimeofthealgorithm. |     |     |     |       |     |     |     |
|         | a   | a        | δ                                |     |     |     |       |     |     |     |
InAppendixC,weprovethatTheorem1withouttheωk termsalsoholdsfor(ε ,ε )-APE.This
1 2
doesnotjustifiesthereductioninsamplecomplexitywhensettingε >0in(ε ,ε )-PSIobservedin
|     |     |     |     |     |     |     |     | 2   | 1 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ourexperimentsbutitatleastguaranteesthattheε -relaxationdoesn’tmakethingsworse.
2
Furthermore,sinceouralgorithmallowsε 1 =0,itisalsoanalgorithmforBAIwhenD =1,ε 1 =0.
WeproveinAppendixEthatinthiscase,thegaps∆′smatchestheclassicalgapsinBAI[2,18]and
i
wederiveitssamplecomplexityfromTheorem1showingthatitissimilarintheorytoUGap[9],
LUCB[15]andLUCB++[26]buthavebetterempiricalperformance.
SketchofproofofTheorem1 UsingProposition24of[20]wefirstprovethatthechoiceofβ i,j
in(4)yieldsP(E)≥1−δforthegoodeventE definedin(1). CombiningthisresultwithLemma2,
| yieldsthatε | -APE-kiscorrectwithprobabilityatleast1−δ. |     |     |     |     |     |     |     |     |     |
| ----------- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1
TheideaoftheremainingproofistoshowthatundertheeventE,ifAPEhasnotstoppedattheend
ofroundt,thentheselectedarma hasnotbeenexploredenough. Thefirstlemmashowingthisis
t
| specifictothestoppingruleτk |      |       | usedforε |              | -PSI-k. |     |        |     |       |     |
| --------------------------- | ---- | ----- | -------- | ------------ | ------- | --- | ------ | --- | ----- | --- |
|                             |      |       | ε1       |              | 1       |     |        |     |       |     |
|                             |      |       |          | holdsandt<τk |         |     | thenωk |     |       |     |
| Lemma3.                     | Letε | ≥andk | ∈[K].    | IfE          |         |     |        | ≤2β | (t).  |     |
|                             |      | 1     |          | t            |         | ε1  |        |     | at,at |     |
Thenexttwolemmasaremoregeneralastheyapplytodifferentstoppingrules.
=τk
Lemma4. Letε 1 ≥0. Letτ ε1 forsomek ∈[K]orτ =τ ε1,ε2 forsomeε 2 ≥0. IfE t holdsand
| t<τ then∆ |     | ≤2β (t). |     |     |     |     |     |     |     |     |
| --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
at at,at
Lemma5. Letε 1 ≥0andτ beasinLemma4. IfE t holdsandt<τ thenε 1 ≤2β at,at (t).
AscanbeseeninAppendixB,theproofsofthesethreelemmasheavilyrelyonthespecificdefinition
ofb t andc t . Inparticular,toproveLemma4and5,wefirstestablishthatwhent<τ,anyarmj ∈A
satisfiesm(b ,j,t)≤β (t).ThemostsophisticatedproofisthenthatofLemma4,whichrelieson
t bt,j
| acasedistinctionbasedonwhetherb |     |     |     | orc | belongstothesetofoptimalarms. |                  |     |          |                   |     |
| ------------------------------- | --- | --- | --- | --- | ----------------------------- | ---------------- | --- | -------- | ----------------- | --- |
|                                 |     |     |     | t   | t                             |                  |     |          |                   |     |
|                                 |     |     |     |     | k                             |                  |     |          | 2βTat (t),wherewe |     |
| Theselemmaspermittoshowthat,onE |     |     |     |     | t ift < τ ε                   | then∆(cid:101)at | <   | 2β at,at | (t) ≤             |     |
1
| defineβntobetheexpressionofβ |     |     |     | (t)whenT | (t)=T | (t)=n. |     | Thenwehave |     |     |
| ---------------------------- | --- | --- | --- | -------- | ----- | ------ | --- | ---------- | --- | --- |
|                              |     |     |     | i,j      | i     | j      |     |            |     |     |
∞
|     |     |          |     | (cid:88)(cid:88) | (cid:110) |      | (cid:110)       |       | (cid:111)(cid:111) |     |
| --- | --- | -------- | --- | ---------------- | --------- | ---- | --------------- | ----- | ------------------ | --- |
|     |     | τ k 1{E} | ≤   |                  | 1 {a      | =a}∧ | ∆(cid:101)a ≤2β | Ta(t) |                    |     |
|     |     | ε        |     |                  | t         |      |                 | a     |                    |     |
1
t=1a∈A
|     |     |     |     |     | (cid:110) |     | (cid:111) |     |     |     |
| --- | --- | --- | --- | --- | --------- | --- | --------- | --- | --- | --- |
(cid:88)
|     |     |     | ≤   | inf | n≥2:∆(cid:101)a | >2βn |     | .   |     | (5) |
| --- | --- | --- | --- | --- | --------------- | ---- | --- | --- | --- | --- |
a∈A
Acarefulupper-boundingoftheRHSof(5)completestheproofofTheorem1.
8

5 Experiments
WeevaluatetheperformanceofAdaptiveParetoExplorationonareal-worldscenarioandonsynthetic
randomBernoulliinstances. Forafaircomparison,Algorithm1of[4],referredtoasPSI-Unif-Elim
and APE are both run with our confidence bonuses β (t) on pairs of arms, which considerably
i,j
improvesingle-armconfidencebonuses3.Asanytimeconfidenceboundsareknowntobeconservative,
weuseK = 1in(4)insteadofitstheoreticalvaluecomingfromaunionbound. Still,inallour
1
experiments,theempiricalerrorprobabilitywas(significantly)smallerthanthetargetδ =0.1.
Real-worlddataset COV-BOOST[24]isphase2trialwhichwasconductedon2883participants
tomeasuretheimmunogenicityofdifferentCovid-19vaccinesasthirddose(booster)invarious
combinationsofinitiallyreceivedvaccines(firsttwodoses). Thisresultedinatotalof20vaccination
strategiesbeingassessed,eachofthemdefinedbythevaccinesreceivedasfirst,secondandthird
dose. Theauthorshavereportedtheaverageresponsesinducedbyeachcandidatestrategyoncohorts
ofparticipants,measuringseveralimmunogenicitymarkers. Fromthisstudy,weextractandprocess
the average response of each strategy to 3 specific immunogenicity indicators: two markers of
antibodyresponseandoneofthecellularresponse. Theoutcomesareassumedtohavealog-normal
distribution[24]. Weusetheaverage(log)outcomesandtheirvariancestosimulateamultivariate
GaussianbanditwithK =20,D =3. WegiveinAppendixH.2someadditionaldetailsaboutthe
processingofthedata,andreportthemeansandvariancesofeacharm. InAppendixH.1wefurther
explainhowAPEcanbesimplyadaptedwhenthemarginalsdistributionsofthearmshavedifferent
variances. Inthisexperiment,wesetε =0,δ =0.1andcomparePSI-Unif-Elimto0-APE-k(called
1
1e1
5.0
1.1
4.8
1.0 4.6
4.4
0.9
4.2
0.8 4.0
3.8
0.7
3.6
APE[k=1] APE[k=2*] APE[k=3] APE[k=20] PSI-Unif-Elim 7.5 8.0 8.5 9.0 9.5 10.0 10.5
anti-spike IgG
esnopser
ralullec
BNT/BNT/m1273
ChAd/ChAd/m1273
Figure1: OntheleftisthelogoftheempiricalsamplecomplexityofPSI-Unif-ElimandAPEonthe
real-worldscenarioplot(right)for2outofthe3immunogenicityindicators.
APE-kinthesequel)fordifferentvaluesofk. Theempiricaldistributionofthesamplecomplexityof
thealgorithms,averagedover2000independentruns,arereportedinFigure1. Theresultsareshown
inlog-scale(y-axisisthelogofthesamplecomplexity)tofitinthesamefigure. As|S⋆|=2,wefirst
observethat,withouttherelaxation(i.e. fork >3),APEoutperformsitsstate-of-the-artcompetitor
PSI-Unif-Elim. Moreoverfork = 1ork = 2, thesamplecomplexityofAPE-k issignificantly
reduced. Fork = 2whenthestoppingtimeτk isreachedsomesub-optimalarmshavepossibly
notyetbeenidentifiedassuch,whilefork =3,eveniftheoptimalarmshavebeenidentified,the
remainingarmshavetobesampledenoughtoensurethattheyaresub-optimalbeforestopping. This
explainsthegapinsamplecomplexitybetweenk = 2andk = 3. InAppendixH.3,wecompare
APEtoanadaptationofPSI-Unif-Elimforthek-relaxation,showingthatAPEisalwayspreferable.
Experiments on random instances To further illustrate the impact of the k-relaxation and to
strengthenthecomparisonwithPSI-Unif-Elim,weranthepreviousalgorithmson2000randomly
generatedmulti-variateBernoulliinstances,withK =5armsanddifferentvaluesofthedimension
D. We set δ = 0.1 and ε = 0.005 (to have reasonable running time). The averaged sample
1
complexitiesarereportedinTable2.WeobservethatAPE(withk =K)uses20to25%lesssamples
3Intheirexperiments,[4]alreadyproposedtheheuristicuseofconfidencebonusesofthisform
9

thanPSI-Unif-Elimandtendstobemoreefficientasthedimensionincreases(andlikelythesizeof
theParetoset,sincetheinstancesarerandomlygenerated). Wealsonotethatidentifyingak-sized
subsetoftheParetosetrequiresconsiderablylesssamplesthanexactPSI.InAppendixH.3wealso
provideexamplesofinstancesforwhichAPEtakesupto3timeslesssamplesthanPSI-Unif-Elim.
ε -APE-1 ε -APE-2 ε -APE-3 ε -APE-4 ε -APE-5 ε -PSI-Unif-Elim
1 1 1 1 1 1
D =2 811 39530 109020 145777 150844 190625
D =4 214 6410 19908 68061 124001 157584
D =8 119 204 405 1448 20336 27270
Table2: Averagesamplecomplexityover2000randomBernoulliinstanceswithK =5arms. On
averagethesizeoftheParetosetwas(2.295,4.0625,4.931)respectivelyforthedimensions2,4,8.
Toillustratetheimpactoftheε relaxation,settingε =0wereportthesamplecomplexityofAPE
2 1
associatedwiththestoppingtimeτ for20equallyspacedvaluesofε ∈[0.01,0.05],averaged
0,ε2 2
over2000randomBernoulliinstances. Figure2showsthedecreaseoftheaveragesamplecomplexity
whenε increases(left)andtheaverageratioofthesizeofthereturnedsettothesizeofthePareto
2
set(right). Notethatforε =0,wehaveO(τ )⊂S⋆. Theaveragesamplecomplexityreported
1 0,ε2
decreasesupto86%fortheinstancewithK =5,D =2whilethereturnedsetcontainsmorethan
90%oftheParetooptimalarms. InAppendixH.3,wefurtherillustratethebehaviorofAPEwiththe
ε relaxationonafixedinstanceindimension2.
2
(a)Averagesamplecomplexityvsε
2
(b)Averageratio|O(τ0,ε2 )|/|S⋆|vsε
2
Figure2: APEwithτ averagedover2000randomBernoulliinstancewithK =5arms.
0,ε2
6 Conclusionandperspective
We proposed and analyzed APE, an adaptive sampling rule in multi-variate bandits models that
whencoupledwithdifferentstoppingrulescantackledifferentrelaxationsofthefixed-confidence
ParetoSetIdentificationproblem. Ourexperimentsrevealedthegoodperformanceoftheresulting
algorithmscomparedtothestate-of-the-artPSIalgorithmaswellasthegreatreductionsinsample
complexity brought by the relaxations. In future work, we intend to make our algorithms more
practicalforpossibleapplicationstoclinicaltrials. Forthispurpose,asmeasuringefficacytakestime,
wewillinvestigateitsadaptationtoabatchsetting,following,e.g. theworkof[13]forBAI.Wewill
alsoinvestigatetheuseofAPEbeyondthefixed-confidencesetting,tothepossiblymorerealistic
fixed-budget[2]oranytimeexploration[14]settings. Tothebestofourknowledge,noalgorithm
existsintheliteratureforPSIinsuchsettings. Finally,followingtheworksof[28,4],wedefinedthe
ε ,ε relaxationswithscalarvalues,sothatthesameslackappliestoallcomponents. Althoughwe
1 2
couldeasilymodifyouralgorithmstotacklevectorialvaluesε ,ε ,sofarwecouldonlyprovea
1 2
dependenceonmin εdinthesamplecomplexity. Weintendtostudytherightquantificationinthe
d 1
samplecomplexitywhenε andε arevectorial.
1 2
10

AcknowledgmentsandDisclosureofFunding
CyrilleKoneisfundedbyanInria/InsermPhDgrant. EmilieKaufmannacknowledgesthesupportof
theFrenchNationalResearchAgencyundertheprojectsBOLD(ANR-19-CE23-0026-04)andFATE
(ANR22-CE23-0016-01).
References
[1] C.AraratandC.Tekin. Vectoroptimizationwithstochasticbanditfeedback. InProceedingsof
The26thInternationalConferenceonArtificialIntelligenceandStatistics,volume206,pages
2165–2190.PMLR,2023.
[2] J.-Y.AudibertandS.Bubeck. Bestarmidentificationinmulti-armedbandits. InCOLT-23th
ConferenceonLearningTheory-2010,page13p.,2010.
[3] J.AudiffrenandL.Ralaivola. Banditsduelingonpartiallyorderedsets. InAdvancesinNeural
InformationProcessingSystems,volume30.CurranAssociates,Inc.,2017.
[4] P.Auer,C.-K.Chiang,R.Ortner,andM.-M.Drugan. Paretofrontidentificationfromstochastic
banditfeedback. InProceedingsofthe19thInternationalConferenceonArtificialIntelligence
andStatistics,volume51,pages939–947.PMLR,2016.
[5] R.DegenneandW.M.Koolen. Pureexplorationwithmultiplecorrectanswers. InAdvancesin
NeuralInformationProcessingSystems,volume32.CurranAssociates,Inc.,2019.
[6] M.-M.DruganandA.Nowe. Designingmulti-objectivemulti-armedbanditsalgorithms:Astudy.
InThe2013InternationalJointConferenceonNeuralNetworks(IJCNN),pages1–8,2013.
[7] M.Ehrgott. MulticriteriaOptimization. Springer,secondeditionedition,2005.
[8] E.Even-Dar,S.Mannor,andY.Mansour. Actioneliminationandstoppingconditionsforthe
multi-armedbanditandreinforcementlearningproblems.JournalofMachineLearningResearch,
7(39):1079–1105,2006.
[9] V. Gabillon, M. Ghavamzadeh, and A. Lazaric. Best arm identification: A unified approach
tofixedbudgetandfixedconfidence. InAdvancesinNeuralInformationProcessingSystems,
volume25.CurranAssociates,Inc.,2012.
[10] A.GarivierandE.Kaufmann. Optimalbestarmidentificationwithfixedconfidence. In29th
AnnualConferenceonLearningTheory,volume49,pages998–1027.PMLR,2016.
[11] K.Jamieson,M.Malloy,R.Nowak,andS.Bubeck. lil’ucb: Anoptimalexplorationalgorithm
formulti-armedbandits. InProceedingsofThe27thConferenceonLearningTheory,volume35,
pages423–439.PMLR,2014.
[12] M. Jourdan, R. Degenne, D. Baudry, R. de Heide, and E. Kaufmann. Top two algorithms
revisited. In Advances in Neural Information Processing Systems, volume 35, pages 26791–
26803.CurranAssociates,Inc.,2022.
[13] K.-S.Jun,K.Jamieson,R.Nowak,andX.Zhu. Toparmidentificationinmulti-armedban-
dits with batch arm pulls. In Proceedings of the 19th International Conference on Artificial
IntelligenceandStatistics,volume51,pages139–148.PMLR,2016.
[14] K-S. Jun and R. Nowak. Anytime exploration for multi-armed bandits using confidence
information. In Proceedings of The 33rd International Conference on Machine Learning,
volume48,pages974–982.PMLR,2016.
[15] S.Kalyanakrishnan,A.Tewari,P.Auer,andP.Stone. PACsubsetselectioninstochasticmulti-
armedbandits. InProceedingsofthe29thInternationalCoferenceonInternationalConference
onMachineLearning,pages227–234.Omnipress,2012.
[16] J.Katz-SamuelsandC.Scott. Feasiblearmidentification. InProceedingsofthe35thInterna-
tionalConferenceonMachineLearning,volume80,pages2535–2543.PMLR,2018.
11

[17] J.Katz-SamuelsandC.Scott. Topfeasiblearmidentification. InProceedingsoftheTwenty-
Second International Conference on Artificial Intelligence and Statistics, pages 1593–1601.
PMLR,2019.
[18] E. Kaufmann, O. Cappé, and A. Garivier. On the complexity of best-arm identification in
multi-armedbanditmodels. JournalofMachineLearningResearch,17(1):1–42,2016.
[19] E.KaufmannandS.Kalyanakrishnan. Informationcomplexityinbanditsubsetselection. In
ConferenceOnLearningTheory,volume30.JMLR:WorkshopandConferenceProceedings,
2013.
[20] E.KaufmannandW.-M.Koolen. Mixturemartingalesrevisitedwithapplicationstosequential
testsandconfidenceintervals. JournalofMachineLearningResearch,22(246):1–44,2021.
[21] T.LattimoreandC.CSzepesvári. BanditAlgorithms. CambridgeUniversityPress,2020.
[22] R.Mehrotra,N.Xue,andM.Lalmas. Banditbasedoptimizationofmultipleobjectivesona
musicstreamingplatform.InProceedingsofthe26thACMSIGKDDInternationalConferenceon
KnowledgeDiscovery&DataMining,page3224–3233.AssociationforComputingMachinery,
2020.
[23] K.Miettinen. Nonlinearmultiobjectiveoptimization. InInternationalSeriesinOperations
ResearchandManagementScience,1998.
[24] A.-P.-S.Munro,L.Janani,V.Cornelius,andetal. SafetyandimmunogenicityofsevenCOVID-
19vaccinesasathirddose(booster)followingtwodosesofChAdOx1nCov-19orBNT162b2
intheUK(COV-BOOST):ablinded,multicentre,randomised,controlled,phase2trial. The
Lancet,398(10318):2258–2276,2021.
[25] A.RoyChaudhuriandS.Kalyanakrishnan. PACidentificationofmanygoodarmsinstochastic
multi-armedbandits. InProceedingsofthe36thInternationalConferenceonMachineLearning,
volume97,pages991–1000.PMLR,2019.
[26] M.Simchowitz,K.Jamieson,andB.Recht. Thesimulator: Understandingadaptivesampling
inthemoderate-confidenceregime. InProceedingsofthe2017ConferenceonLearningTheory,
volume65,pages1794–1834.PMLR,2017.
[27] M. Zuluaga, A. Krause, and M. Püschel. e-pal: An active learning approach to the multi-
objectiveoptimizationproblem. JournalofMachineLearningResearch,17(104):1–32,2016.
[28] M. Zuluaga, G. Sergent, A. Krause, and M. Püschel. Active learning for multi-objective
optimization. In Proceedings of the 30th International Conference on Machine Learning,
volume28,pages462–470.PMLR,2013.
12

Contents
1 Introduction 1
2 ProblemSetting 3
3 AdaptiveParetoExploration 4
3.1 Genericalgorithm(s) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2 Ourinstantiation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4 Theoreticalanalysis 7
5 Experiments 9
| 6 Conclusionandperspective             |     |     |     | 10  |
| -------------------------------------- | --- | --- | --- | --- |
| A Correctnessfordifferentstoppingrules |     |     |     | 14  |
A.1 ProofofLemma1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.2 ProofofLemma2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.3 Calibrationoftheconfidenceintervals . . . . . . . . . . . . . . . . . . . . . . . . 15
| B Samplecomplexityanalysis |     |     |     | 16  |
| -------------------------- | --- | --- | --- | --- |
B.1 ProofofLemma3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B.2 ProofofLemma4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B.3 ProofofLemma5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.4 Auxiliaryresults. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
| C Algorithmforfindingan(ε | ,ε )-cover |     |     | 21  |
| ------------------------- | ---------- | --- | --- | --- |
1 2
| D LowerBound                             |     |     |     | 23  |
| ---------------------------------------- | --- | --- | --- | --- |
| E BestArmIdentification                  |     |     |     | 25  |
| F LUCB1-likeinstantiationofAPE           |     |     |     | 27  |
| G TechnicalLemmas                        |     |     |     | 31  |
| H ImplementationandAdditionalExperiments |     |     |     | 34  |
H.1 Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
H.2 Dataprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
H.3 Additionalexperiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
Additionalexperimentsforε -APE-k
| H.3.1 | 1   | . . . . . . . | . . . . . . . | . . . . . . 36 |
| ----- | --- | ------------- | ------------- | -------------- |
H.3.2 (ε 1 ,ε 2 )-APE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
H.3.3 Comparingε -APE-ktoanadaptationofPSI-Unif-Elim . . . . . . . . . . 39
1
H.3.4 ComparisontosomeBAIalgorithms. . . . . . . . . . . . . . . . . . . . . 40
13

OutlineandNotation
In this section, we provide an outline of the supplemental material and define some additional
notation. AppendixAprovesthecorrectnessofoursalgorithmsandsomeconcentrationlemmas.
InAppendixB,weproveTheorem1andthelemmasusedinitsproof. InAppendixCweanalyze
thecorrectnessandsamplecomplexityofAPEassociatedtothestoppingtimeτ ε1,ε2 . AppendixD
describesourworst-caselowerboundandinAppendixEwerelateouralgorithmtootheralgorithms
forBAI.InAppendixFwederiveanupper-boundontheexpectationofthesamplecomplexityof
ε -APE-k withaLUCB1-likeinstantiationandinAppendixGwerecallorprovesometechnical
1
lemmasthatareusedinthemainproofs. Finally,inAppendixHwegivefurtherdetailsaboutthe
experimentstogetherwithadditionalexperimentalresults.
| Notation |     | Type  |     | Description                   |     |     |         |     |     |
| -------- | --- | ----- | --- | ----------------------------- | --- | --- | ------- | --- | --- |
| A        |     |       |     | Setofarms                     |     |     |         |     |     |
| K        |     | N⋆    |     | Numberofarms                  |     |     |         |     |     |
| D        |     | N⋆    |     | Dimensionornumberofattributes |     |     |         |     |     |
| [n]      |     |       |     | {1,...,n}                     |     |     |         |     |     |
| m        |     | A2 →R |     | m(i,j):=min{µd−µd             |     |     | :d∈[D]} |     |     |
|          |     |       |     |                               |     | j   | i       |     |     |
| M        |     | A2 →R |     | M(i,j):=max{µd−µd             |     |     | :d∈[D]} |     |     |
|          |     |       |     |                               |     | i   | j       |     |     |
| (x)+     |     | R→R+  |     |                               |     |     |         |     |     |
max(0,x)
| µ   |     | RD  |     | Meanofarma∈A |     |     |     |     |     |
| --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
a
| S⋆  |     |     |     | {i∈A:∄j | ̸=i∈A:µ |     | +ε≺µ | }   |     |
| --- | --- | --- | --- | ------- | ------- | --- | ---- | --- | --- |
| ε   |     |     |     |         |         |     | i    | j   |     |
Table3: Tableofnotation.
A Correctnessfordifferentstoppingrules
Inthissection,wegatherandproveresultsthatarerelatedtothecorrectnessofouralgorithms,either
fortheirgenericform(Lemma1andLemma2)orsomespecificcalibration. Werecallthedefinition
oftheevents
|     |     | (cid:92) K (cid:92) (cid:92) D |           |     |     |         |         | (cid:92) ∞ |     |
| --- | --- | ------------------------------ | --------- | --- | --- | ------- | ------- | ---------- | --- |
|     |     | (cid:8) Ld                     | (t)≤µd−µd |     | ≤Ud | (cid:9) |         |            |     |
|     |     | E t =                          |           |     |     | (t)     | and E = | E t        | .   |
|     |     |                                | i,j       | i j | i,j |         |         |            |     |
|     |     | i=1j̸=id=1                     |           |     |     |         |         | t=1        |     |
A.1 ProofofLemma1
Lemma 1. For any round t ≥ 1, if E holds, then for any i,j ∈ A, M−(i,j,t) ≤ M(i,j) ≤
t
M+(i,j,t)andm−(i,j,t)≤m(i,j)≤m+(i,j,t).
| ProofofLemma1. |     | ThisresultsimplyfollowsfromthedefinitionofE |     |     |     |     | . Since |     |     |
| -------------- | --- | ------------------------------------------- | --- | --- | --- | --- | ------- | --- | --- |
t
|     |     | K                 | D                |           |     |     |         |     |     |
| --- | --- | ----------------- | ---------------- | --------- | --- | --- | ------- | --- | --- |
|     |     | (cid:92) (cid:92) | (cid:92) (cid:8) |           |     |     | (cid:9) |     |     |
|     |     | E :=              | Ld               | (t)≤µd−µd |     | ≤Ud | (t) ,   |     |     |
|     |     | t                 | i,j              | i         | j   | i,j |         |     |     |
i=1j̸=id=1
ifE holds,thenforanyi,j
t
M−(i,j,t):=maxLd (t)≤M(i,j):=max(µd−µd)≤maxUd (t):=M+(i,j,t),
|     |     | d i,j |     | d   | i   | j   | d i,j |     |     |
| --- | --- | ----- | --- | --- | --- | --- | ----- | --- | --- |
and the second point follows by noting that m(i,j) = −M(i,j) and m+(i,j,t) :=
−M−(i,j,t);m−(i,j,t):=−M+(i,j,t)foranypairofarms.
Weremarkthatwhenthealgorithmusesconfidencebonusofform(µd(t)−µd(t))±β (t),
|                                   |     |                              |     |           |           |     | (cid:98)i      | (cid:98)j | i,j      |
| --------------------------------- | --- | ---------------------------- | --- | --------- | --------- | --- | -------------- | --------- | -------- |
| M+(i,j,t)                         |     | maxUd (t)=max(µd(t)−µd(t))+β |     |           |           |     |                |           |          |
|                                   |     | :=                           |     | (cid:98)i | (cid:98)j | i,j | (t)=M(i,j,t)+β |           | i,j (t), |
|                                   |     | d i,j                        | d   |           |           |     |                |           |          |
| M−(i,j,t)                         |     | maxLd (t)=max(µd(t)−µd(t))−β |     |           |           |     |                |           |          |
|                                   |     | :=                           |     |           |           | i,j | (t)=M(i,j,t)−β |           | i,j (t), |
|                                   |     | d i,j                        | d   | (cid:98)i | (cid:98)j |     |                |           |          |
| andthepreviouslemmaimpliesthatonE |     |                              | ,   |           |           |     |                |           |          |
t
|     | |M(i,j)−M(i,j,t)|≤β |     | (t) | and |m(i,j)−m(i,j,t)|≤β |     |     |     | (t), |     |
| --- | ------------------- | --- | --- | ----------------------- | --- | --- | --- | ---- | --- |
|     |                     |     | i,j |                         |     |     |     | i,j  |     |
whichisextensivelyusedinoursamplecomplexityanalyses.
14

A.2 ProofofLemma2
Lemma2. AssumeE holds. Forε 1 -PSI(resp. (ε 1 ,ε 2 )-PSI,ε 1 -PSI-k),AdaptiveParetoExploration
combinedwiththestoppingruleτ (resp. τ ,resp. τk)outputsacorrectsubset.
|     |     |     | ε1  | ε1,ε2 | ε1  |     |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
Weshowthecorrectnessofε -PSI-k (foranyk)andwederivethecorrectnessforε -PSIwhich
|     |     |     | 1   |     |     |     |     |     | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
isequivalenttoε -PSI-K. Thecorrectnessof(ε ,ε )-PSIisshownseparatelyinLemma11(see
|     | 1   |     |     |     | 1 2 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
AppendixC).
ProofofLemma2. AssumeE holds. Lett=τk andi∈OPTε1(t). Sincei∈OPTε1(t),forany
ε1
j ̸=i,
E
≥M−(i,j,t)+ε
|     |     |     | M(i,j)+ε | 1   |     | 1 >0, |     |     |     |     |
| --- | --- | --- | -------- | --- | --- | ----- | --- | --- | --- | --- |
thatisi ∈ S⋆ . Therefore, onthe event E, OPTε1(t) ⊂ S⋆ . Thus, ifthe stoppinghasoccurred
| ε1ε1(t)|≥k,sinceinthiscaseO(t)⊂OPTε1(t) |     |     |     |     |     | ε1  |                            |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- |
| because|OPT                             |     |     |     |     |     | ⊂S⋆ | ,alltherecommendedarmswill |     |     |     |
ε1
be(ε )-Paretooptimal. Onthecontrary,if|OPTε1(t)|<k,thenfromthedefinitionofτk itholds
| 1   |     |     |     |     |     |     |     |     | ε1  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
that
|     |     |     | Zε1(t)>0 |     | Zε1(t)>0, |     |     |     |     |     |
| --- | --- | --- | -------- | --- | --------- | --- | --- | --- | --- | --- |
and
|     |     |     | 1   |     | 2   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andtherecommendedsetisthen
|     |     | O(t):=S(t)∪{i∈S(t)c |     |     | :∄j ̸=i:m−(i,j,t)>0}. |     |     |     |     |     |
| --- | --- | ------------------- | --- | --- | --------------------- | --- | --- | --- | --- | --- |
Foranyi∈O(t)c,bythedefinitionoftherecommendedsetandsinceZ
2 (t)>0,
E
|                       |     | ∃j         | ∈A  | suchthat m(i,j)≥m−(i,j,t)>0, |        |     |     |     |     |     |
| --------------------- | --- | ---------- | --- | ---------------------------- | ------ | --- | --- | --- | --- | --- |
| soiisasub-optimalarm. |     | Therefore, |     |                              |        |     |     |     |     |     |
|                       |     |            |     | S⋆                           | ⊂O(t). |     |     |     |     |     |
Moreover,foranyi∈O(t)∩S(t),sinceZε1(t)>0wehavehε1(t)>0,thatis
|     |     |     |     | 1   |     | i   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E
M−(i,j,t)+ε
|     |        | min     | M(i,j)+ε | 1 ≥ | min     |     | 1   | >0. |        | (6) |
| --- | ------ | ------- | -------- | --- | ------- | --- | --- | --- | ------ | --- |
|     |        | j∈A\{i} |          |     | j∈A\{i} |     |     |     |        |     |
|     | S(t)c, |         |          |     |         |     |     |     | Zε1(t) |     |
If i ∈ O(t) ∩ by definition of O(t), we have g i (t) < 0. However, since > 0,
| (t),hε1(t))>0sowealsohavehε1(t)>0and(6)applies. |     |     |     |              |     |     |                    |     | 2   |     |
| ----------------------------------------------- | --- | --- | --- | ------------ | --- | --- | ------------------ | --- | --- | --- |
| max(g i                                         |     |     |     |              |     |     | Thus,foranyi∈O(t), |     |     |     |
| i                                               |     |     |     | i            |     |     |                    |     |     |     |
|                                                 |     |     |     | min M(i,j)+ε |     | >0, |                    |     |     |     |
1
j∈A\{i}
thatisi∈S⋆ ,soS⋆ ⊂O(t)⊂S⋆ . Finallywecanconcludethatε -APE-kandε -APEoutputa
| ε1  |     |     | ε1  |     |     |     | 1   |     | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
correctsubsetonE.
A.3 Calibrationoftheconfidenceintervals
InSection3.2weproposedtoinstantiateouralgorithmswiththeconfidenceintervals
| Ud  | (t)=µd(t)−µd(t)+β |           |           |         | Ld (t)=µd(t)−µd(t)−β |           |           |     |          |     |
| --- | ----------------- | --------- | --------- | ------- | -------------------- | --------- | --------- | --- | -------- | --- |
|     |                   | (cid:98)i | (cid:98)j | i,j (t) | and                  | (cid:98)i | (cid:98)j |     | i,j (t). | (7) |
|     | i,j               |           |           |         | i,j                  |           |           |     |          |     |
WeprovebelowthatE isindeedahigh-probabilityeventforasuitablechoiceofβ (t).
i,j
Lemma6. Letν beabanditwith1-subgaussianmarginals. Fortheconfidenceintervalsdefinedin
(7),with
(cid:118)
|                  | (cid:117)    |         |           |                 |             |     |       |          |        |     |
| ---------------- | ------------ | -------- | --------- | --------------- | ----------- | --- | ------- | -------- | ------- | --- |
|                  | (cid:117)    | (cid:32) | (cid:0)K1 | (cid:1)(cid:33) |             |     |         |          |         |     |
|                  |              | log      |           | (cid:88)        |             |     |         | (cid:88) | 1       |     |
| β (t)=2(cid:117) | (cid:116)Cg |          | δ         | +               | log(4+log(T |     | (t))) |          | .      |     |
| i,j              |              |          |           |                 |             |     | a       |          |         |     |
|                  |              |          | 2         |                 |             |     |         |          | T a (t) |     |
|                  |              |          |           | a∈{i,j}         |             |     |         | a∈{i,j}  |         |     |
theevent
|     | (cid:92) ∞ |          |       | (cid:92) K (cid:92) (cid:92) | D                    |     |     |     |         |     |
| --- | ---------- | -------- | ----- | ---------------------------- | -------------------- | --- | --- | --- | ------- | --- |
|     |            |          |       |                              | (cid:8) Ld (t)≤µd−µd |     | ≤Ud |     | (cid:9) |     |
| E   | =          | E t with | E t = |                              |                      |     |     | (t) |         |     |
|     |            |          |       |                              | i,j                  | i   | j   | i,j |         |     |
|     | t=1        |          |       | i=1j̸=id=1                   |                      |     |     |     |         |     |
issuchthatP(E)≥1−δ.
15

Proof. Byobservingthatforanypairofarmβ =β ,E canberewrittenas
i,j j,i t
D
E = (cid:92) (cid:92) (cid:8) Ld (t,δ)≤µd−µd ≤Ud (t,δ) (cid:9) , (8)
t i,j i j i,j
{i,j}∈Γd=1
D
= (cid:92) (cid:92) (cid:8) |(µd(t)−µd(t))−(µd−µd)|≤β (t) (cid:9) , (9)
(cid:98)i (cid:98)j i j i,j
{i,j}∈Γd=1
whereΓ:= (cid:0) 2 (cid:1) isthesetofpairof2elementsof[K],whichsatisfies|Γ|=K(K−1)/2.Therefore,
[K]
usingaunionbound,
P(Ec) = P(∃t≥1:Echolds),
t
= P (cid:0) ∃t≥1,d∈[D],{i,j}∈Γ:|(µd(t)−µd(t))−(µd−µd)|>β (t) (cid:1) ,
(cid:98)i (cid:98)j i j i,j
D
≤ (cid:88) (cid:88) P (cid:0) ∃t≥1:|(µd(t)−µd(t))−(µd−µd)|>β (t) (cid:1) ,
(cid:98)i (cid:98)j i j i,j
{i,j}∈Γd=1
D
(cid:88) (cid:88) δ
≤ (byProposition24of[20]whichwerecallbelowinLemma7),
K
1
{i,j}∈Γd=1
= δ,
sinceK := K(K−1)D and|Γ|=K(K−1)/2.
1 2
Lemma7(Proposition24of[20]). LetX,Y becentered1−subgaussianrandomvariablesand
δ ∈(0,1).LetX,X ,X ,... be i.i.drandomvariablesandY,Y ,Y ,... be i.i.drandomvariables.
1 2 1 2
Withprobabilityatleast1−δ,forallp,q ≥1,
(cid:12) (cid:12)1(cid:88) p 1(cid:88) q (cid:12) (cid:12) (cid:115) (cid:18) (cid:18) log(1/δ) (cid:19) (cid:19)(cid:18) 1 1 (cid:19)
(cid:12) X − Y (cid:12)≤2 Cg +loglog(e4p)+loglog(e4q) +
(cid:12)p s q s(cid:12) 2 p q
(cid:12) (cid:12)
s=1 s=1
whereCg(x)≈x+log(x).
B Samplecomplexityanalysis
InthissectionweproveTheorem1whichisrestatedbelow.
Theorem1. Fixariskparameterδ ∈(0,1),ε ≥0,letk ≤K andν abanditwith1-subgaussian
1
marginals. Withprobabilityatleast1−δ, ε -APE-k recommendsacorrectsetfortheε -PSI-k
1 1
objectiveandstopsafteratmost
(cid:18) (cid:18) (cid:19)(cid:19)
(cid:88) 88 2K(K−1)D 12e
log log ,
a∈A
∆(cid:101)2
a
δ ∆(cid:101)a
samples,whereforeacha∈A,∆(cid:101)a :=max(∆
a
,ε
1
,ωk).
ProofofTheorem1. The correctness follows from Lemma 2 and the fact that P(E) ≥ 1 − δ
(Lemma 6). The upper-bound on the sample complexity is a direct consequence of Lemma 3,
Lemma4,Lemma5whichareprovedlaterinthissection. Indeed,usingtheselemmaswehavethat,
ifε -APE-khasnotstoppedduringroundti.et<τk andtheeventE holds,then
1 ε1 t
a) ωk ≤2β (t)(Lemma3),
at,at
b) ∆ ≤2β (t)(Lemma4),
at at,at
c) ε ≤2β (t)(Lemma5)
1 at,at
16

holdsimultaneously. Then,ifwedonotcountthefirstK roundsduetoinitialization,andletting
| ∆(cid:101)a :=max(ωk,ε |     | ,∆ ), |     |     |     |     |     |     |     |     |     |
| ---------------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1 a
∞
(cid:88)
|     |     | τk1{E}−1 |     | ≤   | 1{E}1{τk | >t}, |     |     |     |     |     |
| --- | --- | -------- | --- | --- | -------- | ---- | --- | --- | --- | --- | --- |
|     |     | ε1       |     |     |          | ε1   |     |     |     |     |     |
t=1
∞
|     |     |     |     | (cid:88) | 1{max(ωk,ε |     |      |      |       |     |     |
| --- | --- | --- | --- | -------- | ---------- | --- | ---- | ---- | ----- | --- | --- |
|     |     |     |     | ≤        |            |     | ,∆   | )≤2β | (t)}  |     |     |
|     |     |     |     |          |            |     | 1 at |      | at,at |     |     |
t=1
∞
(cid:88)
|     |     |     |     | =   | 1{∆(cid:101)at | ≤2β | (t)} |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | ---- | --- | --- | --- | --- |
at,at
t=1
∞ K
(cid:88)(cid:88)
|     |     |     |     | =   | 1{{a | =a}∧{∆(cid:101)a |     | ≤2β | (t)}} |     |     |
| --- | --- | --- | --- | --- | ---- | ---------------- | --- | --- | ----- | --- | --- |
|     |     |     |     |     |      | t                |     |     | a,a   |     |     |
t=1a=1
K ∞
(cid:88)(cid:88)
|     |     |     |     | =   | 1{{a | =a}∧{∆(cid:101)a |     | ≤2β | (t)}} |     |     |
| --- | --- | --- | --- | --- | ---- | ---------------- | --- | --- | ----- | --- | --- |
|     |     |     |     |     |      | t                |     |     | a,a   |     |     |
a=1t=1
K
(cid:88)
|     |     |     |     | ≤   | inf{n≥2:∆(cid:101)a |     | >2βn}, |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------- | --- | ------ | --- | --- | --- | --- |
a=1
| whereβnistheexpressionofβ |     |     |     | (t)whenT | (t)=T | (t)=n,thatis |     |     |     |     |     |
| ------------------------- | --- | --- | --- | -------- | ----- | ------------ | --- | --- | --- | --- | --- |
|                           |     |     | i,j |          | i     | j            |     |     |     |     |     |
(cid:118)
|     |     |     | (cid:117) (cid:32) | (cid:32) | (cid:0)K1 (cid:1)(cid:33) |                 |     |     | (cid:33) |     |     |
| --- | --- | --- | ------------------ | -------- | ------------------------- | --------------- | --- | --- | -------- | --- | --- |
|     |     |     | (cid:117)          | log      |                           |                 |     |     | 2        |     |     |
|     |     | βn  | =2(cid:116)        | Cg       | δ                         | +2log(4+log(n)) |     |     | .        |     |     |
|     |     |     |                    |          | 2                         |                 |     |     | n        |     |     |
Then,aninversionresultgiveninLemma19yields
|     |             |     |     |                 |             | (cid:18) |     |     | (cid:18)    | (cid:19)(cid:19) |     |
| --- | ----------- | --- | --- | --------------- | ----------- | -------- | --- | --- | ----------- | ---------------- | --- |
|     |             |     |     |                 | 88          | 2K(K−1)D |     |     | 12e         |                  |     |
|     | inf{s≥2:2βs |     |     | <∆(cid:101)a }≤ | log         |          |     | log |             | .                |     |
|     |             |     |     |                 | ∆(cid:101)2 |          | δ   |     | ∆(cid:101)a |                  |     |
a
Therefore,
|     |     |         |     |             | (cid:18) |     |     | (cid:18) | (cid:19)(cid:19) |     |     |
| --- | --- | ------- | --- | ----------- | -------- | --- | --- | -------- | ---------------- | --- | --- |
|     |     |         |     | (cid:88) 88 | 2K(K−1)D |     |     | 12e      |                  |     |     |
|     |     | τk1{E}≤ |     |             | log      |     |     | log      | .                |     |     |
ε1
|     |     |     |     | ∆(cid:101)2 |     | δ   |     | ∆(cid:101)a |     |     |     |
| --- | --- | --- | --- | ----------- | --- | --- | --- | ----------- | --- | --- | --- |
|     |     |     |     | a∈A         | a   |     |     |             |     |     |     |
Wewillnowprovethelemmasinvolvedintheproofofthemaintheorem. Twoofthem(Lemma4,
Lemma5)relyonthefollowingresult,whichisanimportantconsequenceofthedefinitionofthe
APEsamplingrule.
=τk
Lemma8. Letε 1 ≥0,ε 2 ≥0andk ∈[K]. Ifτ orτ =τ ε1,ε2 thefollowingholds. Ift<τ
ε1
| thenforanyj | ∈A,m(b |     | ,j,t)≤β | (t). |     |     |     |     |     |     |     |
| ----------- | ------ | --- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
|             |        |     | t       | bt,j |     |     |     |     |     |     |     |
Proof.
Theproofissplitintwosteps.
| Step1 Ift<τk |     | thenforanyj |     | ∈A,m(b | ,j,t)≤β |      | (t). |     |     |     |     |
| ------------ | --- | ----------- | --- | ------ | ------- | ---- | ---- | --- | --- | --- | --- |
|              |     | ε1          |     |        | t       | bt,j |      |     |     |     |     |
First,notethatt < max(τk,τ )impliesthatZε1(t) ≤ 0orZε1(t) ≤ 0. Bydefinitionofb
|     |     |     | ε1 ε1 |     |     | 1   |     | 2   |     |     | t and |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | ----- |
notingthatM(i,j,t)=−m(i,j,t),wehave
|                      |     |             | b ∈ | argmin      | maxm(i,j,t)−β |          |        | (t). |     |     | (10) |
| -------------------- | --- | ----------- | --- | ----------- | ------------- | -------- | ------ | ---- | --- | --- | ---- |
|                      |     |             | t   |             |               |          |        | i,j  |     |     |      |
|                      |     |             |     | i∈OPTε1(t)c | j̸=i          |          |        |      |     |     |      |
| sothatifthereexistsj |     | suchthatm(b |     | ,j,t)>β     |               | (t),then |        |      |     |     |      |
|                      |     |             |     | t           | bt,j          |          |        |      |     |     |      |
|                      |     |             |     | maxm(b      | ,j,t)−β       |          | (t)>0, |      |     |     |      |
|                      |     |             |     |             | t             | bt,j     |        |      |     |     |      |
j̸=bt
therefore,
|     |     | ∀i∈OPTε1(t)c |     | ,maxm(i,j,t)−β |     |     | (t)>0 | i.e | g (t)>0. |     | (11) |
| --- | --- | ------------ | --- | -------------- | --- | --- | ----- | --- | -------- | --- | ---- |
|     |     |              |     |                |     | i,j |       |     | i        |     |      |
j̸=i
Furthermore,foranyi∈OPTε1(t),hε1(t)>0. Puttingthingstogether,ifthereexistsj suchthat
i
| m(b ,j,t)>β |      | (t)then,Zε1(t)>0andZε1(t)>0. |     |     |     |     |     |     |     |     |     |
| ----------- | ---- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t           | bt,j |                              | 1   |     | 2   |     |     |     |     |     |     |
17

| Step2 Ift<τ |     | thenforanyj |     | ∈A,m(b | ,j,t)≤β |      | (t). |     |     |     |
| ----------- | --- | ----------- | --- | ------ | ------- | ---- | ---- | --- | --- | --- |
|             |     | ε1,ε2       |     |        | t       | bt,j |      |     |     |     |
impliesthatZε1,ε2(t)≤0orZε1,ε2(t)≤0.
| Recallthatbydefinitiont<τ |     |         |       |          |         |        |     |     | Using(10),ifthere |     |
| ------------------------- | --- | ------- | ----- | -------- | ------- | ------ | --- | --- | ----------------- | --- |
|                           |     |         | ε1,ε2 |          | 1       |        |     | 2   |                   |     |
| existsj suchthatm(b       |     | ,j,t)>β |       | (t),then |         |        |     |     |                   |     |
|                           |     | t       |       | bt,j     |         |        |     |     |                   |     |
|                           |     |         |       | maxm(b   | ,j,t)−β | (t)>0. |     |     |                   |     |
|                           |     |         |       |          | t       | bt,j   |     |     |                   |     |
j̸=bt
Combiningthiswith
|     |     | gε2(t):= | max | m(i,j,t)−β |     | (t)+ε | 1{j | ∈OPTε1(t)}, |     |     |
| --- | --- | -------- | --- | ---------- | --- | ----- | --- | ----------- | --- | --- |
|     |     | i        |     |            | i,j |       | 2   |             |     |     |
j∈A\{i}
yields
|     |     | ∀i∈OPTε1(t)c |     | ,0<maxm(i,j,t)−β |     |     |     | (t)≤gε2(t). |     | (12) |
| --- | --- | ------------ | --- | ---------------- | --- | --- | --- | ----------- | --- | ---- |
|     |     |              |     |                  |     |     | i,j |             | i   |      |
j̸=i
Furthermore,sincewehave
|     |     |     |     | ∀i∈OPTε1(t),hε1(t)>0, |     |     |     |     |     | (13) |
| --- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | ---- |
i
theinitialassumptionwouldyieldthatforanyarmi,max(hε1(t),gε2(t))>0,soZε1,ε2(t)>0and
|     |     |     |     |     |     |     | i   | i   | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Zε1,ε2(t)>0.
2
=τk
Weconcludethatifτ orτ ε1,ε2 ,t<τ impliesthatforanyj ∈A,m(b t ,j,t)≤β bt,j (t).
ε1
B.1 ProofofLemma3
| Lemma3. | Letε | ≥andk | ∈[K]. | IfE holdsandt<τk |     |     | thenωk | ≤2β | (t).  |     |
| ------- | ---- | ----- | ----- | ---------------- | --- | --- | ------ | --- | ----- | --- |
|         |      | 1     |       | t                |     | ε1  |        |     | at,at |     |
ProofofLemma3. First, note that if k > |S⋆|, then the lemma holds trivially since ω < 0. In
k
the sequel, we assume E holds and k ≤ |S⋆|. If t < τk then it holds that |OPTε1(t)|< k. So
|                |     |        | t                       |     |       | ε       |     |     |     |     |
| -------------- | --- | ------ | ----------------------- | --- | ----- | ------- | --- | --- | --- | --- |
| S⋆,k∩OPTε1(t)c |     |        | eti∈S⋆,k∩OPTε1(t)c,weha |     |       |         | 1   |     |     |     |
|                |     | ̸=∅. L |                         |     |       |         | v e |     |     |     |
|                |     |        | ωk                      | ≤ ω | = min | M(i,j), |     |     |     |     |
i
j∈A\{i}
(a)
|     |     |     |     | ≤ min | M(i,j,t)+β |     | (t), |     |     |     |
| --- | --- | --- | --- | ----- | ---------- | --- | ---- | --- | --- | --- |
i,j
j∈A\{i}
(b)
|     |     |     |     | ≤ min | M(b | t ,j,t)+β | bt,j | (t), |     |     |
| --- | --- | --- | --- | ----- | --- | --------- | ---- | ---- | --- | --- |
j∈A\{bt}
|     |     |     |     | ≤ M(b | ,c ,t)+β |       | (t), |     |     |     |
| --- | --- | --- | --- | ----- | -------- | ----- | ---- | --- | --- | --- |
|     |     |     |     |       | t t      | bt,ct |      |     |     |     |
(c)
|     |     |     |     | ≤ 2β | (t), |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- |
bt,ct
|     |     |     |     | ≤ 2β | (t), |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- |
at,at
where(a)usesthatE holdsandLemma1, (b)usesthedefinitionofb and(c)followsfromthe
|     |     | t   |     |     |     |     |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
definition of c and the fact that b ∈/ OPTε1(t),which yields M(b ,c ,t) ≤ β (t). The last
|                         | t   |     |                         | t   |     |     |      | t             | t bt,ct |     |
| ----------------------- | --- | --- | ----------------------- | --- | --- | --- | ---- | ------------- | ------- | --- |
| inequalityfollowssincea |     |     | istheleastsampledamongb |     |     | ,c  | andβ | isdecreasing. |         |     |
|                         |     |     | t                       |     |     | t   | t    |               |         |     |
B.2 ProofofLemma4
=τk
Lemma4. Letε 1 ≥0. Letτ forsomek ∈[K]orτ =τ ε1,ε2 forsomeε 2 ≥0. IfE t holdsand
ε1
| t<τ then∆ | ≤2β |     | (t). |     |     |     |     |     |     |     |
| --------- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
at at,at
BeforeprovingtheLemma4,westatethefollowinglemmawhichisusedtoderiveanupperbound
| onthegapofanoptimalarm. |                           |     | Itsproofispostponedtotheendofthesection. |     |      |         |     |     |     |     |
| ----------------------- | ------------------------- | --- | ---------------------------------------- | --- | ---- | ------- | --- | --- | --- | --- |
| Lemma9.                 | ForanyParetooptimalarmi,∆ |     |                                          |     | ≤min | M(i,j). |     |     |     |     |
|                         |                           |     |                                          |     | i    | j̸=i    |     |     |     |     |
ProofofLemma4. AssumethatE holds. Weconsiderfourdifferentcasesdependingonwhetherb
|     |     |     |     | t   |     |     |     |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andc areoptimalorsub-optimal.
t
18

Case1.1 b isaParetooptimalarm. Fromthedefinitionofthegapofanoptimalarmandusing
t
Lemma9itfollows∆ ≤M(b ,c )whichonE andusingLemma1yields
bt t t t
∆ +ε ≤M(b ,c ,t)+β (t)+ε (14)
bt 1 t t bt,ct 1
then,notingthatthereexistsj ∈A\{b }suchthatM(b ,j,t)+ε ≤β (t),bydefinitionofc ,
t t 1 bt,j t
wehave
M(b ,c ,t)+ε ≤β (t), (15)
t t 1 bt,ct
therefore,
∆ +ε ≤ 2β (t).
bt 1 bt,ct
Case1.2 b isasub-optimalarm. Bydefinitionofc andusingM=−m,wehave
t t
c ∈ argmaxm(b ,j,t)+β (t), (16)
t t bt,j
j∈A\{bt}
then,fromthedefinitionofthegapofasub-optimalarmandsinceE holds,weknowthatthereexists
t
anarmb⋆suchthat
t
∆ =m(b ,b⋆) ≤ m(b ,b⋆,t)+β (t),
bt t t t t bt,b⋆ t
(a)
≤ m(b ,c ,t)+β (t),
t t bt,ct
(b)
≤ 2β (t).
bt,ct
where(a)usesthedefinitionofc and(b)usesLemma8.
t
Case2.1 c isaParetooptimalarm. Ifb isalsoanoptimalarm,itfollowsthat∆ ≤M(b ,c )
t t ct t t
whichonE yields∆ ≤M(b ,c ,t)+β (t),then,similarlytocase1.1,wehaveM(b ,j,t)+
t ct t t bt,ct t
ε ≤β (t)so
1 bt,j
∆ +ε ≤2β (t).
ct 1 bt,ct
Now,assumeb isasub-optimalarm. Then,bydefinition,∆ ≤M(b ,c )++∆ . Usingasimilar
t ct t t bt
reasoningtocase1.2,itholdsthat∆ ≤m(b ,c ,t)+β (t),so
bt t t bt,ct
∆ ≤ M(b ,c )++∆ ,
ct t t bt
≤ (M(b ,c ,t)+β (t))++m(b ,c ,t)+β (t),
t t bt,ct t t bt,ct
= (−m(b ,c ,t)+β (t))++m(b ,c ,t)+β (t),
t t bt,ct t t bt,ct
(a)
≤ max(2β (t),m(b ,c ,t)+β (t))
bt,ct t t bt,ct
(b)
≤ 2β (t).
bt,ct
where(a)followsfrom(x−y)++(x+y)≤max(x+y,2x)and(b)followsfromm(b ,c ,t)≤
t t
β (t)(Lemma8).
bt,ct
Case2.2 c isasub-optimalarm. Weknowthatthereexistsanarmc⋆suchthat∆ =m(c ,c⋆).
t t ct t t
Ifc⋆ =b then,sincem(j,i)≤M(i,j)(followsfromthedefinition),wehave
t t
∆ =m(c ,c⋆) = m(c ,b ),
ct t t t t
≤ M(b ,c ),
t t
(a)
≤ M(b ,c ,t)+β (t),
t t bt,ct
(b)
≤ 2β (t),
bt,ct
where (a) follows from E and (b) has been already justified in the case 1.1. If b ̸= c⋆, then by
t t t
definitionofc ,wehave
t
m(b ,c ,t)+β (t)≥m(b ,c⋆,t)+β (t),
t t bt,ct t t bt,c⋆ t
19

whichimpliesthatthereexistsd∈[D]suchthat
µd (t)−µd (t)+β (t)≥µd (t)−µd (t)+β (t)≥
Et
µd −µd ,
(cid:98)ct (cid:98)bt bt,ct (cid:98)c⋆ t (cid:98)bt bt,c⋆ t c⋆ t bt
thenrecallingthatβ =β ,
i,j j,i
µd −µd +2β (t)≥
Et
(µd (t)−µd (t)−β (t))+2β (t)≥µd −µd .
ct bt bt,ct (cid:98)ct (cid:98)bt bt,ct bt,ct c⋆
t
bt
Puttogether,thereexistsd∈[D]suchthat
µd −µd ≤2β (t),
c⋆
t
ct bt,ct
so
∆ =min(µd −µd )≤2β (t),
ct
d
c⋆
t
ct bt,ct
Puttingthefourcasestogether,wehaveprovedthatift<max(τk,τ )thenboth
ε1 ε1,ε2
∆ ≤2β (t) and ∆ ≤2β (t) (17)
bt bt,ct ct bt,ct
holds. Further noting that a is the least sampled among among b ,c and β is non-increasing,
t t t
β (t)≤β (t),(17)yields
bt,ct at,at
∆ ≤2β (t),
at at,at
whichachievestheproof.
B.3 ProofofLemma5
Thefollowinglemmaholdsforeachofthestoppingtimesτ ,τ andτk.
ε1 ε1,ε2 ε1
Lemma5. Letε ≥0andτ beasinLemma4. IfE holdsandt<τ thenε ≤2β (t).
1 t 1 at,at
ProofofLemma5. ByLemma8,wehavem(b ,c ,t)≤β (t)orequivalently
t t bt,ct
M(b ,c ,t)≥−β (t). (18)
t t bt,ct
Then, knowing that b ∈/ OPTε1(t), there exists an arm j such that ε +M(b ,j,t) ≤ β (t).
t 1 t bt,j
Usingfurtherthedefinitionofc ,itfollowsthatε +M(b ,c ,t)≤β (t). Combiningthiswith
t 1 t t bt,ct
inequality(18)andnotingthata istheleastsampledamongb ,c yields
t t t
β (t)≥β (t)≥ε /2.
at,at bt,ct 1
B.4 Auxiliaryresults
WestatethefollowinglemmawhichisusedtoproveLemma9.
Lemma10. Foranysub-optimalarma,thereexistsaParetooptimalarma⋆suchthatµ ≺µ
a a⋆
and∆ =m(a,a⋆)>0. Moreover,Foranyi∈A\S⋆,j ∈S⋆,
a
i) max m(i,k)=max m(i,k),
k∈S⋆ k∈A
ii) Ifi∈argmin M(j,k)thenj istheuniquearmsuchthatµ ≺µ
k∈A\{j} i j
Proof. Assumetherearep < ndominatedarms. Withoutlossofgenerality,wemayassumethey
areµ ,...,µ . Leti ≤ p. SupposethatnoPareto-optimalarmdominatesµ . Sinceµ isnot
1 p 1 i1 i1
optimal,bythelatterassumption,thereexistsi ≤psuchthatµ ≺µ . Ifµ isdominatedbya
2 i1 i2 i2
Paretooptimalarm,thisarmalsodominatesµ (strictdominanceistransitive)whichcontradictsthe
i1
initialassumption. Ifnot,thereexitsi ≤psuchthatµ ≺µ ≺µ . Againwecanusethesame
3 i1 i2 i3
reasoningasbeforefori . Inanycaseweshouldstopinatmostpsteps,otherwisewewouldhave
3
µ ≺µ ≺···≺µ andµ shouldbedominatedbyaPareto-optimalarm,otherwiseitwould
i1 i2 ip ip
beitselfPareto-optimal,whichisnotthecase. Therefore,foranya∈A\S⋆,thereexistsa⋆ ∈S⋆
suchthata⋆ ≺aand∆ =m(a,a⋆)>0.
a
20

Lettingibeasub-optimalarm,sinceforanya∈A\S⋆,thereexistsa⋆ ∈S⋆suchthata≺a⋆,it
followsthat
|     |     | ∀d∈[D], µd−µd | <µd −µd, |     |
| --- | --- | ------------- | -------- | --- |
a i a⋆ i
whichleadstom(i,a)≤m(i,a⋆),so
maxm(i,j)=maxm(i,j)>0,
|     |     | j∈A j∈S⋆ |     |     |
| --- | --- | -------- | --- | --- |
whichachievestheproofofthefirstpointi). Forthesecondpoint,letq ∈A\S⋆ andq′ suchthat
≺q′and
q
q ∈ argminM(j,a).
a∈A\{j}
| Bydirectalgebra,sinceq | ≺q′,wehave |     |     |     |
| ---------------------- | ---------- | --- | --- | --- |
M(j,q′)<M(j,q),
whichisimpossibleifq′
|     | ̸=j | (becauseqbelongstotheargmin). | Therefore,if |     |
| --- | --- | ----------------------------- | ------------ | --- |
q ∈ argminM(j,a)
a∈A\{j}
| isasub-optimalarm,thenj |     | istheonlyarmsuchthatq | ≺j (i.eµ ≺µ | ).  |
| ----------------------- | --- | --------------------- | ----------- | --- |
q j
WenowproveLemma9whichfollowsfromthepreviouslemma.
ProofofLemma9. Ifargmin M(i,j) ⊂ S⋆,thenthelemmafollowsfromthedefinitionofthe
j̸=i
gap of an optimal arm recalled in Section 4. If min M(i,j) = M(i,a), a ∈/ S⋆, then, from
j̸=i
Lemma10,iistheuniquearmwhichdominatesaso∆ a =m(a,i)andusingthedefinitionofthe
gapofanoptimalarm,
|     |     | ∆ ≤ M(a,i)++∆ | ,   |     |
| --- | --- | ------------- | --- | --- |
|     |     | i             | a   |     |
= 0+m(a,i)≤M(i,a),
wherewehaveusedthethefactthatm(p,q)≤M(q,p)foranypairofarmsp,q(whichfollowsfrom
| thedefinition). | Therefore,foranoptimalarmi,wealwayshave |     |     |     |
| --------------- | --------------------------------------- | --- | --- | --- |
∆ ≤minM(i,j).
i
j̸=i
| C Algorithmforfindingan(ε |     | ,ε )-cover |     |     |
| ------------------------- | --- | ---------- | --- | --- |
1 2
Inthissection,weanalysethesamplecomplexityofAPEwhenitisassociatedtothestoppingtime
τ foridentifyingan(ε ,ε )-coveroftheParetoset. Thesamplingruleremainsunchangedanwe
| ε1,ε2 | 1   | 2   |     |     |
| ----- | --- | --- | --- | --- |
provethatthealgorithmdoesnotrequiremoresamplestofindan(ε 1 ,ε 2 )-coverthantosolvethe
ε -PSIproblem.
1
| Algorithm2:(ε | ,ε )-APE |     |     |     |
| ------------- | -------- | --- | --- | --- |
1 2
| Data: parameterε | ≥0  |     |     |     |
| ---------------- | --- | --- | --- | --- |
1
| initialize:sampleeacharmonce,sett=K,T |     | (t)=1foranyi∈A |     |     |
| ------------------------------------- | --- | -------------- | --- | --- |
i
fort=K+1,...,do
M+(i,j,t);
| b t :=argmax | i∈A\OPTε1(t) | min j̸=i  |     |     |
| ------------ | ------------ | --------- | --- | --- |
| c :=argmin   | M−(b         | ,j,t);    |     |     |
| t            | j̸= b        | t         |     |     |
| Zε1,ε2(t)>   | t Zε1,ε2     |           |     |     |
| if           | 0 ∧          | (t)>0then |     |     |
| 1            | 2            |           |     |     |
breakandoutputOPTε1(t);
| samplea                  | :=argmin    | T (t); |     |     |
| ------------------------ | ----------- | ------ | --- | --- |
|                          | t i∈{bt,ct} | i      |     |     |
| Werecallthestoppingtimeτ |             | .      |     |     |
ε1,ε2
21

| Stoppingrule |     | Letε  | ,ε ≥0and0<δ |     | <1.           | Then |                |     |     |     |      |
| ------------ | --- | ----- | ----------- | --- | ------------- | ---- | -------------- | --- | --- | --- | ---- |
|              |     | 1     | 2           |     |               |      |                |     |     |     |      |
|              |     | τ     | :=inf{t≥K   |     | : Zε1,ε2(t)>0 |      | ∧Zε1,ε2(t)>0}, |     |     |     | (19) |
|              |     | ε1,ε2 |             |     | 1             |      |                | 2   |     |     |      |
where,
|     |     |     | Zε1,ε2(t) |     |     | max(gε2(t),hε1(t)) |     |     |     |     |     |
| --- | --- | --- | --------- | --- | --- | ------------------ | --- | --- | --- | --- | --- |
:= min
|     |     |     | 1   |     |     |     | i   | i   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i∈S(t)
Zε1,ε2(t)
|     |     |     |     | :=  | min | max(gε2(t),hε1(t)), |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- |
|     |     |     | 2   |     |     |                     | i   | i   |     |     |     |
i∈S(t)c
and
|     |     | gε2(t) |     |         | m−(i,j,t)+ε |     | 1{j | ∈OPTε1(t)} |     |     |     |
| --- | --- | ------ | --- | ------- | ----------- | --- | --- | ---------- | --- | --- | --- |
|     |     |        | :=  | max     |             |     | 2   |            |     |     |     |
|     |     |        | i   | j∈A\{i} |             |     |     |            |     |     |     |
|     |     | hε1(t) | :=  | min     | M−(i,j,t)+ε |     |     |            |     |     |     |
|     |     |        | i   |         |             |     | 1   |            |     |     |     |
j∈A\{i}
Recommendationrule Whenitisassociatedtothestoppingtimeτ ,APErecommends
ε1,ε2
):=OPTε1(τ
|     |     |     |     | O(τ ε1,ε2 |     |     | ε1,ε2 | ),  |     |     |     |
| --- | --- | --- | --- | --------- | --- | --- | ----- | --- | --- | --- | --- |
which can be understood as follows. When τ is reached, the arms that are not yet identified
|     |     |     |     |     | ε1  | ,ε 2 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
as(nearly)optimalareeitherε -dominatedby a n arminOPTε1(τ )orsub-optimal,whichis
|     |     |     | 2   |     |     |     |     | ε1,ε2 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- |
provenformallyinLemma11.
Lemma11. Fixδ ∈(0,1),ε 1 ,ε 2 ≥0then(ε 1 ,ε 2 )-APErecommendsan(ε 1 ,ε 2 )-coverofthePareto
setontheeventE.
ProofofLemma11. Assume E holds. Let t = τ and i ∈ OPTε1(t). Since i ∈ OPTε1(t),
ε1,ε2
E
for any j ̸= i, M(i,j)+ε ≥ M−(i,j,t)+ε > 0 that is i ∈ S⋆ . Therefore, on the event E,
|          |      |        | 1             |     |      | 1    |          | ε            |     |           |      |
| -------- | ---- | ------ | ------------- | --- | ---- | ---- | -------- | ------------ | --- | --------- | ---- |
| OPTε1(t) | ⊂ S⋆ |        |               |     | τε   | 1,ε2 |          | Zε1, ε1 2(t) | > 0 | Zε1,ε2(t) | > 0. |
|          | ε1   | . When | t he stopping |     | time | is   | reached, | 1            | and | 2         |      |
δ
Underthiscondition,
OPTε1(t)̸=∅.
Indeed,sinceZε1,ε2(t)>0andZε1,ε2(t)>0,ifOPTε1(t)=∅then,bythestoppingruleandsince
|     | 1   |     |     | 2   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
OPTε1(t)=∅,foranyarmi,wewouldhavehε1(t)<0andgε2(t)>0. Thatis,foranyarmi∈A,
|     |     |     |     |     | i   |     |     | i   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E
|     |     |     | ∃j ̸=isuchthat |     | m(i,j)>m−(i,j,t)>0, |     |     |     |     |     |     |
| --- | --- | --- | -------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- |
soeveryarmwouldbestrictlydominated,whichisimpossiblesincetheParetosetcannotbeempty.
| OPTε1(t) |     |     |     |     |     | OPTε1(t)c, |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
Then, ̸= ∅ and for any i ∈ O(t)c = by the stopping rule it holds that
max(gε2(t),hε1(t)) > 0. Furthernotingthatforsucharmi ∈ OPTε1(t)c, hε1(t) < 0, wethus
| i   | i   |     |     |     |     |     |     |     | i   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
havegε2(t)>0,thatis
i
|                  |     |        | m−(i,j,t)+ε |          |     | 1{j       |     |     |     |     |     |
| ---------------- | --- | ------ | ----------- | -------- | --- | --------- | --- | --- | --- | --- | --- |
|                  |     |        |             |          | 2   | ∈O(t)}>0, |     |     |     |     |     |
| whichontheeventE |     | yields |             |          |     |           |     |     |     |     |     |
|                  |     |        |             | m(i,j)+ε | 1{j | ∈O(t)}>0. |     |     |     |     |     |
2
Therefore,forsucharmi,either
| i) ∃j ∈Asuchthat |     |     | m(i,j)>0thatisµ |     | ≺µ  | or  |     |     |     |     |     |
| ---------------- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
|                  |     |     |                 |     | i   | j   |     |     |     |     |     |
ii) ∃j ∈O(t) suchthat m(i,j)+ε >0thatisµ ≺µ +ε withε :=(ε ,...,ε ).
|     |     |     |     | 2   |     | i   | j   | 2 2 | 2   | 2   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Puttogether,O(t)⊂S⋆ andforanyi∈/ O(t),eitheri∈/ S⋆(iisasub-optimalarm)orthereexists
ε1
j ∈O(t)suchthatµ ≺µ +ε 2 . ThusO(t)isan(ε 1 ,ε 2 )-coveroftheParetosetand(ε 1 ,ε 2 )-APE
|                |     | i                         | j   |     |     |     |     |     |     |     |     |
| -------------- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iscorrectfor(ε |     | ,ε )-coveridentification. |     |     |     |     |     |     |     |     |     |
1 2
ThetwolemmasrestatedbelowareusedtoproveidenticallytoTheorem1,themaintheoremofthis
section.
Lemma4. Letε ≥0. Letτ =τk forsomek ∈[K]orτ =τ forsomeε ≥0. IfE holdsand
|           |     | 1   |      | ε1  |     |     |     | ε1,ε2 | 2   | t   |     |
| --------- | --- | --- | ---- | --- | --- | --- | --- | ----- | --- | --- | --- |
| t<τ then∆ | ≤2β |     | (t). |     |     |     |     |       |     |     |     |
at at,at
22

| Thefollowinglemmaholdsforeachofthestoppingtimesτ |     |     |     |     |     |     |     | ,τ    | andτk. |     |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ----- | ------ | --- | --- |
|                                                  |     |     |     |     |     |     | ε1  | ε1,ε2 | ε1     |     |     |
Lemma5. Letε ≥0andτ beasinLemma4. IfE holdsandt<τ thenε ≤2β (t).
|     |     | 1   |     |     |     | t   |     |     | 1   | at,at |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
Theorem3. Fixδ ∈(0,1),ε ,ε ≥0. Then(ε ,ε )-APEoutputsan(ε ,ε )-coverofthePareto
|     |     |     |     | 1 2 |     | 1 2 |     |     | 1 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
setwithprobabilityatleast1−δusingatmost
|     |     |     |     | 88  | (cid:18) | 2K(K−1)D |     | (cid:18) 12e | (cid:19)(cid:19) |     |     |
| --- | --- | --- | --- | --- | -------- | -------- | --- | ------------ | ---------------- | --- | --- |
(cid:88)
|                           |     |     |     |         | log |       | log |     |     |     | (20) |
| ------------------------- | --- | --- | --- | ------- | --- | ----- | --- | --- | --- | --- | ---- |
|                           |     |     |     | (∆ε)2   |     | δ     |     | ∆ε  |     |     |      |
|                           |     |     | a∈A | a       |     |       |     | a   |     |     |      |
| samples,whereforalla∈A,∆ε |     |     |     | :=max(∆ |     | ,ε ). |     |     |     |     |      |
|                           |     |     |     | a       |     | a 1   |     |     |     |     |      |
This is the first problem-dependent sample complexity upper-bound for the (ε ,ε )-cover of the
|     |     |     |     |     |     |     |     |     |     | 1 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Pareto set. In particular, this result holds for the ε-accurate Pareto set identification [27] which
correspondstotheparticularcaseε =ε =εoftheParetosetcover.Therefore,(ε,ε)-APEcouldbe
|     |     |     |     | 1   | 2   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
comparedtoε-PALforε-accurateParetoset,whichhoweverreliesonaGaussianprocessmodeling
assumption.
Whilethissamplecomplexityresultupper-bounddoesnotclearlyshowthedependenceinε ,we
2
notethatforsomeproblems,wehaveanearlymatchinglowerboundthatdoesnotdependonε . In
2
particular,considerthecaseD =1,ε =0,ε >0andassumethereisauniquebestarm(classical
|     |     |     |     |     | 1   | 2   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
assumptioninBAI)a analgorithmfor(ε ,ε )-coveridentificationisrequired
|     |     | ⋆   | . Forthissetting, |     |     |     |     | 1 2 |     |     |     |
| --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
tooutputasetS(cid:98)suchthatS(cid:98)⊂ S⋆ = {a }andforanyi ̸= a eitherµ < µ orµ ≤ µ +ε
|     |     |     |     |     | ⋆   |     |     | ⋆   | i a⋆ | i   | a⋆ 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ---- |
which trivially holds as long as S(cid:98) ⊂ S⋆. Therefore, this problem is equivalent to (exact) Best
ArmIdentification. AlmostmatchedlowerboundsforBAIareknownanddoesnotdependonε
2
([18, 26, 10]). This observation can be generalized to any configuration where there is a unique
(Pareto)optimalarm. LettingD ≥1,ε =0,ε >0andν abanditwithoneParetooptimalarma ,
|     |     |     |     |     | 1   | 2   |     |     |     |     | ⋆   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
anyalgorithmfor(ε ,ε )-coveringisrequiredtooutputasetS(cid:98)⊂S⋆ ={a }. Andforanyi̸=a
|     |     | 1   | 2   |     |     |     |     |     | ⋆   |     | ⋆   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
whichtriviallyholdsaslongasS(cid:98)⊂S⋆
| eitherµ i | ≺µ a⋆ | orµ i | ≺µ a⋆ | +ε 2 |     |     |     |     | ={a | ⋆ }. So,ontheses |     |
| --------- | ----- | ----- | ----- | ---- | --- | --- | --- | --- | --- | ---------------- | --- |
instances,(0,ε 2 )-coveringisequivalent0-PSIandthenearlymatchedlowerof[4]for0-PSIdoesnot
| dependonε | (Theorem17therein). |     |     |     |     |     |     |     |     |     |     |
| --------- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2
In our experiments (see subsection H.3), we will see that in configurations with multiple Pareto
optimalarms,theparameterε canstillhelptoempiricallyreducethesamplecomplexity.Quantifying
2
itspreciseimpactonthesamplecomplexityisleftasfuturework.
D LowerBound
Inthissection,wegiveagap-dependentlower-boundforthek-relaxationinsomeconfigurations.
We
usethechangeofdistributionlemmaof[18](Lemma1therein).
with|S⋆|=p≥3suchthatfork
| Theorem2. | Thereexistsabanditinstanceν |     |     |     |     |     |     |     | ∈{p,p−1,p−2} |     |     |
| --------- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- |
anyδ-correctalgorithmfor0-PSI-kverifies
|     |     |     |     |      |     | (cid:18) (cid:19) | K        |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | ----------------- | -------- | --- | --- | --- | --- |
|     |     |     |     |      | 1   | 1                 | (cid:88) | 1   |     |     |     |
|     |     |     |     | E (τ | )≥  | log               |          | ,   |     |     |     |
|     |     |     |     | ν    | δ   |                   | (∆k)2    |     |     |     |     |
|     |     |     |     |      | D   | δ                 |          | a   |     |     |     |
a=1
| where∆k     | :=∆ | +ωk   | andτ            | isthestoppingtimeofthealgorithm. |     |                  |     |     |              |      |     |
| ----------- | --- | ----- | --------------- | -------------------------------- | --- | ---------------- | --- | --- | ------------ | ---- | --- |
|             | a   | a     |                 | δ                                |     |                  |     |     |              |      |     |
|             |     | |S⋆|. | w.l.o.gassumeS⋆ |                                  |     | {1,...,p}andS⋆,k |     |     |              |      | RD  |
| Proof. Letp | =   | K =   |                 |                                  |     | =                |     |     | = {1,...,k}. | Letµ | 0 ∈ |
andfor(p−2)≤i≤p,define
|     |     |     |     |     |    | 2p−iω ifd=1 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |

|     |     |     |     | µd := | −2p−iω | elseifd=2, |     |     |     |     |     |
| --- | --- | --- | --- | ----- | ------ | ---------- | --- | --- | --- | --- | --- |
i
 µd
else.
0
for1≤i≤p−3,
|     |     |     |     |     |  (4+2i)ω |     | ifd=1 |     |     |     |     |
| --- | --- | --- | --- | --- | --------- | --- | ----- | --- | --- | --- | --- |

|     |     |     |     | µd := | −(4+2i)ω |     | elseifd=2 |     |     |     |     |
| --- | --- | --- | --- | ----- | -------- | --- | --------- | --- | --- | --- | --- |
i
 µd
else.
0
23

Letν beabanditwhereeacharmiisamultivariateGaussianwithmeanµ andcovariancematrix
i
I i.e ν ∼ N(µ ,I ) (with I the identity matrix in dimension D). By direct calculation, for
| D i | i   | D   | D   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
1≤i,j ≤p−3,
M(i,j)=M(j,i)=2ω|i−j|,
| andforp−2≤i,j | ≤p, |     |     |     |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
M(i,j)=M(j,i)=2pω|2−i−2−j|,
| fori≤p−3and(p−2)≤j   |     |                            | ≤p, |     |     |      |     |     |
| -------------------- | --- | -------------------------- | --- | --- | --- | ---- | --- | --- |
|                      |     | M(i,j)=M(j,i)=(4+2i−2p−j)ω |     |     |     | ≥2ω. |     |     |
| Therefore,computingω |     | andδ+foranyi∈[p]yields     |     |     |     |      |     |     |
|                      |     | i                          | i   |     |     |      |     |     |
δ+
|     |     |     | := min | min(M(i,j),M(j,i)), |     |     |     |     |
| --- | --- | --- | ------ | ------------------- | --- | --- | --- | --- |
i
j∈[p]\{i}
|     |     |     |  ω |     | ifi=p, |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- |

|     |     |     | = 2p−i−1ω |     | ifi∈{p−2,p−1} |     |     |     |
| --- | --- | --- | --------- | --- | ------------- | --- | --- | --- |
2ω
else,
additionally,foranyi≤p,
|     |     | ω   | i := minM(i,j), |     |     |     |     |     |
| --- | --- | --- | --------------- | --- | --- | --- | --- | --- |
j̸=i

|     |     |     | ω   |     | ifi=p, |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- |

2p−i−1ω
|     |     |     | =   |     | ifi∈{p−2,p−1} |     |     |     |
| --- | --- | --- | --- | --- | ------------- | --- | --- | --- |
|     |     |     | 2ω |     | else.         |     |     |     |
Thus,
|                                                                |     |     | ω(p) =ω(p−1) | =ωandω(p−2) |     | =2ω. |     |     |
| -------------------------------------------------------------- | --- | --- | ------------ | ----------- | --- | ---- | --- | --- |
| Letγ >0. Foranyoptimalarmi,sinceM(i,i+1)=M(i+1,i)=δ+,thevector |     |     |              |             |     |      |     |     |
i
µ +δ++γ
i i
−δ+−γ
Paretodominatesµ orµ andµ ≺µ orµ . Moreover,itiseasytoobserve
|     | i+1 |     | i−1 i | i   | i+1 | i−1 |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- |
thatfork ∈{p−2,p−1,p}andanyi∈[p],
+δ++ω(k)+γ
µ
|     |     |     |     | i i |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Pareto dominates 1 (if k ∈ {p,p − 1}) or 2 (if k = p − 2) other optimal arms. Letting k ∈
{p−2,p−1,p},foranyi∈[p],wedefinethealternativebanditν(i)whichisalsoGaussianwith
| thesamecovariancematrixI |     |     | andmeansgivenby |     |     |     |     |     |
| ------------------------ | --- | --- | --------------- | --- | --- | --- | --- | --- |
D

|     |     |     | µ           |     | ifj ̸=i    |                   |     |      |
| --- | --- | --- | ----------- | --- | ---------- | ----------------- | --- | ---- |
|     |     |   | j           |     |            |                   |     |      |
|     | µ(  | i)  | +−ω(k)−γ    |     |            |                   | 1   |      |
|     |     | =   | µ −δ        |     | ifj =iandP | ν (j ∈S(cid:98))≥ |     | (21) |
|     | j   |     | j i         |     |            |                   | 2   |      |
|     |     | µ | +δ ++ω(k)+γ |     | ifj =iandP | (j ∈S(cid:98))<   | 1 . |      |
|     |     |     | j i         |     |            | ν                 |     |      |
2
Therefore,sinceAisδ−correct,andbywhatprecedes,
| • ifP | (i∈S(cid:98))≥ | 1   | thenP (i∈S(cid:98))≤δand |     |     |     |     |     |
| ----- | -------------- | --- | ------------------------ | --- | --- | --- | --- | --- |
|       | ν              | 2   | ν(i)                     |     |     |     |     |     |
| • ifP | (i∈S(cid:98))< | 1   | thenP (i∈S(cid:98))≥1−δ. |     |     |     |     |     |
|       | ν              | 2   | ν(i)                     |     |     |     |     |     |
Thefirstpointfollowssimplyfromthedefinitionofδ+andthefactthatbydesignM(i,j)=M(i,j)
i
∈{p,p−1},inthebanditν(i),atleastonearmofS⋆(ν)is
fori,j ∈[p],Forthesecondpoint,ifk
nolongeroptimal,then|S⋆(ν(i))|≤ p−1 ≤ k.SoP (i ∈ S(cid:98)) ≥ 1−δ. Ifk = p−2sincetwo
ν(i)
armsofS⋆(ν)arenowdominated,wehave|S⋆(ν(i))|≤ p−2 = k,henceP (i ∈ S(cid:98)) ≥ 1−δ
ν(i)
LettingKLdenotetheKulback-LeiberdivergenceandusingLemma1of[18],onF −measurable
τ
event
(cid:40)
1
|     |     |     | {i∈S(cid:98)} |            | ifP ν (i∈S(cid:98))≥ | ,   |     |     |
| --- | --- | --- | ------------- | ---------- | -------------------- | --- | --- | --- |
|     |     |     | E =           |            |                      | 2   |     |     |
|     |     |     | i {i∈/        | S(cid:98)} | ifP (i∈S(cid:98))<   | 1 , |     |     |
ν
2
24

1
| forwhichP | (E )≥ | andP | (E   | )≤δ,itcomesthat |     |     |     |     |
| --------- | ----- | ---- | ---- | --------------- | --- | --- | --- | --- |
|           | ν i   | 2    | ν(i) | i               |     |     |     |     |
(cid:88)
|     |     | E   | (T (τ | ))KL(ν | ,ν(i))≥d(P | (E  | ),P (E )), |     |
| --- | --- | --- | ----- | ------ | ---------- | --- | ---------- | --- |
|     |     |     | ν a   | δ      | a a        | ν i | ν(i) i     |     |
a∈A
hence
|     |     | E   | (T (τ ))KL(ν | ,ν(i))≥d(P |     | (E ),P | (E )), | (22) |
| --- | --- | --- | ------------ | ---------- | --- | ------ | ------ | ---- |
|     |     | ν   | i δ          | i          | i   | ν i    | ν(i) i |      |
whered(x,y) = xlog(x/y)+(1−x)log((1−x)/(1−y))isthebinaryrelativeentropy. Since
| P (E )≥ | 1 andP | (E )≤δ,(22)yields(see[18]), |     |     |     |     |     |     |
| ------- | ------ | --------------------------- | --- | --- | --- | --- | --- | --- |
| ν i     | ν(i)   | i                           |     |     |     |     |     |     |
2
|     |        |           |      | 1       | 1 (cid:18) | (cid:18) 1 (cid:19) | (cid:18) 1 (cid:19)(cid:19) |     |
| --- | ------ | --------- | ---- | ------- | ---------- | ------------------- | --------------------------- | --- |
|     | E ν (T | i (τ δ )) | ≥    |         |            | log +log            |                             |     |
|     |        |           |      | ,ν(i))2 |            | 2δ                  | 2(1−δ)                      |     |
|     |        |           | KL(ν | i       |            |                     |                             |     |
i
|     |     |     |       | 1      |     | (cid:18) 1 | (cid:19) |     |
| --- | --- | --- | ----- | ------ | --- | ---------- | -------- | --- |
|     |     |     | =     |        | log |            |          |     |
|     |     |     |       | ,ν(i)) |     | δ(1−δ)     |          |     |
|     |     |     | 2KL(ν | i      |     |            |          |     |
i
1
|     |     |     | ≥   |     | log(1/δ). |     |     |     |
| --- | --- | --- | --- | --- | --------- | --- | --- | --- |
,ν(i))
|     |     |     | 2KL(ν | i   |     |     |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- |
i
Bydirectalgebra,wecompute(independentmarginalssincethecovarianceisdiagonalI ),
D
|     | ,ν(i))= |     | 1               |     |     |      | D              |     |
| --- | ------- | --- | --------------- | --- | --- | ---- | -------------- | --- |
|     | KL(ν    |     | ∥µ −δ+−ω(k)−γ−µ |     |     | ∥2 = | (−δ+−ω(k)−γ)2. |     |
|     | i       | i   | 2 j             | i   |     | j 2  | 2 i            |     |
=δ+.
Notingthatonthisinstanceallthearmsareoptimal,wehaveforanyarmi,∆ i Finally,letting
i
γ −→0provesthatforanyarmi,
1
|     |     |     | E ν | (T i (τ δ ))≥ |     | log(1/δ), |     |     |
| --- | --- | --- | --- | ------------- | --- | --------- | --- | --- |
D(∆k)2
i
furthernotingthatE(τ )= (cid:80)K E(T (τ ))achievestheproof. Wehavechosenadiagonalmatrix
|     |     | δ   | i=1 | i δ |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
matrixsimplicity,webelievethatchoosingcarefullycorrelatedobjectiveslikein[4]couldgivea
tighterlowerboundespeciallyregardingthedependenceinthedimensionD.
E BestArmIdentification
In this section, we discuss the sample complexity and the performance of APE associated to the
stoppingruleτ1forBAI.NotingthatwhenD =1,theParetosetisjusttheargmaxoverthemeans,
0
BAIandPSIarethesameforuni-dimensionalbandits. Forthissettingweshouldexpectalgorithms
forPSItobecompetitivewithexistingalgorithmsforBAI.Wewillshowthatitisactuallythecasefor
APE.LetD =1andν beaone-dimensionalK-armedbandit. Lettinga ⋆ denotetheuniqueoptimal
armofthebanditν,i.eS⋆
={a },onecaneasilycheckthatthegapsdefinedforPSImatchesthe
⋆
| commonnotionofgapsforBAI.Indeed,foranya̸=a |     |     |     |     |     | ,   |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
⋆
|     |     |     |     | ∆ a := | maxm(a,j), |     |     |     |
| --- | --- | --- | --- | ------ | ---------- | --- | --- | --- |
j∈S⋆
|     |     |     |     | =   | m(a,a | ⋆ )  |     |     |
| --- | --- | --- | --- | --- | ----- | ---- | --- | --- |
|     |     |     |     | =   | µ     | −µ , |     |     |
|     |     |     |     |     | a⋆    | a    |     |     |
and
|     |     |     | ∆ = | min{M(j,a⋆)++∆ |     | }= min∆ | ,   |     |
| --- | --- | --- | --- | -------------- | --- | ------- | --- | --- |
|     |     |     | a⋆  |                |     | j       | j   |     |
|     |     |     |     | j̸=a⋆          |     | j̸=a⋆   |     |     |
whichmatchesthedefinitionofthegapintheone-dimensionalbanditsetting([2,18]). Therefore,
thesamplecomplexityofAPEforBAIcanbededucedfromTheorem1.
Theorem 4. Let δ ∈ (0,1),K ≥ 2 and ν a K−armed bandit with a unique best arm a and
⋆
1-subgaussiandistributions. APEassociatedtobestoppingtimeτ1identifiesthebestarma⋆with
0
probabilityatleast1−δusingatmostthefollowingnumberofsamples
|     |     |     | K           | (cid:18) |     | (cid:18) | (cid:19)(cid:19) |     |
| --- | --- | --- | ----------- | -------- | --- | -------- | ---------------- | --- |
|     |     |     | (cid:88) 88 | 2K(K−1)  |     | 12e      |                  |     |
|     |     |     |             | log      |     | log      | .                |     |
|     |     |     | ∆2          |          | δ   | ∆        |                  |     |
|     |     |     |             | a        |     |          | a                |     |
a=1
25

In particular, the k relaxation is not meaningful in this setting. Under the unique optimal arm
assumption,thealgorithmwillalwaysstopwhenthebestarmhasbeenidentified. Andweremark
| thatfromthedefinitionofω |     |     |     | ’s  |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i
|           |     | ω =      | minM(a | ,j)=   | min∆  |     | =∆ and | ∀i̸=a | ,   | ω <0, |     |
| --------- | --- | -------- | ------ | ------ | ----- | --- | ------ | ----- | --- | ----- | --- |
|           |     | 1        |        | ⋆      |       | j   | a⋆     |       | ⋆   | i     |     |
|           |     |          | j̸=a⋆  |        | j̸=a⋆ |     |        |       |     |       |     |
| soforanyk |     | ≤K,max(ω |        | ,∆ )=∆ | .     |     |        |       |     |       |     |
|           |     |          | k      | a      | a     |     |        |       |     |       |     |
Remark2. Theorem4couldbeslightlyimproved. OntheeventE weconsiderthatforanypairof
armsthedifferenceoftheirempiricalmeandoesnotdeviatetoomuchfromitsactualvalue. ForBAI,
sinceweknowthatthereisauniqueoptimalarm(enforcedbyassumption),itissufficienttocontrol
thedifferencebetweenthebestarmandanyotherarm,thereforewecouldreplacetheK(K−1)/2
termduetounionboundintheconfidencebonusbyK−1andwecouldshowthatthiswillreflectin
thesamplecomplexitybyreplacingK(K−1)by2(K−1). However,thiscannotbedoneingeneral
forPSIsincewedonotknowinadvancethenumberofoptimalarms.
WhenD =1,APEreducestosampleateachroundt,theleastsampledamong
|     |     |     |     |     |     |         | (cid:26) | (cid:27) |     |     |      |
| --- | --- | --- | --- | --- | --- | ------- | -------- | -------- | --- | --- | ---- |
|     |     |     |     | b   | :=  | argmax  | minU     | (t) ,    |     |     | (23) |
|     |     |     |     | t   |     |         | i,j      |          |     |     |      |
|     |     |     |     |     |     | i       | j̸=i     |          |     |     |      |
|     |     |     |     | c   | :=  | argminL | (t),     |          |     |     | (24) |
|     |     |     |     | t   |     |         | bt,j     |          |     |     |      |
j̸=bt
whereU (t):=µ (t)−µ (t)+β (t)andL (t):=µ (t)−µ (t)−β (t)areupperandlower
|     | i,j | (cid:98)i | (cid:98)j |     | i,j | i,j | (cid:98)i | (cid:98)j | i,j |     |     |
| --- | --- | --------- | --------- | --- | --- | --- | --------- | --------- | --- | --- | --- |
bounds on the difference µ i −µ j . To be in the same setting as LUCB and UGapEc which uses
(t),whereβ′sareconfidence
| confidenceintervalonsinglearms,wewouldhaveβ |     |     |     |     |     |     | i,j (t):=β | i (t)+β | j   |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- | ------- | --- | --- | --- |
i
bonusesonsinglearmssuchthatL (t):=µ (t)−β (t)andU (t):=µ (t)+β (t)arelowerand
|                          |     |     |     |                            | i   | (cid:98)i | i   | i   | (cid:98)i | i   |     |
| ------------------------ | --- | --- | --- | -------------------------- | --- | --------- | --- | --- | --------- | --- | --- |
| upperconfidenceboundsonµ |     |     |     | . Then(23)and(24)rewriteas |     |           |     |     |           |     |     |
i
|     |     |     |     |        |        | (cid:26) |              |       | (cid:27) |     |     |
| --- | --- | --- | --- | ------ | ------ | -------- | ------------ | ----- | -------- | --- | --- |
|     |     |     |     | b t := | argmax |          | U i (t)−maxL | j (t) | ,        |     |     |
j̸=i
i
|     |     |     |     | c := | argmaxU |     | (t), |     |     |     |     |
| --- | --- | --- | --- | ---- | ------- | --- | ---- | --- | --- | --- | --- |
|     |     |     |     | t    |         |     | j    |     |     |     |     |
j̸=bt
ThisresemblesthesamplingruleofUGap,whichdefines
|     |     |     |       |     |           |     | (cid:26)   |      | (cid:27) |     |     |
| --- | --- | --- | ----- | --- | --------- | --- | ---------- | ---- | -------- | --- | --- |
|     |     |     | bUGap |     | := argmax |     | L (t)−maxU |      | (t) ,    |     |     |
|     |     |     |       | t   |           |     | i          | j    |          |     |     |
|     |     |     |       |     |           | i   |            | j̸=i |          |     |     |
cUGap
|     |     |     |     |     | := argmaxU |     | (t), |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | ---- | --- | --- | --- | --- |
|     |     |     |     | t   |            |     | i    |     |     |     |     |
j̸=bt
andalsopullstheleastsampledsofar. Wenotethatavariantofouralgorithminwhichbothb t ,c t
wouldbesampled(inthespiritofLUCB[15])couldalsobeanalyzedusingthesamearguments
employedintheproofofTheorem1.
| Notethatwhenε |     | =0,foranyi∈S(t)c,g |     |     |     | (t)>h0(t). | Indeed,bydefinition, |     |     |     |     |
| ------------- | --- | ------------------ | --- | --- | --- | ---------- | -------------------- | --- | --- | --- | --- |
|               |     | 1                  |     |     |     | i          | i                    |     |     |     |     |
h0(t)=min(M(i,j,t)−β
|     |     |     |      |     |     | i,j | (t))=min(−m(i,j,t)−β |     |     | i,j (t)) |     |
| --- | --- | --- | ---- | --- | --- | --- | -------------------- | --- | --- | -------- | --- |
|     |     | i   | j̸=i |     |     |     | j̸=i                 |     |     |          |     |
andsincei∈S(t)c,thereexistsi⋆suchthatm(i,i⋆,t)>0(i.eµ (t)≺µ (t))andso
(cid:98)i (cid:98)i⋆
|     |     |     | −m(i,i⋆,t)−β |     |     | (t)<m(i,i⋆,t)−β |     |     | (t). |     |     |
| --- | --- | --- | ------------ | --- | --- | --------------- | --- | --- | ---- | --- | --- |
|     |     |     |              |     |     | i,i⋆            |     |     | i,i⋆ |     |     |
Therefore,
|          |     | min(−m(i,j,t)−β |     |     | (t)):=h0(t)<max(m(i,j,t)−β |     |      |     | (t)):=g | (t). |     |
| -------- | --- | --------------- | --- | --- | -------------------------- | --- | ---- | --- | ------- | ---- | --- |
|          |     |                 |     | i,j |                            | i   |      |     | i,j     | i    |     |
|          |     | j̸=i            |     |     |                            |     | j̸=i |     |         |      |     |
| Thusforε |     | =0,             |     |     |                            |     |      |     |         |      |     |
1
|     |     |     |     |     | Z0(t)= |     | min g (t). |     |     |     | (25) |
| --- | --- | --- | --- | --- | ------ | --- | ---------- | --- | --- | --- | ---- |
|     |     |     |     |     | 2      |     | i          |     |     |     |      |
i∈S(t)c
(t)insteadofZ0fori=0
| Inthesequel,forthissection,weremovethedependenceonε |     |     |     |     |     |     |     | towriteZ |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- |
|                                                     |     |     |     |     |     |     |     | 1        | i   | i   |     |
andi=1. Inparticular,whenD =1,ε =0,thestoppingtimeτ canbesimplifiedto
|     |     |     |     |     | 1          |     |            | 0   |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | ---------- | --- | --- | --- | --- |
|     |     |     |     |     | τ =inf{t≥K |     | :Z (t)>0}, |     |     |     |     |
|     |     |     |     |     | 0          |     | 1          |     |     |     |     |
whichisaconsequenceofthefollowinglemma.
26

Lemma12. ForD =1,ε =0,
1
inf{t∈N⋆ :Z (t)>0}=inf{t∈N⋆ :Z (t)>0∧Z (t)>0}.
1 1 2
ProofofLemma12. Let S(t) = {a }. Using the definition of h0,g and (25), Z (t) and Z (t)
(cid:98)t i i 1 2
simplifiesto
Z (t) = min{µ (t)−µ (t)−β (t)}, (26)
1
i̸=(cid:98)at
(cid:98)(cid:98)at (cid:98)i (cid:98)at,i
(cid:26) (cid:27)
Z (t) = min max[µ (t)−µ (t)−β (t)] . (27)
2 (cid:98)j (cid:98)i i,j
i̸=(cid:98)at j̸=i
Wehave:
Z (t)>0 =⇒ ∀i̸=a ,µ (t)−µ (t)−β (t)>0,
1 (cid:98)t (cid:98)(cid:98)at (cid:98)i (cid:98)at,i
=⇒ ∀i̸=a ,max[µ (t)−µ (t)−β (t)]>0,
(cid:98)t (cid:98)j (cid:98)i j,i
j̸=i
=⇒ Z (t)>0.
2
Thus,Z (t)>0=⇒(Z (t)>0∧Z (t)>0)andthereverseholdstrivially. So
1 1 2
Z (t)>0⇐⇒(Z (t)>0∧Z (t)>0).
1 1 2
Lettinga denotetheempiricalbestarmaftertrounds,thestoppingruleofAPE(withtheinstantiation
(cid:98)t
proposedinSection3.2basedonconfidenceintervalsonpairsofarms)reducesto
 
τ =inf  t≥K :∀i̸=a , (µ (cid:98)(cid:98)at (t)−µ (cid:98)i (t))2 ≥2Cg (cid:18) log(K 1 /δ) (cid:19) +2 (cid:88) log(4+log(T (t))) 
0  (cid:98)t 2 (cid:16) 1 + 1 (cid:17) 2 a 
Ta(cid:98)t (t) Ti(t) a∈{(cid:98)at,i}
which is very close to a Generalized Likelihood Ratio (GLR) stopping rule assuming Gaussian
distributionswithvariance1fortherewards(whichisknowntobealsocorrectforsub-Gaussian
rewards) [10, 20]. This modified stopping rule compared to LUCB1 and UGapEc can partially
explainstheempiricalimprovementobservedinSectionH.3.
F LUCB1-likeinstantiationofAPE
Inthissectionwederiveanupperboundontheexpectationofthesamplecomplexityτk whenAPE
ε1
isrunwithconfidencebonusessimilartoLUCB1[15]. ThisisdifferentfromTheorem1forwhich
thesamplecomplexityisboundedonlyonthehigh-probabilityeventE butas,formanyalgorithms
inpure-exploration[26,9,4]wedonotcontrolwhathappensonEc. Therefore,ourgoalhereisto
upper-boundE(τ)insteadofE(1{E}τ)whichwedidinTheorem1. Toadaptthestrategyemployed
in[15],weusesimilarconfidencebonuses,thuswedefineforanyarmi,
(cid:115)
2
(cid:18) 5KDt4(cid:19)
β (t)= log , (28)
i T (t) 2δ
i
andforanypairi,j ∈A,β (t)=β (t)+β (t). RecallingthedefinitionofE andE introducedin
i,j i j t
Section3.1,
K D ∞
E := (cid:92) (cid:92) (cid:92) (cid:8) Ld (t,δ)≤µd−µd ≤Ud (t,δ) (cid:9) , and E = (cid:92) E ,
t i,j i j i,j t
i=1j̸=id=1 t=1
the lemma hereafter shows that with the choice of β ’s in (28) and for 1-subgaussian marginals,
i
P(E)≥1−δ.
Lemma13. ItholdsthatP(E)≥1−δ.
27

Proof. Letting
|     |     |     |     |             | K        | D                |            |         |     |     |     |
| --- | --- | --- | --- | ----------- | -------- | ---------------- | ---------- | ------- | --- | --- | --- |
|     |     |     |     |             | (cid:92) | (cid:92) (cid:8) |            | (cid:9) |     |     |     |
|     |     |     |     | E(cid:101)t | :=       | |µ d             | (t)−µd |≤β | (t) ,   |     |     |     |
|     |     |     |     |             |          | (cid:98) i       | i          | i       |     |     |     |
i=1d=1
| wehaveE(cid:101)t  |     | ⊂E. Indeed,onE(cid:101)t |           | ,foranyi,j |           | ∈Aandd≤D,              |                |                  |       |      |      |
| ------------------ | --- | ------------------------ | --------- | ---------- | --------- | ---------------------- | -------------- | ---------------- | ----- | ---- | ---- |
|                    |     | µd(t)−µd(t)−β            |           | (t)−β      | (t)≤µd−µd |                        | ≤µd(t)−µd(t)+β |                  | (t)+β | (t), |      |
|                    |     | (cid:98)i                | (cid:98)j | i          | j         | i                      | j (cid:98)i    | (cid:98)j        | i     | j    |      |
| whichcombinedwithβ |     |                          | i,j       | (t)=β      | i (t)+β   | j (t)yieldsE(cid:101)t | ⊂E             | t so             |       |      |      |
|                    |     |                          |           |            |           | ∞                      | ∞              |                  |       |      |      |
|                    |     |                          |           |            |           | (cid:88)               | (cid:88)       |                  |       |      |      |
|                    |     |                          |           | P(Ec)≤     |           | P(Ec)≤                 |                | P(E(cid:101) c). |       |      | (29) |
|                    |     |                          |           |            |           |                        | t              | t                |       |      |      |
|                    |     |                          |           |            |           | t=1                    | t=1            |                  |       |      |      |
ApplyingHoeffding’sinequalitytothe1-subgaussianmarginalsyields
|     |              |      | K                | D        |          |         |         |     |     |     |     |
| --- | ------------ | ---- | ---------------- | -------- | -------- | ------- | ------- | --- | --- | --- | --- |
|     |              |      | (cid:88)(cid:88) | (cid:0)  |          |         | (cid:1) |     |     |     |     |
|     | P(E(cid:101) | c) ≤ |                  | P |µ     | d (t)−µd | |>β (t) | ,       |     |     |     |     |
|     |              | t    |                  | (cid:98) | i        | i i     |         |     |     |     |     |
i=1d=1
|     |     |     | K                 | D t      |             |            |           |     | (cid:115) | (cid:18) 5KDt4(cid:19) |     |
| --- | --- | --- | ----------------- | -------- | ----------- | ---------- | --------- | --- | --------- | ---------------------- | --- |
|     |     |     | (cid:88) (cid:88) | (cid:88) |             |            |           |     | 2         |                        |     |
|     |     | ≤   |                   | P(|µd    |             | −µd|>βt,s) | whereβt,s | =   | log       |                        | ,   |
|     |     |     |                   |          | (cid:98)i,s | i          |           |     |           |                        |     |
s 2δ
i=1d=1s=1
|     |     |     | K                        | D t |     |     |     |     |     |     |     |
| --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | (cid:88)(cid:88)(cid:88) |     | 4δ  |     |     |     |     |     |     |
|     |     | ≤   |                          |     |     | ,   |     |     |     |     |     |
5KDt4
i=1d=1s=1
4δ
|     |     | =   | .   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
5t3
Finally,
∞
(cid:88)
|     |     |     |     |     | P(Ec) | ≤   | P(E(cid:101) | c), |     |     |     |
| --- | --- | --- | --- | --- | ----- | --- | ------------ | --- | --- | --- | --- |
t
t=1
|     |     |     |     |     |     |     | 4δ ∞ | 1   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
(cid:88)
|     |     |     |     |     |     | ≤   |     | ,   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | 5   | t3  |     |     |     |
t=1
|     |     |     |     |     |     | ≤   | δ.  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Wecannowstatethemaintheoremofthissection.
APErunwiththeβ′s
| Theorem5. |     | Letε 1 | ≥0,k | ≤K andν | abanditwith1-subgaussianmarginals. |     |     |     |     |     |     |
| --------- | --- | ------ | ---- | ------- | ---------------------------------- | --- | --- | --- | --- | --- | --- |
i
of (28)andassociatedtothestoppingtimeτk outputsavalidsetanditsexpectedsamplecomplexity
ε1
isupper-boundedasfollows:
|       |     |           |     |            | (cid:18) | (cid:19) |              |     |     |     |     |
| ----- | --- | --------- | --- | ---------- | -------- | -------- | ------------ | --- | --- | --- | --- |
|       |     |           |     | √          |          | 5KD      | √            |     | 8π2 |     |     |
|       |     | E (τk)≤64 |     | eHlog      |          | +256     | eHlog(256H)+ |     |     | +1, |     |
|       |     | ν         | ε1  |            |          |          |              |     |     |     |     |
|       |     |           |     |            |          | 2δ       |              |     | 15  |     |     |
|       |     | (cid:80)K |     |            | )−2.     |          |              |     |     |     |     |
| withH | :=  | max(∆     |     | at ,ε 1 ,ω | k        |          |              |     |     |     |     |
a=1
ProofofTheorem1. ThecorrectnessfollowsfromLemma2combinedwithLemma13. Itremains
to upper-bound E(τk). Note that this proof technique has been already used in [15, 19, 14] for
ε1
| LUCB-likealgorithms. |     |     | Letn≥1tobespecifiedlaterand |     |     |     |     |     |     |     |     |
| -------------------- | --- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:92)
|     |     |     |     |     | E(n) | =   |     | E . |     |     | (30) |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | ---- |
t
t∈[1n,n]
2
Remarkthat
n
|     |     |          |     |          |     |     | (cid:88) 1{{τk |              |     |     |      |
| --- | --- | -------- | --- | -------- | --- | --- | -------------- | ------------ | --- | --- | ---- |
|     |     | E(n)∩{τk |     | >n}holds |     | =⇒  |                | >t}∩E(n)}=n. |     |     | (31) |
|     |     |          |     | ε1       |     |     |                | ε1           |     |     |      |
t=1
28

Wewillshowthatforsomechoiceofn,theRHSof(31)willbestrictlylessthannsotheLHSdoes
nothold. Weproceedbyupper-boundingtheRHS
|     | n        |           |     |     | n        |            |     |     |      |
| --- | -------- | --------- | --- | --- | -------- | ---------- | --- | --- | ---- |
|     | (cid:88) |           |     | n   | (cid:88) |            |     |     |      |
|     | 1{{τk    | >t}∩E(n)} | ≤   | +   | 1{{τk    | >t}∩E(n)}, |     |     | (32) |
|     | ε1       |           |     |     |          | ε1         |     |     |      |
2
|     | t=1 |     |     |     | t=n |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2
|     |     |     |     | n   | (cid:88) n |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
1{{τk
|     |     |     | ≤   | +   |     | >t}∧E | t }. |     | (33) |
| --- | --- | --- | --- | --- | --- | ----- | ---- | --- | ---- |
|     |     |     |     | 2   |     | ε1    |      |     |      |
t=n
2
FromLemma4,Lemma5andLemma3,wehavethatforanyt∈[n,n],
2
|     | {τk >t}∩E |     | =⇒max(∆ | ,ε  | ,ω )≤2β |       | (t), |     |     |
| --- | --------- | --- | ------- | --- | ------- | ----- | ---- | --- | --- |
|     | ε1        |     | t       | at  | 1 k     | at,at |      |     |     |
withβ (t)=2β (t). Therefore,usingthisresultbackin(32)andlettingc :=(5KD/(2δ))1/4,
| at,at               | at            |     |            |                |     |      | δ   |     |     |
| ------------------- | ------------- | --- | ---------- | -------------- | --- | ---- | --- | --- | --- |
| ∆(cid:101)a :=max(∆ | ,ε ,ω )yields |     |            |                |     |      |     |     |     |
|                     | a 1 k         |     |            |                |     |      |     |     |     |
| n                   |               |     | n          |                |     |      |     |     |     |
| (cid:88)            |               |     | n (cid:88) |                |     |      |     |     |     |
| 1{{τ                | k >t}∩E(n)}   | ≤   | +          | 1{∆(cid:101)at | ≤4β | (t)} |     |     |     |
|                     | ε             |     |            |                | at  |      |     |     |     |
|                     | 1             |     | 2          |                |     |      |     |     |     |
| t=1                 |               |     | t=n        |                |     |      |     |     |     |
2
n K
n (cid:88)(cid:88)
|     |     | ≤   | +   | 1{(a | =a)∧∆(cid:101)a | ≤4β | (t)}, |     |     |
| --- | --- | --- | --- | ---- | --------------- | --- | ----- | --- | --- |
|     |     |     | 2   |      | t               |     | a     |     |     |
t=na=1
2
K n
|     |     |     | n (cid:88) | (cid:88) |        |     | 128         |      |     |
| --- | --- | --- | ---------- | -------- | ------ | --- | ----------- | ---- | --- |
|     |     | ≤   | +          | 1{{a     | =a}∧{T |     | (t)≤ log(c  | t)}} |     |
|     |     |     |            |          | t      | a   |             | δ    |     |
|     |     |     | 2          |          |        |     | ∆(cid:101)2 |      |     |
|     |     |     | a=1t=1n    |          |        |     | a           |      |     |
2
K n
|     |     |     | n (cid:88)(cid:88) |      |        |     | 128         |      |     |
| --- | --- | --- | ------------------ | ---- | ------ | --- | ----------- | ---- | --- |
|     |     | ≤   | +                  | 1{{a | =a}∧{T |     | (t)≤ log(c  | n)}} |     |
|     |     |     |                    |      | t      | a   |             | δ    |     |
|     |     |     | 2                  |      |        |     | ∆(cid:101)2 |      |     |
|     |     |     | a=1t=n             |      |        |     | a           |      |     |
2
|     |     |     | n (cid:88)128 K |             |      |     |     |     |     |
| --- | --- | --- | --------------- | ----------- | ---- | --- | --- | --- | --- |
|     |     | ≤   | +               | log(c       | δ n) |     |     |     |     |
|     |     |     | 2               | ∆(cid:101)2 |      |     |     |     |     |
a
a=1
n
|     |     | ≤   | +128Hlog(c |     | n), |     |     |     |     |
| --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
δ
2
| whereH := (cid:80) | ∆(cid:101)−2. Then,choosingnsuchthat |     |     |     |     |     |     |     |     |
| ------------------ | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
|                    | a a                                  |     |     |     |     |     |     |     |     |
n
|     |     |     | +128Hlog(c |     | n)<n, |     |     |     |     |
| --- | --- | --- | ---------- | --- | ----- | --- | --- | --- | --- |
|     |     |     | 2          | δ   |       |     |     |     |     |
thatis
|     |      |       | (cid:26) |           |     |      | (cid:27) |     |      |
| --- | ---- | ----- | -------- | --------- | --- | ---- | -------- | --- | ---- |
|     |      |       |          | 128Hlog(c |     | s) 1 |          |     |      |
|     | n>T⋆ | :=inf | s∈N⋆     | :         |     | δ <  | ,        |     | (34) |
|     |      |       |          |           | s   | 2    |          |     |      |
wouldyield
n
(cid:88)
|     |     |     | 1{{τk >t}∩E(n)}<n, |     |     |     |     |     | (35) |
| --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | ---- |
ε1
t=1
so
|     |     |     | E(n)∩{τk | >n}=∅, |     |     |     |     |     |
| --- | --- | --- | -------- | ------ | --- | --- | --- | --- | --- |
ε1
whichmeans
|     |     |     | {τk >n}⊂E(n)c. |     |     |     |     |     |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
ε1
Therefore,foranyn>T⋆,
|     |     |     | {τk >n}⊂E(n)c. |     |     |     |     |     |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
(36)
ε1
Thus,
|     | (τk) |        | (cid:0) τk1{τk | ≤T⋆}+τk1{τk |      | >T⋆} | (cid:1) |     |     |
| --- | ---- | ------ | -------------- | ----------- | ---- | ---- | ------- | --- | --- |
|     | E    | = E    |                |             |      |      |         |     |     |
|     | ν ε1 |        | ν ε1           | ε1          | ε1   | ε1   |         |     |     |
|     |      | ≤ T⋆+E | (τk1{τk        |             | >T⋆} |      |         |     |     |
|     |      |        | ν              | ε1 ε1       |      |      |         |     |     |
∞
|     |     | T⋆+ | (cid:88) | (τk |     |     |     |     |     |
| --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
|     |     | ≤   |          | P   | >n) |     |     |     |     |
ν ε1
n=T⋆+1
∞
|     |     | T⋆+ | (cid:88) | (E(n)c), |     |     |     |     |     |
| --- | --- | --- | -------- | -------- | --- | --- | --- | --- | --- |
|     |     | ≤   |          | P ν      |     |     |     |     |     |
n=T⋆+1
29

using(30)andunionboundyields,
n
(cid:88) 4δ
P(E(n)c) ≤ ,
5t3
t=n
2
4δ (1/2)n
≤ ,
5 (1/2)3n3
16δ 1
= .
5 n2
Then,
16δπ2
E (τk) ≤ T⋆+
ν ε1 5 6
8π2
≤ T⋆+ .
15
Upper-boundingT⋆willconcludetheproof.
Lemma14. Itholdsthat
1 (cid:18) (cid:18) 1 (cid:19)(cid:19) √
T⋆−1≤ exp −W − ≤256 eHlog(256c H).
c −1 256c H δ
δ δ
Finally,
√ (cid:16) (cid:17) 8π2
E (τ ) ≤ 256 eHlog 256(5KD/(2δ))1/4H + +1, (37)
ν δ 15
√ (cid:18) 5KD (cid:19) √ 8π2
≤ 64 eHlog +256 eHlog(256H)+ +1, (38)
2δ 15
whichachievestheproof.
Remark3. Thesametechniquecouldbeappliedtoupper-boundE (τ ).
ν ε1,ε2
NowweproveLemma14.
ProofofLemma14. Wehave
log(c s) 1 log(c s) 1
128H δ < ⇐⇒ δ < (39)
s 2 s 256H
then,usingLemma15yields
(cid:40)
s>0 if 1 <c /e
(39) =⇒ 256H δ
0<s≤ 1 ors≥N⋆ else,
cδ
with
(cid:18) (cid:18) (cid:19)(cid:19)
1 1
N⋆ = exp −W − .
c −1 256c H
δ δ
Therefore,
(cid:26) log(c s) 1 (cid:27) (cid:26) 1 if 1 <c /e
T⋆ =inf s∈N⋆ :128H δ < ≤ 256H δ
s 2 ⌈N⋆⌉ else.
UsingCorollary1,toupperboundN⋆yields
√
T⋆−1≤256 eHlog (256c H), (40)
+ δ
wherelog (x)=max(0,log(x)).
+
30

G TechnicalLemmas
| Lemma15. | Leta,b>0. |         | Ifb<a/ethen |     |     |      |          |     |                           |
| -------- | --------- | ------- | ----------- | --- | --- | ---- | -------- | --- | ------------------------- |
|          |           |         |             |     |     |      | (cid:18) |     | (cid:18) (cid:19)(cid:19) |
|          |           | log(ax) |             |     | 1   | 1    |          |     | b                         |
|          |           |         | <b=⇒0<x≤    |     |     | orx≥ | exp      | −W  | − .                       |
|          |           | x       |             |     | a   | a    |          | −1  | a                         |
Moreover,ifb≥a/e,thenforanyx>0,log(ax)/x≤b.
Proof. Wehave
|     |     |         |     |     |           | (cid:18) | (cid:19)   |     |     |
| --- | --- | ------- | --- | --- | --------- | -------- | ---------- | --- | --- |
|     |     | log(ax) |     |     |           | 1        | 1          | b   |     |
|     |     |         | <b  | =⇒  | −         | log      | <          |     |     |
|     |     | x       |     |     | ax        | ax       |            | a   |     |
|     |     |         |     |     |           |          | b          | 1   |     |
|     |     |         |     | =⇒  | ylog(y)>− |          | ,          | y = |     |
|     |     |         |     |     |           |          | a          | ax  |     |
|     |     |         |     |     | 1         |          | b          |     |     |
|     |     |         |     | =⇒  |           | ≥1or −   | <ylog(y)<0 |     |     |
|     |     |         |     |     | ax        |          | a          |     |     |
since−b/a>−1/eandthenegativebranchW −1 oftheLambertfunctionisdecreasingon[−1/e,0),
|     |     | log(ax) |     |      |     | 1   |             |     |        |
| --- | --- | ------- | --- | ---- | --- | --- | ----------- | --- | ------ |
|     |     | <b      | =⇒  | 0<x≤ |     | orW | (ylog(y))≤W |     | (−b/a) |
|     |     |         |     |      |     | −1  |             |     | −1     |
|     |     | x       |     |      |     | a   |             |     |        |
1
|     |     |     | =⇒  | 0<x≤ |     | or log(y)≤W |     | (−b/a) |     |
| --- | --- | --- | --- | ---- | --- | ----------- | --- | ------ | --- |
|     |     |     |     |      |     | a           |     | −1     |     |
1
|     |     |     | =⇒  | 0<x≤ |     | orax≥exp(−W |        | (−b/a)) |          |
| --- | --- | --- | --- | ---- | --- | ----------- | ------ | ------- | -------- |
|     |     |     |     |      |     | a           |        | −1      |          |
|     |     |     |     |      |     | 1           | 1      |         |          |
|     |     |     | =⇒  | 0<x≤ |     | orx≥        | exp(−W | −1      | (−b/a)). |
|     |     |     |     |      |     | a           | a      |         |          |
Provingthesecondpartofthelemmajustfollowsfromlog(x)≤x/e.
Thefollowinglemmaistakenfrom[12]
| Lemma16([12]). |     | Foranyx∈[−e−1,0), |     |     |     |     |     |     |     |
| -------------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
(cid:40) (cid:41)
1 1
−log(−x)+log(−log(−x))≤−W −1 (x)≤−log(−x)+log(−log(−x))+min , (cid:112)
2 −xlog(−x)
| Corollary1. | Let0<a<1/e. |     | Itholdsthat |     |     |      |          |          |     |
| ----------- | ----------- | --- | ----------- | --- | --- | ---- | -------- | -------- | --- |
|             |             |     |             |     |     | e1/2 | (cid:18) | (cid:19) |     |
1
|     |     |     | exp(−W |     | (−a))≤ |     | log | .   |     |
| --- | --- | --- | ------ | --- | ------ | --- | --- | --- | --- |
|     |     |     |        |     | −1     | a   |     | a   |     |
Proof. UsingLemma16yields,
1
|     |     | −W  |     | (−a)≤−log(a)+log(−log(a))+ |     |     |     |     | ,   |
| --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- |
−1
2
andtakingexponbothsidesgivestheresult.
| Lemma17. | Let∆2 | >0. | Then,fort≥2, |          |              |                  |        |                     |     |
| -------- | ----- | --- | ------------ | -------- | ------------ | ---------------- | ------ | ------------------- | --- |
|          |       |     |              | (cid:18) | (cid:18) 3e2 | (cid:19)(cid:19) | loglog | (cid:0) e4t (cid:1) |     |
1
|     |     | t≥  | log | 2log |     | =⇒  |     |     | <∆2. |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | ---- |
|     |     |     | ∆2  |      | 2∆2 |     |     | t   |      |
Proof. Wenotethatif∆2 ≥ e,thentheresultfollowstriviallysinceitcanbeeasilycheckedthatfor
3
t≥2,
e
loglog(e4t)≤
t.
3
Therefore,inthesequel,weassume∆2
|     |     |     |     |      | <e/3. | Let               |                  |     |     |
| --- | --- | --- | --- | ---- | ----- | ----------------- | ---------------- | --- | --- |
|     |     |     |     |      |       | (cid:18) (cid:18) | (cid:19)(cid:19) |     |     |
|     |     |     |     |      | 1     |                   | 3e2              |     |     |
|     |     |     |     | t := | log   | 2log              |                  | ,   |     |
∆
|     |     |     |     |     | ∆2  |     | 2∆2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
31

and
1
log(log(e4t)).
g(t)=t−
∆2
Then,
1
|     |     |     |     | g′(t)=1− |     |     |     | ,   |     |     |
| --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
∆2tlog(e4t)
andg′(t)≥0fortsuchthat∆2tlog(e4t)≥1. UsingtheLambertfunctionW ,whichisincreasing
0
on[0,∞),
e4
|     |     | ∆2tlog(e4t)≥1 |     | ⇐⇒  | e4tlog(e4t)≥ |     |     |     |     | (41) |
| --- | --- | ------------- | --- | --- | ------------ | --- | --- | --- | --- | ---- |
∆2
|     |     |     |     |     |     |     |     | (cid:18) (cid:19) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- |
e4
|     |     |     |     | ⇐⇒  | log(e4t)≥W |     |     |     |     | (42) |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | ---- |
0
∆2
|     |     |     |     |     |     |     |     | (cid:18) (cid:18) | e4 (cid:19)(cid:19) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | ------------------- | --- |
1
|                    |     |         |     | ⇐⇒  | t≥t0 | :=  | exp | W   |     | (43) |
| ------------------ | --- | ------- | --- | --- | ---- | --- | --- | --- | --- | ---- |
|                    |     |         |     |     |      |     | e4  | 0   | ∆2  |      |
| andbydefinitionofW |     | ,wehave |     |     |      |     |     |     |     |      |
0
|     |     |     |     | W   | (x)exp(W | (x))=x, |     |     |     |     |
| --- | --- | --- | --- | --- | -------- | ------- | --- | --- | --- | --- |
|     |     |     |     | 0   |          | 0       |     |     |     |     |
so
|     |     |     |     | (cid:18) (cid:18) | e4 (cid:19)(cid:19) | e4  |         | 1   |     |     |
| --- | --- | --- | --- | ----------------- | ------------------- | --- | ------- | --- | --- | --- |
|     |     |     | exp | W                 |                     | =   |         | ,   |     |     |
|     |     |     |     | 0                 | ∆2                  | ∆2W | (e4∆−2) |     |     |     |
0
andtherefore,
|     |     |     |     |     | 1   | 1       |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
|     |     |     |     | t0  | =   |         | .   |     |     |     |
|     |     |     |     |     | ∆2W | (e4∆−2) |     |     |     |     |
0
| Wewillshowthatt |     | >t0. | Indeed,sinceW |     | isincreasing, |                      |     |     |     |     |
| --------------- | --- | ---- | ------------- | --- | ------------- | -------------------- | --- | --- | --- | --- |
|                 |     | ∆    |               |     | 0             |                      |     |     |     |     |
|                 |     |      | 1             |     |               | (cid:18) e4 (cid:19) |     |     |     |     |
(3e3)=3
|     |     |     | >3/e | =⇒  | W 0 |         | ≥W  | 0   |     |     |
| --- | --- | --- | ---- | --- | --- | ------- | --- | --- | --- | --- |
|     |     |     | ∆2   |     |     | ∆2      |     |     |     |     |
|     |     |     |      |     | 1   |         | 1   | 1 1 |     |     |
|     |     |     |      | =⇒  |     |         |     | ≤   | ,   |     |
|     |     |     |      |     | ∆2W | (e4∆−2) |     | 3∆2 |     |     |
0
thatis
1 1
|     |     |     |     |     | t0 ≤ | .   |     |     |     | (44) |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | ---- |
3∆2
Ontheotherside,
|     |     |     |     |     | (cid:18) | (cid:18) 3e2 | (cid:19)(cid:19) |     |     |     |
| --- | --- | --- | --- | --- | -------- | ------------ | ---------------- | --- | --- | --- |
1
|     |     | >3/e | =⇒  | log | 2log |                  | >log(2log(9e/2)) |     |     |     |
| --- | --- | ---- | --- | --- | ---- | ---------------- | ---------------- | --- | --- | --- |
|     |     | ∆2   |     |     |      | 2∆2              |                  |     |     |     |
|     |     |      |     |     | 1    |                  |                  | 1   | 1   |     |
|     |     |      | =⇒  | t   | ≥    | log(2log(9e/2))> |                  |     |     |     |
∆
|     |     |     |     |     | ∆2  |     |     | 3∆2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Therefore,
|     |     |     |     | 1 1 | log(2log(9e/2)) |     |     |      |     |     |
| --- | --- | --- | --- | --- | --------------- | --- | --- | ---- | --- | --- |
|     |     |     | t0  | ≤   | <               |     |     | ≤t . |     |     |
∆
|                        |     |                           |     | 3∆2          |            | ∆2                    |     |     |     |      |
| ---------------------- | --- | ------------------------- | --- | ------------ | ---------- | --------------------- | --- | --- | --- | ---- |
| Thus,wehaveshownthatt0 |     |                           | ≤t  | andforanyt≥t |            | ,g′(t)≥0so            |     |     |     |      |
|                        |     |                           |     | ∆            |            | ∆                     |     |     |     |      |
|                        |     |                           |     | ∀t≥t         | , g(t)≥g(t |                       | ).  |     |     | (45) |
|                        |     |                           |     |              | ∆          |                       | ∆   |     |     |      |
| Showingthatg(t         |     | )>0,willconcludetheproof. |     |              |            | Lettinga=3e2/2,wehave |     |     |     |      |
∆
|     |     |       |     | 1                |     |     |     | 1           |        |      |
| --- | --- | ----- | --- | ---------------- | --- | --- | --- | ----------- | ------ | ---- |
|     |     |       |     | log(2log(a/∆2))− |     |     |     | log(log(e4t |        |      |
|     | g(t | ∆ )>0 | ⇐⇒  |                  |     |     |     |             | ∆ ))>0 | (46) |
|     |     |       |     | ∆2               |     |     | ∆2  |             |        |      |
log(2log(a/∆2))−log(log(e4t
|     |     |     | ⇐⇒  |     |     |     |     | ∆ ))>0 |     | (47) |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---- |
2log(a/∆2)−log(e4t
|     |     |     | ⇐⇒  |                   |     |     | ∆ )>0   |     |     | (48) |
| --- | --- | --- | --- | ----------------- | --- | --- | ------- | --- | --- | ---- |
|     |     |     |     | log(a/∆2)−log(e4t |     |     | ∆2/a)>0 |     |     |      |
|     |     |     | ⇐⇒  |                   |     |     | ∆       |     |     | (49) |
e4
a
|     |     |     | ⇐⇒  | −   | ∆2t | >0  |     |     |     | (50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
|     |     |     |     | ∆2  | a   | ∆   |     |     |     |      |
e4
a
|     |     |     | ⇐⇒  | −   | log(2log(a/∆2))>0, |     |     |     |     | (51) |
| --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | ---- |
|     |     |     |     | ∆2  | a                  |     |     |     |     |      |
32

then,observingthatforx≥12,
x
|     |     |     |     |     | log(2log(x))≤ |     |     | ,   |     |     |     |
| --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
e2
andsince
|     |     |     |     | a/∆2 | >(3e2/2)×(3/e)>12, |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ------------------ | --- | --- | --- | --- | --- | --- |
wehave
1 a
log(2log(a/∆2))≤
(52)
e2∆2
so,using(52)yieldsthattheLHSof(51)islargerthan
|                        |     |     |            |     |      | 3 1  | 1        |     |     |     |     |
| ---------------------- | --- | --- | ---------- | --- | ---- | ---- | -------- | --- | --- | --- | --- |
|                        |     |     |            |     |      | e2   | −e2      | ,   |     |     |     |
|                        |     |     |            |     |      | 2 ∆2 | ∆2       |     |     |     |     |
| whichisalwayspositive. |     |     | Therefore, |     |      |      |          |     |     |     |     |
|                        |     |     |            |     | ∀t≥t | ,    | g(t )>0, |     |     |     |     |
|                        |     |     |            |     |      | ∆    | ∆        |     |     |     |     |
thatis
loglog(e4t)
|     |     |     |     |     | ∀t≥t | ,   |     | <∆2. |     |     | (53) |
| --- | --- | --- | --- | --- | ---- | --- | --- | ---- | --- | --- | ---- |
∆
t
(cid:113)
Let δ ∈ (0,1),∆ > 0 and c > 0. Let f : t (cid:55)→ g(δ)+cloglog(e4t) where g is a
Lemma 18.
t
| non-negativefunction. |     |     | Then,foranyα∈(0,1)andt≥2, |     |     |          |          |     |                          |           |      |
| --------------------- | --- | --- | ------------------------- | --- | --- | -------- | -------- | --- | ------------------------ | --------- | ---- |
|                       |     |     | (cid:18)                  |     |     | (cid:18) | (cid:18) |     | (cid:19)(cid:19)(cid:19) |           |      |
|                       |     | 1   | 1                         |     | c   |          |          | c   |                          |           |      |
|                       | t≥  |     | g(δ)+                     |     | log | 2log     |          |     |                          | =⇒f(t)<∆. | (54) |
+
|     |     | ∆2  | α   | 1−α |     |     | (1−α)∆2 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
Proof. Lettingt≥2,wehave
|     |     |     |     |     | 1     | 1      | g(δ) |       |     |     |      |
| --- | --- | --- | --- | --- | ----- | ------ | ---- | ----- | --- | --- | ---- |
|     |     |     |     | t≥t | :=    | g(δ)=⇒ |      | ≤α∆2. |     |     | (55) |
|     |     |     |     |     | 1 α∆2 |        |      |       |     |     |      |
t
Furthermore,usingLemma17yields
|     |     | c   |     | (cid:18) | (cid:18) | 3e2 | (cid:19)(cid:19) | loglog(e4t) |     |     |     |
| --- | --- | --- | --- | -------- | -------- | --- | ---------------- | ----------- | --- | --- | --- |
≤(1−α)∆2/c.
| t≥t | 2 := |         | log | 2log |          |     |     | =⇒  |     |     | (56) |
| --- | ---- | ------- | --- | ---- | -------- | --- | --- | --- | --- | --- | ---- |
|     |      | (1−α)∆2 |     | +    | 2(1−α)∆2 |     |     |     | t   |     |      |
Combining(55)and(56)yieldsfort≥2,
|     |     |     |     | t≥max(t |     | ,t )=⇒f(t)2 |     | <∆2, |     |     |     |
| --- | --- | --- | --- | ------- | --- | ----------- | --- | ---- | --- | --- | --- |
|     |     |     |     |         | 1   | 2           |     |      |     |     |     |
so
|          |            |     |     | t≥t     | +t ≥max(t |     | ,t )=⇒f(t)<∆. |     |     |     | (57) |
| -------- | ---------- | --- | --- | ------- | --------- | --- | ------------- | --- | --- | --- | ---- |
|          |            |     |     | 1       | 2         |     | 1 2           |     |     |     |      |
| Lemma19. | Let∆>0andδ |     |     | ∈(0,1). | Let       |     |               |     |     |     |      |
(cid:114)
2Cg(log(1/δ)/2)+4loglog(e4t)
|     |     |     | f(t):=4 |     |     |     |     |     |     | .   |     |
| --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
t
Then
|     |     |     |                  |     |     |     |     | (cid:18) | (cid:18) | (cid:19)(cid:19) |     |
| --- | --- | --- | ---------------- | --- | --- | --- | --- | -------- | -------- | ---------------- | --- |
|     |     |     |                  |     |     |     | 88  | 4        | 12e      |                  |     |
|     |     |     | inf{n≥2:f(n)<∆}≤ |     |     |     | log | log      |          | .                |     |
∆2
|     |     |     |     |     |     |     |     | δ   | ∆   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Proof. Wehave
(cid:114)
32Cg(log(1/δ)/2)+64loglog(e4t)
|     |     |     | f(t)= |     |     |     |     |     | .   |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
t
Therefore,letting
|     |     |     | g(δ):=32Cg(log(1/δ)/2) |     |     |     |     | and | c=64 |     |     |
| --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | ---- | --- | --- |
33

andfurtherusingLemma18yieldsforanyα∈(0,1)andt≥2,
|     |     |     |     | t≥t | α =⇒f(t)<∆, |     |     |     |     |
| --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- |
where
|     |     | (cid:18)        |     |     |     |                      |     | (cid:19) |     |
| --- | --- | --------------- | --- | --- | --- | -------------------- | --- | -------- | --- |
|     | 1   | 32              |     |     | 64  | (cid:0)              |     | (cid:1)  |     |
| t   | :=  | Cg(log(1/δ)/2)+ |     |     | log | 2log(96e2(1−α)−1∆−2) |     | .        |     |
α ∆2
|     |     | α   |     |     | 1−α |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
SinceCg(x)≈x+log(x)[20],andlog(x)≤x/ewehave
|     |     | (cid:18) | 16 (cid:19) | 1         | 64                         |     |     |     |     |
| --- | --- | -------- | ----------- | --------- | -------------------------- | --- | --- | --- | --- |
| ∆2t |     |          |             |           | log(2log(96e2(1−α)−1∆−2)). |     |     |     |     |
|     | α ≤ | 16+      |             | log(1/δ)+ |                            |     |     |     |     |
|     |     |          | e α         |           | 1−α                        |     |     |     |     |
Takingα=α⋆suchthat
|     |     |     |     | (cid:18) | 16 (cid:19) 1 | 64  |     |     |     |
| --- | --- | --- | --- | -------- | ------------- | --- | --- | --- | --- |
|     |     |     |     | 16+      | =             | ,   |     |     |     |
|     |     |     |     |          | e α⋆ 1−α⋆     |     |     |     |     |
thatissetting
1+e
α⋆
|     |     |     |     |     | =   | ,   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1+5e
yields
|     |     |     |     |      | (cid:18) (cid:18) | 96e2     | (cid:19)(cid:19) |     |     |
| --- | --- | --- | --- | ---- | ----------------- | -------- | ---------------- | --- | --- |
|     |     |     |     | 64   | 2                 |          |                  |     |     |
|     |     | ∆2t | ≤   |      | log log           |          | .                |     |     |
|     |     |     | α⋆  | 1−α⋆ | δ                 | (1−α⋆)∆2 |                  |     |     |
Bynumericalevaluation,
|     |     | 64  |     |     |     | 96  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
≈130<122,
|     |     |      | ≈86<88 |     | and  |     |     |     |     |
| --- | --- | ---- | ------ | --- | ---- | --- | --- | --- | --- |
|     |     | 1−α⋆ |        |     | 1−α⋆ |     |     |     |     |
so
|     |     |     |     |        | (cid:18) | (cid:18) (cid:19)(cid:19) |     |     |      |
| --- | --- | --- | --- | ------ | -------- | ------------------------- | --- | --- | ---- |
|     |     |     |     |        | 4        | 12e                       |     |     |      |
|     |     |     | ∆2t | <88log | log      | .                         |     |     | (58) |
α⋆
|     |     |     |     |     | δ   | ∆   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Therefore,puttingtheseresultstogether,fort≥2
|     |     |     |     | (cid:18) | (cid:18) | (cid:19)(cid:19) |     |     |     |
| --- | --- | --- | --- | -------- | -------- | ---------------- | --- | --- | --- |
|     |     |     |     | 88       | 4 12e    |                  |     |     |     |
|     |     | t≥t | :=  | log      | log      | =⇒f(t)<∆         |     |     |     |
|     |     |     | ⋆   | ∆2       | δ ∆      |                  |     |     |     |
whichyields
|     |     |     | inf{n≥2:f(n)<∆}≤max(2,t |     |     | ⋆ ). |     |     | (59) |
| --- | --- | --- | ----------------------- | --- | --- | ---- | --- | --- | ---- |
H ImplementationandAdditionalExperiments
Inthissection,wegiveadditionaldetailsabouttheexperimentsandadditionalexperimentalresults.
H.1 Implementation
Setup WehaveimplementedthealgorithmsmainlyinC++17compiledwithGCC12andinterfaced
withpythonthroughthecythonpackage. TheexperimentsarerunonanARM648GBRAM/8
core/256GBdiskstoragecomputer. ForthefunctionCg wehaveusedtheapproximationCg(x)≈
x+log(x)whichisusuallyadmitted[20]. Fortheexperimentsonreal-worldscenariowegeneratea
certainnumberofseeds(usually2000)andweuseadifferentseedforeachrunonthesamebandit.
Thisprocedureisidenticalforeveryexperimentwherewereporttheaveragesamplecomplexity
on the same bandit. To assess the robustness of our algorithm, the experiments on the synthetic
datasetconsistedinrandomlyuniformlysamplingsomebanditmeansforeachconfiguration. For
eachsampledbandit,thealgorithmscomparedarerunonceonthesameinstanceandwenotetheir
empiricalsamplecomplexity. Finally,wereporttheaveragesamplecomplexityacrossallthebandits
ofthesameconfiguration.
34

Timeandmemorycomplexity Thetimecomplexityofε -APE-k isO(K2D)anditsmemory
1
complexityisO(K2). ThemaincomputationalbottleneckisthecomputationoftheM(i,j,t)for
each (i,j) ∈ K ×K, which requires a triple-nested for-loop over [K]×[K]×[D]. To give an
idea of the runtime, a single run on a random Bernoulli instance with K = 1000,D = 10 takes
around4minutesfor0.1−APE-1000onapersonalcomputer(asingle3GHzARMcoreused,8GB
RAM,256GBdiskstorage)withδ =0.01andnoparticularoptimization. Dueitsfullysequential
nature,ouralgorithmmayhaveahighercomputationalcostcomparedtouniformsamplingstrategies.
However,inourimplementationthemosttime-consumingoperationwasactuallycollectingasample
fromtheselectedarm(s),especiallyformultivariateGaussians. Sothatfinally,intheexperiments,
ouralgorithmwhichultimatelyrequirelesssampleshadinpracticeasimilarcomputationalcost
comparedtoPSI-Unif-Elimwhichusesuniformsampling.
Adaptationtobanditswithmarginalsofdifferentscaling Wehavepresentedthealgorithmand
theresultsspecializedtothecasewhereallthemarginalsare1-subgaussian. Indeedourresultscan
besimplyextendedwherethemarginalsareinsteadallσ-subgaussian. Furthermore,thereisasimple
way to adapt the algorithm to the case where the marginals have different known subgaussianity
parameter (i.e different scaling) but they are the same for every arm. The idea is to rescale each
observationwiththesubgaussianityparameters. Letσ := σ ,...σ ,σ > 0. Assumingthatthe
1 D i
marginaldistributionsofeacharmarerespectivelyσ ,...σ -subgaussian,eachobservationX ,s
1 D At
fromarmA willberescaledcomponent-wisetoXd /σ beforebeinggiventothealgorithm. Itis
t At,s d
easytoseethatthisrescalingdoesnotchangetheParetosetsinceallthemeansaredividedbythe
samevaluescoordinate-wise.
Furthermore,bydefining
(cid:32) (cid:33)
µd−µd
Mσ(i,j) := max i j ,
d σ d
andmσ(i,j)=:−Mσ(i,j),alltheresultsprovedfor1-subgaussiandistributionsstillholdsusing
mσ andMσ inthedefinitionofthegaps(Section4).
H.2 Dataprocessing
Dataset Thedatasetisextractedfrom[24]andsomeprocessingstepsareappliedtocomputethe
covariancematrixofthedistribution. First,asobservedin[24],the3immunogenicityindicators
extracted are weakly correlated, therefore, we assume the covariance matrix to be diagonal. To
computethevarianceofthemarginals,weusethelog-normalassumptionasassumedforthedata
reportedin[24]. Usingthislog-normalassumptiontheauthorshaveprovidedforeacharmandeach
indicator: thegeometricalmean,thesamplesizeanda95%confidenceintervalonthegeometrical
meanbasedonthecentrallimittheorem.
ForeachoftheK =20arms(combinationofthreedoses),weusetheseinformationtocomputethe
samplevarianceofeachimmunogenicityindicator. Moreover,wecomputethearithmeticaverageof
thelogoutcomeswhichisobtainedbytakingthelog(basee)ofthegeometricalempiricalmean:
x¯ = log(x¯ ),
geometrical
(cid:32) n (cid:33)1/n 
(cid:89)
= log x
i
,
i=1
n
(cid:88)
= n−1 log(x ),
i
i=1
wherex ,...,x aretheobservationswhichareassumedtobelog-normal. x¯representsbyassump-
1 n
tiontheempiricalmeanofaGaussiandistribution,whichweuseasaproxyforitstrue,unknown
mean. Fromthis,webuiltabanditmodelwhereeacharmisa3-dimensionalGaussiandistribution
withindependentcoordinates,whosemeansaregivenbythecorrespondingmeanestimates(reported
inTable4)andinwhichthevarianceofeachindicatoristhepooledvarianceoverthedifferentarms
(giveninTable5). Samplinganarminthisbanditsimulatesthemeasurementofthe(logofthe)3
immunogenicitycriteriainconsiderationonanewpatient.
35

The 20 arms are classified into two groups. Each three/four-letters acronym denotes a vaccine
candidate. PrimeBNT/BNTcorrespondstogivingBNTasfirstandseconddoseandsimilarlyfor
PrimeChAd/ChAd. ForexampleChAdinthegroupPrimeBNT/BNTmeanstogiveBNTasfirst
andseconddoseandChAdasthirddose(booster).
Table4: Tableoftheempiricalarithmeticmeanofthelog-transformedimmuneresponseforthree
immunogenicityindicators. Eachacronymcorrespondstoavaccine. Therearetwogroupsofarms
correspondingtothefirst2doses: onewithprimeBNT/BNT(BNTasfirstandseconddose)and
thesecondwithprimeChAd/ChAd(ChAdasfirstandseconddose). Eachrowinthetablegivesthe
valuesofthe3immuneresponsesforanarm(i.e. acombinationofthreedoses).
Immuneresponse
| Dose1/Dose2 | Dose3(booster) |               |      |                     |
| ----------- | -------------- | ------------- | ---- | ------------------- |
|             |                | Anti-spikeIgG | NT   | 50 cellularresponse |
|             | ChAd           | 9.50          | 6.86 | 4.56                |
|             | NVX            | 9.29          | 6.64 | 4.04                |
|             | NVXHalf        | 9.05          | 6.41 | 3.56                |
|             | BNT            | 10.21         | 7.49 | 4.43                |
|             | BNTHalf        | 10.05         | 7.20 | 4.36                |
PrimeBNT/BNT
|     | VLA     | 8.34  | 5.67 | 3.51 |
| --- | ------- | ----- | ---- | ---- |
|     | VLAHalf | 8.22  | 5.46 | 3.64 |
|     | Ad26    | 9.75  | 7.27 | 4.71 |
|     | m1273   | 10.43 | 7.61 | 4.72 |
|     | CVn     | 8.94  | 6.19 | 3.84 |
|     | ChAd    | 7.81  | 5.26 | 3.97 |
|     | NVX     | 8.85  | 6.59 | 4.73 |
|     | NVXHalf | 8.44  | 6.15 | 4.59 |
|     | BNT     | 9.93  | 7.39 | 4.75 |
|     | BNTHalf | 8.71  | 7.20 | 4.91 |
PrimeChAd/ChAd
|     | VLA     | 7.51  | 5.31 | 3.96 |
| --- | ------- | ----- | ---- | ---- |
|     | VLAHalf | 7.27  | 4.99 | 4.02 |
|     | Ad26    | 8.62  | 6.33 | 4.66 |
|     | m1273   | 10.35 | 7.77 | 5.00 |
|     | CVn     | 8.29  | 5.92 | 3.87 |
Table5: Pooledvarianceofeachgroup.
Immuneresponse
|     | Anti-spikeIgG |     | NT cellularresponse |     |
| --- | ------------- | --- | ------------------- | --- |
50
| Pooledsamplevariance |     | 0.70 | 0.83 | 1.54 |
| -------------------- | --- | ---- | ---- | ---- |
H.3 Additionalexperiments
| H.3.1 Additionalexperimentsforε | -APE-k |     |     |     |
| ------------------------------- | ------ | --- | --- | --- |
1
Inthissectionweshowthatforsomeinstancesouralgorithmcanrequireupto3timeslesssamples
comparedtoPSI-Unif-Elim. ThisisduetothestrategyofPSI-Unif-Elimwhichcontinuesampling
armsidentifiedasoptimaluntilthereareshownnottodominateanyarmintheactiveset.Forexample
onFigure3a,theoptimalarm2is"easy"toidentifyassuch. However,sinceitslightlydominatesthe
sub-optimalarm1,PSI-Unif-Elimshouldcontinuesamplingarm2untilarm1isremovedfromthe
activeset(likelythiswillhappenwhenthealgorithm"sees"thatarm1isdominatedbyarm3). We
wouldexpectouradaptivesamplingruletoavoidthisbehaviour.
Figure3bshowsthatAPEtakesnearlyhalftheaveragesamplecomplexityofPSI-Unif-Elimonthis
instance. Inparticular,Table6showstheaveragenumberofpullstakenbyPSI-Unif-Elimdivided
bytheaveragenumberofpullstakenby0-APE-K foreacharm. Wecanobservethatthemajor
differenceinsamplecomplexityisduetoarm2beingpullednearly6timesmorebyPSI-Unif-Elim
thanAPE.
36

1e4
0 3.5
0.9 1 2 3.0
3
0.8 4
2.5
0.7
2.0
0.6
1.5
0.5
1.0
0.4
0.5
0.3
0.4 0.5 0.6 0.7 0.8 APE[k=5] PSI-Unif-Elim
(a)Instance (b)Samplecomplexity
Figure3: Forthisinstance(left)S⋆ ={0,2,3}andontherightistheaveragesamplecomplexity
overthetrialsonthesameinstance.
Arm 0 1 2 3 4
Averageratioofpulls 3.08 1.36 5.68 1.28 1.40
Table6: AveragenumberofpullstakenbyPSI-Unif-Elimdividedbytheaveragenumberofpulls
takenby0-APE-K foreacharm.
ByincreasingthenumberofarmsandthedimensionwecangenerateinstancessimilartoFigure3a
wherethegapbetweenouralgorithmandPSI-Unif-Elimisevenlarger. Wechoseaspecificinstance
whereK =12,D =10andthereare11optimalarms. Onthisinstance(Figure4),wecanseethat
ouralgorithmuses3timeslesssamplesthanPSI-Unif-Elim.
1e5
1.75
1.50
1.25
1.00
0.75
0.50
0.25
0.00
APE[k=12] PSI-Unif-Elim
Figure4: AveragesamplecomplexityonaspecificinstancewithK =12,D =10and|S⋆|=11
Finally,combiningtheseadditionalexperimentswiththeresultsofTable2weobservethatonaverage
ε -APE-kperformsnearly20%betterthanAPEbuttherearesomeinstanceswherethegapcanbe
1
evenlarger. Ofcourse,thisalsomeansthatthereshouldexistinstancesinwhichtheimprovementis
smallerthan20%tocompensateforinstanceslikeFigure3a.ButwenotethatinstanceslikeFigure3a
areveryunlikelytobegeneratedrandomlysotheyshouldonlybeafewinthe2000instancesused
forFigure3a. Sothat20%isfairlyrepresentativeoftheaverageimprovementon“normalinstances"
(excludinginstanceslikeFigure3awheretheimprovementcanbewaylarger).
37

0.50 0
1
0.45 2
0.40
0.35
0.30
0.25
0.20
0.15
0.10
| 0.30 0.35 | 0.40 0.45   | 0.50 0.55 0.60 0.65 |                     |     |     |
| --------- | ----------- | ------------------- | ------------------- | --- | --- |
|           | (a)Instance |                     | (b)Samplecomplexity |     |     |
Figure5: Forthisinstance(left)S⋆ = {0,1,2}. Thedifferenceinonxandyaxisis0.1between
arm0and1and0.05betweenarm1and2. Therightmostfigureisthesamplecomplexityaveraged
over2000trials.
H.3.2 (ε ,ε )-APE
1 2
We investigate the empirical behavior of (ε ,ε )-APE for identifying an (ε ,ε )-cover. We set
|     |     | 1 2 |     | 1   | 2   |
| --- | --- | --- | --- | --- | --- |
ε 1 = 0,δ = 0.01 and we test different values of ε 2 ∈ {0,0.05,0.1,0.2,0.25}. We average the
resultsover2000independenttrialswithdifferentseedsonthesameinstance(Figure5a). Weuse
multi-variateBernoulliwithindependentmarginals. TheinstanceofFigure5aisatoyexamplewhere
(ε ,ε )-coveringcanbemeaningfulandreducethesamplecomplexity. The3Paretooptimalvectors
1 2
arechosenbyhandandthelast2vectorsarerandomlyuniformlygenerated.
WecanobserveonFigure5bthatthesamplecomplexitydecreasesasε
|     |     |     |     | 2 increases. | Thisisfurther |
| --- | --- | --- | --- | ------------ | ------------- |
confirmed in Figure 6which shows theempirical sample complexity and the averagesize of the
recommendedcoverversusε for50equally-spacedvaluesofε between0and1/2. Thedrops
|     |     | 2   | 2   |     |     |
| --- | --- | --- | --- | --- | --- |
observedinFigure6bcorrespondtothevaluesofε forwhichAPEremovesanoptimalarmfromthe
2
covertosavesomesamples(Figure6a).Amajordecreaseinthesamplecomplexitycorrespondsmore
orlesstoanarmbeingremovedfromtherecommendedset. WeobserveinFigure7thehistogramof
occurrenceofeacharmintherecommendedsetfor3valuesofε 2 correspondingmoreorlesstothe
middleofeachplateau. Wecanseethatforε =0.15,arm0isalwaysrecommended,buttheothers
2
arerecommendedonhalfoftheruns. Forε =0.4,thealgorithmnearlyalwaysrecommendarm0,
2
| whichasthelargestω | term(i.etheeasiesttoidentifyasoptimal). |     |     |     |     |
| ------------------ | --------------------------------------- | --- | --- | --- | --- |
i
18000
3.00
16000
2.75
| 2.50 |     |     | 14000 |     |     |
| ---- | --- | --- | ----- | --- | --- |
| 2.25 |     |     | 12000 |     |     |
|)2,1
| 2.00 |     | 2,1 |       |     |     |
| ---- | --- | --- | ----- | --- | --- |
| (    |     |     | 10000 |     |     |
|
1.75
8000
1.50
6000
1.25
| 1.00           |     |             | 4000                   |         |         |
| -------------- | --- | ----------- | ---------------------- | ------- | ------- |
| 0.0 0.1        | 0.2 | 0.3 0.4 0.5 | 0.0 0.1                | 0.2 0.3 | 0.4 0.5 |
|                | 2   |             |                        | 2       |         |
| (a)Average|O(τ |     | )|vsε       | (b)Samplecomplexityvsε |         |         |
|                |     | ε1,ε2 2     |                        |         | 2       |
Figure6: OntheleftistheaveragelengthofO(τ )(overthedifferentruns)versusε andonthe
|     |     | ε1,ε2 |     |     | 2   |
| --- | --- | ----- | --- | --- | --- |
rightistheempiricalsamplecomplexity(averagedovertheruns)versusε 2 . Theempiricalprobability
oferrorwasequaltozero.
38

The plateau in the sample complexity for large values of ε (> 0.3) is explained by the fact the
2
algorithmneedstoidentifyatleastoneoptimalarm(whichisreflectedinthesizeofthereturnedset
Figure6a). Indeed,forε =0fixed,analgorithmfor(ε ,ε )-coveringstillneedtoassertthatthe
|     |     | 1   |     |     | 1 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
armsintherecommendedsetaretrulyoptimalwhichwillrequiresomesamplesevenwhenε isvery
2
large. Thus,forε 2 >0.3thealgorithmneedtoidentifyatleastoneoptimalarmandwecanseeon
Figure6athatforthesevaluesofε 2 ,therecommendedsetcontainsonlyoneoptimalarm. Actually
wecanobserveempiricallythatthe"limit"samplecomplexityobservedinFigure6bisclosetothe
averagesamplecomplexityof0-APE-1onthesameinstance(4073samples).
| (a)Averageε | =0.05 |     |     | (b)ε =0.15 |     |     | (c)ε =0.4 |
| ----------- | ----- | --- | --- | ---------- | --- | --- | --------- |
|             | 2     |     |     | 2          |     |     | 2         |
Figure7: Histogramofthenumberoftimeseacharmisintherecommendedsetoverthe2000runs
| for3valuesofε | .   |     |     |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | --- |
2
| H.3.3 Comparingε | -APE-ktoanadaptationofPSI-Unif-Elim |     |     |     |     |     |     |
| ---------------- | ----------------------------------- | --- | --- | --- | --- | --- | --- |
1
Inthissection,wecompareε -APE-k toanadaptationofPSI-Unif-Elimwhichstopsearlierifat
1
leastkoptimalarmshavebeenidentified. Thepseudo-codeofthealgorithmisgiveninalgorithm3.
Asshownin[4],armsinP (t)arealreadyidentifiedasoptimalbutwhenthegoalistoidentifythe
1
Pareto set, some of them (namely P 1 (t)\P 2 (t)) need to be sampled again until all the arms they
potentiallydominateareremovedfromA(t)andonlythenthosearmswillbelongtoP 2 (t)andwill
beremovedfromtheactiveset. However,forthek-relaxation,identifyingkoptimalarmsisenough
sothealgorithmcanstopassoonas|P(t−1)∪P (t)|≥k. Itthisneveroccurs,thenthealgorithm
1
willfollowtheinitialstoppingcondition,thatistostopwhenA(t)=∅.
| Algorithm3:AdaptationofPSI-Unif-Elimfortheε |     |     |     |     | -APE-kobjective |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --------------- | --- | --- |
1
| Data: parameterε | ≥0,k | ≤K  |     |     |     |     |     |
| ---------------- | ---- | --- | --- | --- | --- | --- | --- |
1
| initialize:A(0)=A,t←1,P(0)=P |     |     | (0)=P | (0)=∅ |     |     |     |
| ---------------------------- | --- | --- | ----- | ----- | --- | --- | --- |
1 2
| fort=1,2,... | do  |     |     |     |     |     |     |
| ------------ | --- | --- | --- | --- | --- | --- | --- |
sampleeacharminA(t−1)once;
| A(t)←{i∈A(t−1):∀j |     | ∈A(t−1),m(i,j,t)≤β |     |     | (t)}; |     |     |
| ----------------- | --- | ------------------ | --- | --- | ----- | --- | --- |
i,j
| P (t)←{i∈A(t):∀j |                     | ∈A(t)\{i},M(i,j,t)+ε |        |            | ≥β    | (t)}; |     |
| ---------------- | ------------------- | -------------------- | ------ | ---------- | ----- | ----- | --- |
| 1                |                     |                      |        |            | 1 i,j |       |     |
| P (t)←{j         | ∈P (t):∄i∈A(t)\P    |                      | (t)s.t | M(i,j,t)+ε | ≤β    | (t)}; |     |
| 2                | 1                   |                      | 1      |            | 1     | i,j   |     |
| A(t)←A(t)\P      | (t)andP(t)←P(t−1)∪P |                      |        | (t);       |       |       |     |
|                  | 2                   |                      |        | 2          |       |       |     |
ifA(t)=∅then
breakandoutputP(t);
| if|P(t−1)∪P            | 1 (t)|≥kthen |     |      |     |     |     |     |
| ---------------------- | ------------ | --- | ---- | --- | --- | --- | --- |
| breakandoutputP(t−1)∪P |              |     | (t); |     |     |     |     |
1
Wesetδ =0.1andwecomparebothalgorithmson2typeofrandomlyuniformlygeneratedBernoulli
instances. ForthefirsttypewesetK =10,D =2andforthesecondone,wesetK =50,D =2.
FortheinstanceswithK =10wesetε =0.05andrunbothalgorithmson2000randomBernoulli
1
|                | ForthesecondtypeofinstancesK |     |     | =   | 50,wesetε | = 0.1andwebenchmarkthe |     |
| -------------- | ---------------------------- | --- | --- | --- | --------- | ---------------------- | --- |
| instanceswith. |                              |     |     |     |           | 1                      |     |
algorithmson500randominstances.TheaveragesizeoftheParetosetwas2.90(forK =10,D =2)
and4.51(K = 50,D = 2). Figure8showstheaveragesamplecomplexityofthealgorithmsfor
differentvaluesofk ∈{1,...,5}. Wecanobservethatthedifferencebetweenε -APE-kandAPE
1
for the 5 values reported is more important for K = 50 than for K = 10. Put together, these
experimentsshowthatouralgorithmisstillpreferableforthek-relaxation.
39

35000
30000 1-APE-k
Heuristic-PSI-Unif-Elim
30000
25000
25000
20000
20000
15000
15000
10000
10000
| 5000           |         |     | 5000 |                |         |     |
| -------------- | ------- | --- | ---- | -------------- | ------- | --- |
| 0              |         |     |      | 0              |         |     |
| k=1 k=2        | k=3 k=4 | k=5 |      | k=1 k=2        | k=3 k=4 | k=5 |
| (a)K =10,D=2,ε | 1 =0.05 |     |      | (b)K =50,D=2,ε | 1 =0.1  |     |
Figure8: Averagesamplecomplexityon2000randominstances(left)and500randominstances
(right).
H.3.4 ComparisontosomeBAIalgorithms
We evaluate the performance of APE for Best Arm Identification (D = 1) on two
randomly generated instances: one with K = 5 and means (rounded) X :=
1
(0.25,0.16,0.87,0.22,0.98), and the second one with K = 10 and means (rounded) X :=
2
(0.43,0.33,0.56,0.85,0.20,0.93,0.70,0.82,0.56,0.78)WeusetheinstantiationofAPEwithconfi-
dencebonusesonpairofarmsbutwithoutthepossibleimprovementinvokedinRemark2. UGap
andLUCBareimplementedwiththetightestknown(toourknowledge)confidencebonustaken
from[17](inspiritofthefinite-timelawoftheiteratedlogarithm[11]). LUCB++isusedwiththe
improvedschemegivenin[26]. Wesetδ =0.01buttheempiricalerrorwaswaysmaller. Theresults
areaveragedover1000independenttrials.
Onthey-axisisthesamplecomplexityinunitsof(anapproximationof)thelowerboundofBAIfor
| Gaussianrewardswithσ | =1/2 4([18,26]): |            |          |     |     |     |
| -------------------- | ---------------- | ---------- | -------- | --- | --- | --- |
|                      |                  | (cid:18) 1 | (cid:19) | K 1 |     |     |
(cid:88)
|     | Hlog |      | withH | = . |     |     |
| --- | ---- | ---- | ----- | --- | --- | --- |
|     |      | 2.4δ |       | 2∆2 |     |     |
i=1 i
| 1e1 |     |     |     | 1e1 |     |     |
| --- | --- | --- | --- | --- | --- | --- |
2.00
| 2.0 |     |     | 1.75 |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- |
1.50
1.5
1.25
1.00
1.0
0.75
0.50
0.5
0.25
|     |     | lbd |     |     |     | lbd |
| --- | --- | --- | --- | --- | --- | --- |
0.0
| LUCB++ | LUCB APE UGap |     |     | LUCB++ | LUCB APE UGap |     |
| ------ | ------------- | --- | --- | ------ | ------------- | --- |
| (a)H   | =94.84        |     |     | (b)H   | =223.20       |     |
Figure9: EmpiricalsamplecomplexityexpressedinunitsofthelowerboundHlog(1/2.4δ)(blue
line)onarandominstancewithK = 5(left)andK = 10(right),δ = 0.01. Thebluelinehasa
coordinateof1onthey-axis(lower-bound).
4asBernoullidistributionsare1/2-subgaussian
40