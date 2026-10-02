# Time Series with Konrad: a foundation model for causality?

*zero-shot causal inference, classical baselines, and a marketing budget*

KONRAD BANACHEWICZ  
OCT 02, 2026

<!-- EDITORIAL NOTES — remove before publication.
This is a causal-inference special; no unverified episode number has been assigned.
Replace the companion-notebook placeholder with its public URL. The source notebook is causalfm/causal_foundation_models.ipynb.
Each GRAPH/TABLE PLACEHOLDER below names an existing, title-free PNG in causalfm/artifacts/demo/notebook_png/. Keep the supplied captions outside the image.
Numerical results were checked against the current notebook and CSV artifacts, rather than the older substack_post.md or RESULTS.md summaries. This article describes the executed demo, not the unrun reference configuration.
END EDITORIAL NOTES -->

Causal inference has been helping us evaluate interventions for decades. We have randomized experiments, regression adjustment, propensity scores, and an impressive collection of methods whose names contain “robust.” Then the generative AI era arrived, and apparently every statistical problem needed a foundation model. Cause and effect was unlikely to escape. ;-)

The proposition is attractive: load a pretrained model, give it observations from a new problem, and estimate who would benefit from treatment. The foundation weights stay frozen, while observed treatments and outcomes supply context for the new problem. We could spend less time building another estimation pipeline and more time deciding what to do. But causal inference has a particular complication: a convincing prediction does not automatically tell us what will happen when we intervene.

**A foundation model can improve how we estimate an effect; the data and assumptions still determine which effect we can identify.** We will work through regression, S/T/X learners, weighting and doubly robust estimation, then compare them with CausalFM and CausalPFN. The goal is practical understanding: can these models recover effects, help allocate a marketing budget, and survive the assumptions that make the exercise meaningful?

📓 All code lives in the companion notebook:  
**[COMPANION NOTEBOOK URL — add the public link before publishing]**  
Local source: [causal_foundation_models.ipynb](/Users/kbanachewicz/Documents/time-series-with-Konrad/causalfm/causal_foundation_models.ipynb).

## Groundwork: who buys, and who changes their mind?

Suppose we want to send a marketing email. A response model predicts who will spend money after receiving it. That is useful, but some enthusiastic customers would have bought anyway. Paying to contact them can produce an excellent response rate and very little incremental value.

We need the **conditional average treatment effect**, or **CATE**:

$$
\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x].
$$

Here, $Y(1)$ is spending if we send the email, $Y(0)$ is spending if we do not, and $X$ contains information available before assignment. In plain English: how much additional spending does the email cause, on average, among customers with these characteristics? Averaging this function across the target population gives the **average treatment effect**, or **ATE**.

We observe only one outcome for each customer. Identification therefore needs a bridge from observed comparisons to that missing counterfactual. Randomization supplies one. With observational data, the usual adjustment approach assumes that our recorded covariates contain the common causes of assignment and outcome. We also need **overlap**: comparable customers must have some chance of receiving either assignment.

> **GRAPH PLACEHOLDER 01 — insert `004_figure_identification.png` here.**
>
> *Caption: Customer characteristics can influence both email assignment and spending. Randomized assignment removes that selection mechanism; observational adjustment must account for it.*

Read the arrows before choosing the estimator. If previous purchasing affects both targeting and future spending, the raw difference between recipients and nonrecipients mixes selection with the email effect. A more flexible model can describe that mixture very accurately.

## Start with regression, then let the effect bend

Our first rung is **interaction OLS**. Ordinary regression becomes a heterogeneous-effect model when treatment can interact with customer characteristics:

$$
\mathbb{E}[Y\mid X=x,T=t]=\beta_0+x^\top\beta+t(\theta_0+x^\top\theta).
$$

The first part predicts baseline spending. The expression multiplied by treatment describes how the expected email effect varies across customers. Setting $t$ to one and zero, then subtracting, leaves $\theta_0+x^\top\theta$.

This gives us an interpretable baseline with few moving parts. Its restriction is the linear shape. If treatment helps only above a spending threshold, or interacts with several customer attributes, we may want something more flexible.

The **S-learner** fits one outcome model using both $X$ and treatment as predictors. The **T-learner** fits separate outcome models in the treated and control groups. Both ultimately calculate:

$$
\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x).
$$

We predict the outcome under each assignment and take the difference. One shared model can borrow information across groups, although it can also underuse treatment when baseline outcome variation dominates. Two models allow different response surfaces, but each has fewer observations to learn from. Our notebook uses gradient boosting for these outcome models.

The **X-learner** takes another step. It uses the two outcome models to construct estimated effects for observed customers, fits effect models to those estimates, and combines their predictions using the treatment propensity. It targets the difference more directly, while inheriting uncertainty from the first-stage outcome models.

Deeper trees and smaller leaves allow more local variation; shallower trees and larger leaves smooth more aggressively. Neither extreme is universally correct. We fix the configurations before evaluation, so this exercise compares declared baselines rather than a contest to tune against known test effects.

## Correcting for who received treatment

Flexible outcome models still have to contend with treatment selection. The **propensity score**, $e(x)=P(T=1\mid X=x)$, describes how likely a customer is to receive treatment given their observed characteristics.

**Inverse probability weighting**, or **IPW**, upweights customers whose observed assignment was relatively unlikely. Its population-average estimator is:

$$
\widehat{ATE}_{IPW}=\frac{1}{n}\sum_i\left[\frac{T_iY_i}{\hat e(X_i)}-\frac{(1-T_i)Y_i}{1-\hat e(X_i)}\right].
$$

Those fractions reconstruct treated and untreated comparisons for the target population. **Hájek weighting** normalizes the weights within each group. When propensities approach zero or one, a few observations can carry enormous influence. A large dataset can then behave like a much smaller one.

**Augmented IPW**, or **AIPW**, combines outcome predictions with a weighted correction for their errors:

$$
\Gamma_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)
+\frac{T_i[Y_i-\hat\mu_1(X_i)]}{\hat e_i}
-\frac{(1-T_i)[Y_i-\hat\mu_0(X_i)]}{1-\hat e_i}.
$$

Start with predicted uplift, then correct it using the observed outcome residual in the customer's actual treatment group. The mean score estimates ATE. Under the identification and overlap assumptions, consistency can survive a wrong outcome specification if the propensity specification is correct, or vice versa. That is the **double robustness** in the name.

We use **cross-fitting**: each customer's nuisance predictions come from models trained on other observations. A **DR-learner** goes further and regresses these corrected scores on $X$ to learn CATE. Our final classical competitor, **CausalForestDML**, uses cross-fitted nuisance adjustment and a forest to estimate heterogeneous effects.

A controlled experiment makes the distinction visible. The population ATE is exactly one. Across 30 simulated cohorts, AIPW produces:

```text
{'both specifications correct': 1.000,
 'only outcome correct': 1.002,
 'only propensity correct': 0.969,
 'both specifications wrong': 1.737}
```

> **GRAPH PLACEHOLDER 02 — insert `021_figure_double_robustness_aipw.png` here.**
>
> *Caption: AIPW bias across four combinations of outcome and propensity specifications. The true population ATE is 1; results average 30 simulated cohorts.*

One correct specification kept the estimate close to truth here. Two wrong specifications produced substantial bias. This is a property about estimation under assumptions, not permission to ignore missing confounders. Nor does it guarantee that learning a detailed CATE function from a small sample will work well.

## What does “zero-shot” mean for causality?

We have assembled several ways to learn from one dataset. Foundation models move some of that learning effort into pretraining across many synthetic causal problems.

**CausalFM** and **CausalPFN** use prior-data fitted networks: pretrained transformers that receive a new dataset as context and infer effects. CausalFM develops priors for several causal identification settings; this notebook uses its standard CATE checkpoint. CausalPFN is pretrained on simulated processes satisfying ignorability, the assumption that observed covariates suffice for treatment adjustment. See the [CausalFM paper](https://arxiv.org/abs/2506.10914) and [CausalPFN paper](https://arxiv.org/abs/2506.07918).

Think of the synthetic training distribution as a collection of worked examples of plausible causal worlds. The model learns which patterns in a small dataset tend to correspond to which effects. That prior can be valuable when a new dataset is too small to support a complicated estimator from scratch. It can also be a poor match for the new problem.

**Here, “zero-shot” means no fine-tuning of the foundation weights.** We still supply labeled context rows containing covariates, treatment assignments, and factual outcomes. CausalPFN's implementation also fits a local model used for retrieval. This is in-context estimation with frozen foundation weights; “no target data” and “no local fitting anywhere” would both misdescribe the notebook.

We load the real released checkpoints, with fixed revisions and verified hashes. Each primary comparison gives the models identical labeled context rows. Pretraining cost has already been paid, and measured local runtime includes loading and context preparation. We should judge the complete workflow rather than assume that a frozen model must be faster.

## Can the models recover effects we know?

To answer that, we need a setting where the effect is known. Our simulations provide independent training and query cohorts. **IHDP** provides a complementary test: real covariates with simulated outcomes and known conditional mean effects, using the supplied training and test partitions.

The error metric is **CATE RMSE**, sometimes reported as square-root PEHE:

$$
\mathrm{RMSE}_{CATE}=\sqrt{\frac{1}{n}\sum_i[\hat\tau(X_i)-\tau(X_i)]^2}.
$$

It measures distance from the true conditional mean effect at the evaluation covariates. Smaller is better. We never pass that truth into fitting, scaling, or model selection.

On three IHDP outcome replications, the comparison is:

| Method | Mean CATE RMSE |
|:--|--:|
| CausalPFN | 0.286 |
| T-learner | 0.440 |
| X-learner | 0.554 |
| S-learner | 0.627 |
| Interaction OLS | 0.715 |
| Causal forest | 1.544 |
| CausalFM | 2.643 |
| DR-learner | 3.891 |

> **GRAPH PLACEHOLDER 03 — insert `013_figure_ihdp_cate_rmse.png` here.**
>
> *Caption: Mean CATE RMSE across three IHDP outcome replications. The outcomes are simulated, and the replications reuse covariates.*

CausalPFN is the strongest method in this small comparison. CausalFM is substantially worse than several classical alternatives. A shared label, “causal foundation model,” does not produce a shared result.

The DR-learner's weak score also illustrates the earlier caveat: robustness of an average-effect estimator does not guarantee accurate heterogeneous-effect learning in a small sample. These are fixed configurations, with only three replications. We have evidence for including CausalPFN in the comparison, rather than a universal ordering of algorithms.

The simulations supply a useful reality check. With 512 context rows in the nonlinear scenario:

```text
{'Interaction OLS': 0.511, 'CausalPFN': 0.530, 'CausalFM': 0.544}
```

> **GRAPH PLACEHOLDER 04 — insert `011_figure_sample_efficiency_nonlinear.png` here.**
>
> *Caption: Mean CATE RMSE versus context size in the nonlinear simulation, averaged over two seeds.*

OLS slightly leads both foundation models at 512 rows. CausalFM is competitive here despite its weaker IHDP result. With two seeds, tiny differences deserve restraint, but the practical lesson is clear: keep the inexpensive baseline. It has survived several generations of fashionable alternatives. ;-)

## From effect estimates to an email campaign

Recovering a simulated effect is encouraging. Spending money requires a different test.

We use [Kevin Hillstrom's randomized email experiment](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html): 64,000 customers across three assignment groups, with spending measured over two weeks. We predeclare men's email versus no email as the binary comparison. Conditional on those two arms, assignment probability is 0.5.

Our first data check separates predictors from consequences. Customer history and other baseline attributes can enter $X$. Visits, conversion, and spending measured after assignment cannot be baseline predictors. Historical spending and the outcome called “spend” may sound similar; mixing them would make an impressively unhelpful model.

These are customer-level experimental records, so the notebook uses stratified random splits. There is no forecasting horizon that requires a chronological cutoff here. A future campaign would introduce a separate transport question about changes in customers, creatives, and conditions.

We reserve validation and test data first. The demo then uses 512 labeled context rows, 1,500 validation customers, and 2,500 randomized test customers. Shared encoding and scaling use baseline covariates from the full training pool; calling this strictly “only 512 raw rows of information” would overstate the restriction. Full-data OLS and T-learner fits appear separately as a practical comparison.

We also create an observational training challenge. Retention probabilities depend on baseline covariates and recorded assignment, producing a known propensity while preserving the baseline population distribution in expectation. We retain factual records without changing their treatments or outcomes. The validation and test assignments remain randomized.

> **GRAPH PLACEHOLDER 05 — insert `017_figure_propensity_overlap.png` here.**
>
> *Caption: Treatment propensities in the deliberately selected observational training sample. The independent evaluation samples remain randomized.*

The distributions show the selection problem we have introduced. They also show why overlap belongs in the workflow: if comparable treated and untreated customers become rare, neither weighting nor a pretrained model acquires the missing comparisons for free.

## The budget makes the question concrete

Our economic scenario assumes a **40% contribution margin** and **$0.10 per assigned email**. Those are assumptions, not observed Hillstrom fields. A customer's estimated net contribution is:

$$
0.40\hat\tau(x)-0.10.
$$

We rank customers by this contribution and select positive values up to the contact cap. The primary cap is 20%: at most 500 contacts among 2,500 test customers, costing at most $50. A cap does not require us to spend everything.

Validation selects the model at that predeclared budget. It can choose “treat nobody” if no candidate has positive estimated validation profit. We then freeze the choice before looking at test outcomes.

All policies receive the same cross-fitted AIPW evaluation, using the known randomization probability. Candidate models choose whom to contact; separate evaluator models supply outcome adjustments. IPW is an additional check. We cannot compute true individual-effect errors on Hillstrom because both counterfactual outcomes are never observed.

> **GRAPH PLACEHOLDER 06 — insert `035_figure_profit_by_budget_randomized.png` here.**
>
> *Caption: Estimated incremental test profit per eligible customer across budget fractions, after randomized training. Curves show point estimates; model selection used validation at the fixed 20% cap.*

The curves explore how different policies behave as the budget changes. They are not a second opportunity to choose a winner. Selecting a new model or cap from these test curves would turn the holdout into another training tool.

> **GRAPH PLACEHOLDER 07 — insert `036_figure_profit_by_budget_observed_confounding.png` here.**
>
> *Caption: The corresponding test-profit curves after deliberately confounded training. Evaluation still uses the independent randomized holdout.*

The constructed observational sample changes the learned rankings, but the deployment conclusion still depends on the locked policy and its uncertainty. Here are those two choices:

| Training regime | Validation-selected model | Contacts | Estimated total incremental profit | 95% interval |
|:--|:--|--:|--:|:--|
| Randomized | Interaction OLS | 500 | −$194.25 | [−$444.85, $56.36] |
| Observed confounding | CausalPFN | 418 | $42.59 | [−$261.45, $346.63] |

**Neither selected policy establishes positive incremental profit relative to treating nobody.** The positive CausalPFN point estimate leaves considerable room for a loss. The different training samples also mean we cannot interpret the comparison as evidence that introducing confounding improves performance.

These intervals describe evaluation uncertainty for the locked policies; they do not include every source of retraining and model-selection variability. The notebook also reports paired comparisons with random targeting, uplift calibration, and centered revenue-gain curves. Those dollar-gain curves are not interchangeable with every library's binary-conversion Qini convention.

This is where a good benchmark score becomes a research result rather than an automatic deployment decision.

## Better allocation still depends on the economics

Equal contact costs make allocation straightforward. With heterogeneous costs, maximizing total profit under a budget becomes a **0/1 knapsack problem**. Ranking by benefit-to-cost ratio is a heuristic.

The notebook's small illustrative example makes the difference concrete: three actions cost $10, $20, and $30 and have net contributions of $60, $100, and $120. Under a $50 cap, ratio ranking selects the first two for $160. Exact allocation selects the second and third for $220.

Better allocation cannot compensate for wrong effect estimates. Nor is the assumed contact cost sacred. The notebook revalues the same locked marketing policies as that cost changes:

> **GRAPH PLACEHOLDER 08 — insert `043_figure_contact_cost_sensitivity.png` here.**
>
> *Caption: Incremental profit under alternative contact costs, holding the selected customers fixed and contribution margin at 40%.*

Read this as economic sensitivity. Customer selections stay fixed, so increasing unit cost also changes total expenditure. It is not a sequence of newly optimized campaigns under one fixed dollar budget. Maximizing total profit and maximizing an ROI ratio are different objectives, too.

## The assumptions come back to collect

So far, our observational challenge used measured covariates. What happens when an important common cause is missing?

The notebook separately stresses overlap and hidden confounding. Weak overlap concentrates influence in a few observations; weight clipping trades variance against bias, while trimming would also change the target population. We inspect weight distributions and effective sample sizes rather than treating the raw row count as reassurance.

In the hidden-confounder simulation, the missing variable affects treatment and baseline outcomes. The true CATE still depends on observed $X$, so the effect-error target remains clear. All four inspected methods nevertheless show substantial upward ATE bias:

```text
{'CausalFM': 1.720, 'CausalPFN': 1.620,
 'DR-learner': 1.640, 'Interaction OLS': 1.962}
```

> **GRAPH PLACEHOLDER 09 — insert `053_figure_foundation_model_stress_ate_bias.png` here.**
>
> *Caption: Mean ATE bias under baseline conditions, weak overlap, and hidden confounding, with two fitted models per method and scenario.*

The hidden-confounding bars are large across the board. A prior learned from synthetic causal problems did not establish exchangeability in this new problem. Extra measurements, additional identifying assumptions, or a different experimental design would be needed. CausalFM's separate instrumental-variable and frontdoor settings are outside this test.

We also include a formal **Cinelli–Hazlett sensitivity analysis**. It asks how strongly an omitted variable would need to relate to treatment and outcome, after adjustment, to erase an additive OLS coefficient. Observed covariates provide benchmarks for those hypothetical strengths. The [sensemakr documentation](https://carloscinelli.com/sensemakr/reference/sensemakr.html) explains this parameterization.

> **GRAPH PLACEHOLDER 10 — insert `050_figure_formal_sensitivity.png` here.**
>
> *Caption: Sensitivity of an additive OLS treatment coefficient to omitted confounding, expressed through two partial R-squared quantities.*

The zero contour marks combinations that erase the point estimate. Benchmarks help judge the scale of those combinations, but they do not prove that omitted variables are weak. This calculation concerns the stated OLS coefficient; it does not supply sensitivity bounds for every foundation-model CATE or campaign profit estimate.

## A wide interval can be very accommodating

CausalPFN also produces native intervals by sampling conditional potential-outcome outputs and differencing them. In this small diagnostic, every queried mean CATE fell inside its interval. That sounds reassuring until we inspect width.

> **TABLE PLACEHOLDER 11 — insert `055_table_native_interval_summary.png` here.**
>
> *Caption: Native CausalPFN interval containment and width, with calibration disabled, 32 query points per fit, and two fits per scenario.*

Mean width is **6.839** at baseline and **9.274** under hidden confounding. Broad intervals can contain truth even while the point estimates are biased. Queries within a fitted model are dependent, and two fits cannot establish nominal coverage.

We should therefore read this as a descriptive containment check against mean CATE. The sampled quantity and the mean-effect target need careful distinction. Likewise, CausalFM's Gaussian-mixture quantiles should not simply be relabeled as confidence intervals for mean CATE.

## When one customer's treatment affects another customer

There is one more way to answer the wrong question accurately. Suppose a promotion helps a targeted seller by diverting demand from nearby sellers. Individual uplift can be positive while the marketplace loses overall.

This is **interference**, part of the territory covered by **SUTVA**, the stable unit treatment value assumption. The notebook simulates independent four-person communities. We randomize community treatment probabilities, then individual assignments. An outcome depends on own treatment and whether any other member receives treatment.

These are simulated communities. Hillstrom contains no measured network links, so we make no claim that its emails had these spillovers.

With negative spillovers, the true average effect of treating everyone rather than nobody is **−0.206**. Naively using ordinary individual CATE predictions as rollout forecasts gives:

```text
{'CausalFM': 0.361, 'CausalPFN': 0.499,
 'DR-learner': 0.516, 'Interaction OLS': 0.493}
```

> **GRAPH PLACEHOLDER 12 — insert `063_figure_foundation_models_under_interference.png` here.**
>
> *Caption: Individual-effect models used as deliberate, naive forecasts of total rollout value. Under negative spillovers, every model predicts a positive increment while the simulated total effect is negative.*

This is a deliberate mismatch of targets. The models receive baseline features and own treatment, without the exposure representation required for the rollout question. The result illustrates a modeling mistake; it is not a fair contest to identify which architecture best handles networks. Cluster-aware standard errors alone would not repair that mistake either.

The exposure-aware analysis instead distinguishes direct, spillover, and total effects. Known randomization probabilities support **Horvitz–Thompson** and **Hájek** estimators, with uncertainty computed across independent communities. This follows the experimental-design, exposure-mapping, and estimand framework developed by [Aronow and Samii](https://arxiv.org/abs/1305.6156).

> **GRAPH PLACEHOLDER 13 — insert `061_figure_interference_policy_value_spillover_minus_1.png` here.**
>
> *Caption: Policy value under negative spillovers in the separate exposure-aware simulation. The horizontal axis is target treatment saturation.*

Accounting for exposure tracks the simulated policy-value curve, while the naive additive forecast misses its consequences. That works here because the exposure mapping is sufficient by construction and the design supports all relevant states. The saturation budget is an expected contact budget, unlike the earlier exact individual allocation cap.

Well-defined treatment versions matter, too. Men's and women's emails are different actions. Pooling them can define a specified mixture, but changing the mixture or creative changes the intervention whose effect we are trying to transport.

## Closing time

We climbed from regression interactions through flexible outcome learners, assignment weighting, doubly robust estimation, and frozen causal foundation models. Then we asked whether their predictions supported profitable decisions and whether those decisions survived missing confounders and spillovers. Each step added something useful; none made the earlier identification questions disappear.

**Trying a causal foundation model makes sense when we treat it as an estimator inside a causal workflow.** CausalPFN earned its place in this comparison through its IHDP result. OLS earned its place by remaining competitive in the simulations. CausalFM's varying performance across settings made the case for testing the particular checkpoint on the particular problem.

The next experiment should randomize policy assignment: the selected targeting policy versus an incumbent under the same budget. Predeclare the intervention, outcome horizon, costs, minimum worthwhile gain, and guardrails such as unsubscribes. Where peer effects are plausible, randomize communities or supported saturation strategies and evaluate community value.

Zero-shot inference can reduce the work of building an estimator. We still have to decide what intervention means, which comparisons identify its effect, and what evidence would justify spending the budget. Those parts of causal inference have kept their jobs.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
