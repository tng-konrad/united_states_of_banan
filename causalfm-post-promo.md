# LinkedIn promo

Does zero-shot causal inference work well enough to guide a marketing budget?

We tested CausalFM and CausalPFN against six classical estimators, from interaction regression and S/T/X learners to a DR-learner and causal forest.

Three findings stood out:

• CausalPFN achieved CATE RMSE of 0.286 versus 0.440 for the best classical baseline on three IHDP outcome replications. Encouraging, with a small benchmark behind it.

• In a real randomized email dataset, a validation-selected CausalPFN policy had estimated incremental test profit of $42.59. Its 95% interval ran from −$261.45 to $346.63. Positive deployment value remained unresolved.

• In a separate simulation with negative spillovers, both foundation models' individual-effect predictions implied positive rollout value. The true total effect was negative.

“Zero-shot” needs a definition here: the foundation weights stay frozen, but the models still receive observed treatments and outcomes as context.

The new post walks through causal inference, uplift modeling, doubly robust estimation, and budget allocation—with the notebook, comparative results, and the assumptions behind each decision.

Where would you spend the next unit of effort: a better effect estimator, richer covariates, or a better experiment?

Read the post: [ADD SUBSTACK POST URL]

#CausalInference #FoundationModels #UpliftModeling

---

# SEO description

Can zero-shot causal inference guide real decisions? CausalFM and CausalPFN meet classical estimators, marketing budgets, and hidden confounding.
