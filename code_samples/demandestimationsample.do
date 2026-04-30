* ------------------------------------------------------------------
* Purpose:
* Estimate a flexible demand specification with price interactions
* and high-dimensional fixed effects, retaining only variables with
* meaningful within-store-week variation. Then simulate counterfactual
* pricing to evaluate impacts on demand, revenue, and consumer welfare.
*
* Outline:
* - Construct price interaction terms with key indicators
* - Filter interactions and covariates by residual (within-FE) variation
* - Estimate demand using reghdfe with selected variables
* - Simulate counterfactual price changes
* - Recover implied shares, quantities, and revenues
* - Compute welfare via log-sum formula
* - Visualize distributions of counterfactual outcomes
* ------------------------------------------------------------------


* These are store/product characteristics we think might shift how
* price-sensitive customers are — e.g., do shoppers near a competitor
* react more to price? Do promo items behave differently?
local dummies compnear compby compcub urban zone dtcomp farthest promo
local keep_interactions

foreach var of local dummies {

    * Interact price with each characteristic — this lets the price
    * coefficient shift depending on, say, whether there's a nearby
    * Costco or whether the product is on promotion
    gen price_wmean_`var' = price_wmean * `var'

    * Strip out store-week fixed effects to see how much variation
    * is actually left — that residual variation is all that identifies us
    reghdfe price_wmean_`var', absorb(store_week) resid
    predict resid_`var', resid
    summarize resid_`var'

    * If the residual sd is tiny, the interaction is basically absorbed
    * by the fixed effects and won't help us — drop it to keep the
    * model clean and avoid spurious estimates
    if r(sd) > 0.05 {
        local keep_interactions "`keep_interactions' price_wmean_`var'"
    }
    else {
        drop price_wmean_`var'
        display "Dropped price_wmean*`var' due to low within-store-week variation (sd=`r(sd)')"
    }

    drop resid_`var'
}

display "Keeping interactions: `keep_interactions'"


* Same filter for the continuous covariates (income, age, education, etc.)
* If a variable barely moves within a store-week, it can't tell us much
* beyond what the fixed effect already captures
quietly {
    local keep_covars
    foreach var of local allcovars {

        reghdfe `var', absorb(store_week) resid
        predict resid_`var', resid
        summarize resid_`var'

        if r(sd) > 0.05 {
            local keep_covars "`keep_covars' `var'"
        }

        drop resid_`var'
    }
}

display "Keeping covariates: `keep_covars'"


* Estimate the Berry logit demand model on whatever survived the
* variation filter above. The ##c() syntax interacts price continuously
* with each covariate — so e.g. price sensitivity can flex with local
* income or age demographics. Store-week FEs absorb demand shocks common
* to a market; UPC FEs handle persistent differences in product appeal.
* Standard errors clustered at the store level.
eststo large: reghdfe y ///
    price_wmean ///
    c.price_wmean##c.(`keep_covars') ///
    `keep_interactions' ///
    , ///
    absorb(store_week UPC) ///
    vce(cluster store)

* delta_hat is our predicted mean utility including fixed effects —
* we need this to compute counterfactual shares later
predict double delta_hat, xb
predict yhat_large, xb


****************************************************
* Counterfactual: Uniform +10% price increase
****************************************************

* Scenario 1: every product gets a flat 10% price increase
gen double price_cf_uniform = price_wmean * 1.10

* Shift predicted utility by the price change, scaled by each product's
* own price coefficient (beta_price_j varies by product since it
* incorporates the continuous interactions estimated above)
gen double delta_cf_uniform = delta_hat ///
    + beta_price_j * (price_cf_uniform - price_wmean)

* Sum exp(delta) across all products in each store-week market —
* this is the logit denominator we need to convert utilities to shares
bysort store week: egen double denom_cf_uniform = total(exp(delta_cf_uniform))

* Standard logit share formula: exp(delta_j) / sum_k(exp(delta_k))
gen double share_cf_uniform = exp(delta_cf_uniform) / denom_cf_uniform

* Scale shares back to quantities, holding market size (customer
* traffic) fixed at its baseline level
gen double q_cf_uniform = share_cf_uniform * total_q_old

* How much did each product's quantity change vs baseline?
gen double dq_uniform = q_cf_uniform - quantity_prod

* Revenue = new price × new quantity; delta is the change vs baseline
gen double rev_cf_uniform = price_cf_uniform * q_cf_uniform
gen double d_rev_uniform  = rev_cf_uniform - revenue_sum

* Aggregate to store-week for market-level reporting
bysort store week: egen double total_q_cf_uniform   = total(q_cf_uniform)
bysort store week: egen double total_rev_cf_uniform = total(rev_cf_uniform)

gen double d_total_q_uniform   = total_q_cf_uniform - total_q_old
gen double d_total_rev_uniform = total_rev_cf_uniform - total_rev_old

* Consumer welfare via the log-sum formula: the change in the log of
* the choice set's total "appeal," converted to dollars using the
* average price coefficient as the marginal utility of income.
* beta_bar should be negative (higher price = lower utility).
summ beta_price_j
local beta_bar = r(mean)

gen double welfare_uniform = ///
    (ln(denom_cf_uniform) - ln(sum_exp_old)) / (-`beta_bar')


****************************************************
* Visualization of baseline and counterfactual outcomes
****************************************************

* Five scenarios in total:
*   uniform  — flat +10% price increase across all products
*   premium  — +15% only for the top 20 UPCs by total revenue
*   size     — +15% size for products in the bottom size quartile
*   elast    — -25% price for products in the top decile of own-price elasticity
*   inst     — -25% price for instant oatmeal products specifically
local scenarios uniform premium size elast inst

* Get a sense of the revenue distribution before anything changes,
* both at the product-market level and aggregated to store-week
hist revenue_sum, percent ///
    title("Baseline Revenue per Product-Market") ///
    xlabel(, grid) ylabel(, grid)
graph export "baseline_revenue.png", replace

hist total_rev_old, percent ///
    title("Baseline Total Revenue by Store/Week") ///
    xlabel(, grid) ylabel(, grid)
graph export "baseline_total_revenue.png", replace

* For each scenario, plot the distribution of quantity changes, revenue
* changes, and welfare — useful for seeing not just the average effect
* but how much it varies across products and markets
foreach scenario of local scenarios {

    hist dq_`scenario', percent ///
        title("ΔQuantity (`scenario')") ///
        xlabel(, grid) ylabel(, grid)
    graph export "dq_`scenario'.png", replace

    hist d_rev_`scenario', percent ///
        title("ΔRevenue (`scenario')") ///
        xlabel(, grid) ylabel(, grid)
    graph export "d_rev_`scenario'.png", replace

    hist welfare_`scenario', percent ///
        title("Welfare (`scenario')") ///
        xlabel(, grid) ylabel(, grid)
    graph export "welfare_`scenario'.png", replace

    * Welfare change relative to baseline — since we're computing
    * the log-sum difference directly, this is already a change measure,
    * but naming it d_welfare makes that framing explicit in the output
    gen double d_welfare_`scenario' = welfare_`scenario'
    hist d_welfare_`scenario', percent ///
        title("ΔWelfare Change (`scenario')") ///
        xlabel(, grid) ylabel(, grid)
    graph export "d_welfare_`scenario'.png", replace
}
