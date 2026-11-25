source("scripts/bmjgh_figures/maternal_panels.R")
caption <- paste("**Figure S3. Additional infant protection from coadministration across maternal vaccination coverage.**",
  "Each point gives the mean discounted infant years of life lost (YLL) averted by adding the intranasal vaccine to a 90% adolescent Tdap programme at the maternal coverage indicated. Both strategies use the same maternal and adolescent Tdap coverage. The intranasal vaccine uses reference profile 05 with three-year mean protection.",
  "Lines connect evaluated coverages. The two horizons are offset slightly at each coverage to show both intervals. Vertical intervals give the 2.5th and 97.5th percentiles of 1,000 paired clinical-risk draws at fixed transmission parameters. Axis ranges vary by setting.")
draw_maternal_coverage("S3", "coadmin", caption)
