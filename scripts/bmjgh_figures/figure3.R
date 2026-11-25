source("scripts/bmjgh_figures/maternal_panels.R")
caption <- paste(
  "**Figure 3. Additional infant protection across maternal vaccination coverage.**",
  "Points show mean discounted infant years of life lost (YLL) averted by adding standalone intranasal vaccination at 90% eligible adolescent uptake with three-year mean protection (reference profile 05). Each comparison holds maternal coverage at the plotted level and adolescent Tdap coverage at its setting-specific baseline.",
  "Lines connect evaluated maternal coverage levels. The 20- and 30-year estimates are slightly offset. Intervals span the central 95% of 1,000 paired clinical-risk draws conditional on fitted transmission parameters.",
  "Positive values indicate infant YLL averted. Axis ranges vary by setting, and setting titles give baseline maternal coverage. Panel G labels Brazil’s near-zero 30-year means and intervals."
)
draw_maternal_coverage(3, "standalone", caption, ncol = 4, height = 110, pretty_axes = TRUE)
