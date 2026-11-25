# Mechanisms in supplementary Table S8.
source("scripts/bmjgh_figures/supplement_style.R")
blue <- "#0072B2"
orange <- "#D55E00"
arrow <- grid::arrow(length=grid::unit(1.6, 'mm'), type='closed')
box <- function(xmin, xmax, ymin, ymax, fill='#F5F6F7', colour='#747678') {
  annotate('rect', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
           fill=fill, colour=colour, linewidth=.3)
}
label <- function(x, y, text, colour='#222222', size=8, parse=FALSE) {
  annotate('text', x=x, y=y, label=text, colour=colour,
           size=size/.pt, lineheight=1.12, parse=parse)
}
link <- function(x, y, xend, yend, colour='#747678') {
  annotate('segment', x=x, y=y, xend=xend, yend=yend,
           colour=colour, linewidth=.35, arrow=arrow)
}

p <- ggplot() +
  box(2, 25, 38, 49) + label(13.5, 43.5, 'Infectious sources\nTypical + mild') +
  box(2, 25, 54, 62, '#FFF3EC', orange) +
  label(13.5, 58, 'Onward multiplier', orange) +
  link(13.5, 54, 13.5, 49, orange) + link(25, 43.5, 32, 43.5) +
  label(47, 44, 'J[h] == frac(sum(v[j]*(I[tj]+rho*I[mj]), j==0, 2), N[h])', parse=TRUE, size=9) +
  label(47, 32, 'Full contact-group population', size=7) +
  label(47, 27, 'All states and nasal layers', size=7) +
  link(62, 43.5, 67, 43.5) +
  box(67, 81, 38, 49) + label(74, 43.5, 'Contacts\nSeasonality') +
  link(81, 43.5, 87, 43.5) +
  box(87, 98, 38, 49) + label(92.5, 43.5, 'lambda[a]', parse=TRUE) +
  label(92.5, 58, 'Importation', size=7) + link(92.5, 54, 92.5, 49) +
  box(2, 16, 8, 19) + label(9, 13.5, 'Exposure') +
  label(9, 5, 'lambda[a]', parse=TRUE, size=8) +
  link(16, 13.5, 23, 13.5) +
  box(23, 48, 8, 19, '#EEF6FC', blue) +
  label(35.5, 13.5, 'u[j]*sigma[X]*lambda[a]*X[aj]', parse=TRUE) +
  label(35.5, 5, 'Infection inflow', size=7) +
  label(35.5, 24, 'Acquisition multiplier', blue) +
  link(35.5, 21.5, 35.5, 19, blue) +
  link(48, 13.5, 54, 13.5) +
  box(54, 65, 8, 19) + label(59.5, 13.5, 'Latent\ninfection') +
  link(65, 13.5, 71, 13.5) +
  box(71, 84, 8, 19) + label(77.5, 13.5, 'Typical / mild\ninfection') +
  link(84, 13.5, 90, 13.5) +
  box(90, 99, 8, 19) + label(94.5, 13.5, 'Recovery', size=7) +
  coord_cartesian(xlim=c(0, 100), ylim=c(0, 65), expand=FALSE) +
  theme_void(base_family='sans') + theme(plot.margin=margin(2, 2, 2, 2, 'mm'))

export_supplement(p, "S8_table_schematic", 183, 105)
