source("scripts/bmjgh_figures/supplement_style.R")

nmb_colours <- c('Best standard'='#747678','Standalone nasal'='#0072B2','Nasal with Tdap'='#D55E00')
su <- read.csv("source_data/figures/figureS24.csv")
su$curve <- factor(su$curve,levels=names(nmb_colours))
facets <- as.vector(t(outer(gsub('_',' ',figure_countries),c('20 years','30 years'),paste,sep=' | ')))
su$setting_horizon <- factor(paste(gsub('_',' ',su$country),paste(su$horizon,'years'),sep=' | '),levels=facets)
s <- ggplot(su,aes(price,expected_incremental_nmb/1e6,colour=curve,linetype=curve))+
  geom_line(linewidth=.4)+facet_wrap(~setting_horizon,ncol=4,scales='free')+
  scale_colour_manual(values=nmb_colours,name=NULL)+
  scale_linetype_manual(values=c('Best standard'='dashed','Standalone nasal'='solid','Nasal with Tdap'='solid'),name=NULL)+
  scale_x_continuous(breaks=scales::breaks_pretty(n=3),labels=scales::label_number(accuracy=NULL),expand=expansion(mult=c(.02,.03)))+
  scale_y_continuous(breaks=scales::breaks_pretty(n=3),labels=scales::label_number(accuracy=NULL),expand=expansion(mult=c(.06,.08)))+
  labs(x='Delivered nasal price (2024 Int$/dose)',y='Expected incremental NMB (million Int$)')+
  theme(strip.text=element_text(size=6),panel.spacing=grid::unit(4,'mm'),legend.position='bottom')

export_supplement(s, "S24", 183, 240)
