source("scripts/bmjgh_figures/supplement_style.R")

dur <- read.csv("source_data/figures/figureS25.csv")
dur$country_label <- factor(dur$country_label, levels=country_levels)
dur$horizon_label <- factor(dur$horizon,levels=c(20,30),labels=c("20-year evaluation","30-year evaluation"))
duration_plot <- ggplot(dur,aes(mean_years,maximum_nasal_price_vs_best_standard,
                                colour=horizon_label,shape=horizon_label)) +
  geom_hline(yintercept=0,linewidth=.2,colour="#888888",linetype="dashed") +
  geom_line(linewidth=.4)+geom_point(size=1.5,stroke=.35) +
  facet_wrap(~country_label,ncol=4,scales="free_y") +
  scale_colour_manual(values=c("#0072B2","#D55E00"),name=NULL) +
  scale_shape_manual(values=c(16,17),name=NULL) +
  scale_x_continuous(breaks=c(1,3,6,10),expand=expansion(mult=c(.08,.08))) +
  scale_y_continuous(expand=expansion(mult=c(.1,.1))) +
  labs(x="Mean protection duration (years)",y="Maximum nasal cost per dose (Int$2024)") +
  theme(panel.spacing=grid::unit(5,"mm"))

export_supplement(duration_plot, "S25", 183, 118)
