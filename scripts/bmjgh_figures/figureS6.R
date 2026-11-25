source("scripts/bmjgh_figures/supplement_style.R")

time <- read.csv("source_data/figures/figureS6.csv")
annual_series <- do.call(rbind,lapply(c('current','nasal'),function(k) {
 d <- time;d$programme <- if(k=='current') 'Current provision' else 'Standalone nasal';d$cases <- d[[paste0(k,'_reported_cases')]];d
}))
annual_series$programme <- factor(annual_series$programme,levels=c('Current provision','Standalone nasal'))
series_cols <- c('Current provision'='#777777','Standalone nasal'='#0072B2')
annual_panel <- function(country) ggplot(annual_series[annual_series$country==country,],aes(year,cases,colour=programme,linetype=programme))+
 geom_vline(xintercept=2046,colour='#AAAAAA',linewidth=.25,linetype='dotted')+
 geom_line(linewidth=.45)+scale_colour_manual(values=series_cols,name=NULL)+
 scale_linetype_manual(values=c('solid','dashed'),name=NULL)+
 scale_x_continuous(breaks=c(2027,2036,2046,2056),expand=expansion(mult=c(.02,.02)))+
 scale_y_continuous(labels=scales::label_number(scale_cut=scales::cut_short_scale()),expand=expansion(mult=c(.02,.08)))+
 facet_wrap(~country) + labs(x=NULL,y='Annual reported cases')
cumulative_panel <- function(country) ggplot(time[time$country==country,],aes(year,cumulative_reported_cases_averted))+
 geom_hline(yintercept=0,colour='#777777',linewidth=.25)+
 geom_vline(xintercept=2046,colour='#AAAAAA',linewidth=.25,linetype='dotted')+
 geom_line(colour='#0072B2',linewidth=.5)+
 geom_point(data=time[time$country==country & time$year %in% c(2046,2056),],size=1.6,colour='#0072B2')+
 scale_x_continuous(breaks=c(2027,2036,2046,2056),expand=expansion(mult=c(.02,.02)))+
 scale_y_continuous(breaks=scales::breaks_pretty(n=5),labels=scales::label_number(scale_cut=scales::cut_short_scale()),expand=expansion(mult=c(.10,.10)))+
 labs(x='Calendar year',y='Cumulative reported cases averted')
ta <- annual_panel('Brazil');tb <- annual_panel('Japan');tc <- cumulative_panel('Brazil');td <- cumulative_panel('Japan')
tp <- (ta+tb)/(tc+td)+plot_layout(guides='collect')+plot_annotation(tag_levels='a',
 caption='Dotted lines mark 2046, the 20-year endpoint. Positive cumulative differences indicate fewer cases with nasal vaccination.',
 theme=theme(plot.caption=element_text(size=7,hjust=0))) & theme(legend.position='bottom')

export_supplement(tp, "S6", 183, 150)
